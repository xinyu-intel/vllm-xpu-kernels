#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T>
struct EpCombineRingScalarKernel {
  const T* expert_output_ptr;
  const int64_t* rank_output_ptrs;
  const int64_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t rank;
  int64_t world_size;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    const int64_t target_rank = (rank + step + 1) % world_size;
    const int64_t global_token_idx =
        target_rank * num_tokens_per_rank + local_token_idx;

    const int64_t topk_base = global_token_idx * topk;
    bool has_owned = false;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) {
        has_owned = true;
        break;
      }
    }
    if (!has_owned) return;

    float acc = 0.0f;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) {
        const float weight = topk_weights_ptr[topk_base + k];
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        acc +=
            weight * static_cast<float>(
                         expert_output_ptr
                             [static_cast<int64_t>(src_row) * hidden_size + h]);
      }
    }

    T* target_buf = reinterpret_cast<T*>(rank_output_ptrs[target_rank]);
    const int64_t dst_offset =
        (rank * num_tokens_per_rank + local_token_idx) * hidden_size + h;
    target_buf[dst_offset] = static_cast<T>(acc);
  }
};

template <typename scalar_t, int VEC_SIZE>
struct EpCombineRingVecKernel {
  const scalar_t* expert_output_ptr;
  const int64_t* rank_output_ptrs;
  const int64_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * world_size * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t pair_idx = idx / hidden_vecs;
    const int32_t step = pair_idx % world_size;
    const int32_t local_token_idx = pair_idx / world_size;

    const int32_t target_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx =
        target_rank * num_tokens_per_rank + local_token_idx;
    const int32_t h_start = vec_h * VEC_SIZE;

    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
    bool has_owned = false;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        has_owned = true;
        break;
      }
    }
    if (!has_owned) return;

    float acc[VEC_SIZE] = {};
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        const float weight = topk_weights_ptr[topk_base + k];
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        const scalar_t* src = expert_output_ptr +
                              static_cast<int64_t>(src_row) * hidden_size +
                              h_start;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src[i]);
        }
      }
    }

    scalar_t* target_buf =
        reinterpret_cast<scalar_t*>(rank_output_ptrs[target_rank]);
    scalar_t* dst =
        target_buf +
        (static_cast<int64_t>(rank) * num_tokens_per_rank + local_token_idx) *
            hidden_size +
        h_start;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      dst[i] = static_cast<scalar_t>(acc[i]);
    }
  }
};

}  // namespace moe
}  // namespace vllm

at::Tensor ep_combine(
    const at::Tensor& expert_output,
    const at::Tensor& rank_output_ptrs,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    at::Tensor output,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_output_ptrs.dim() == 1 && rank_output_ptrs.size(0) == world_size,
      "ep_combine: rank_output_ptrs must be 1D with size == world_size");
  TORCH_CHECK(
      rank_output_ptrs.scalar_type() == at::kLong,
      "ep_combine: rank_output_ptrs must be int64");
  TORCH_CHECK(
      expert_output.dim() == 2,
      "ep_combine: expert_output must be 2D [num_tokens * topk, hidden]");
  TORCH_CHECK(expert_output.is_contiguous());
  TORCH_CHECK(topk_idx.dim() == 2, "ep_combine: topk_idx must be 2D");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kLong,
      "ep_combine: topk_idx must be int64");
  TORCH_CHECK(topk_idx.is_contiguous());
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.size(0) == topk_idx.size(0) &&
          scatter_idx.size(1) == topk_idx.size(1),
      "ep_combine: scatter_idx must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "ep_combine: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(
      topk_weights.dim() == 2 && topk_weights.size(0) == topk_idx.size(0) &&
          topk_weights.size(1) == topk_idx.size(1),
      "ep_combine: topk_weights must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "ep_combine: topk_weights must be float32");
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(output.dim() == 2, "ep_combine: output must be 2D");
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size);

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = expert_output.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "ep_combine: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      output.size(0) == num_tokens_per_rank,
      "ep_combine: output first dim must be num_tokens_per_rank");
  TORCH_CHECK(
      output.size(1) == hidden_size,
      "ep_combine: output hidden size must match expert_output");

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return output;
  }

  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(output.scalar_type(), "ep_combine", [&]() {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    if (hidden_size % VEC_SIZE == 0) {
      const int64_t hidden_vecs = hidden_size / VEC_SIZE;
      const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
      const int64_t blocks = (total + threads - 1) / threads;
      auto kfn = vllm::moe::EpCombineRingVecKernel<sycl_t, VEC_SIZE>{
          reinterpret_cast<const sycl_t*>(expert_output.data_ptr<scalar_t>()),
          rank_output_ptrs.data_ptr<int64_t>(),
          topk_idx.data_ptr<int64_t>(),
          scatter_idx.data_ptr<int32_t>(),
          topk_weights.data_ptr<float>(),
          static_cast<int32_t>(num_tokens_per_rank),
          static_cast<int32_t>(hidden_size),
          static_cast<int32_t>(topk),
          static_cast<int32_t>(rank),
          static_cast<int32_t>(world_size),
          static_cast<int32_t>(hidden_vecs),
          base_experts,
          rem_experts,
          boundary};
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
            kfn);
      });
    } else {
      const int64_t total = world_size * num_tokens_per_rank * hidden_size;
      const int64_t blocks = (total + threads - 1) / threads;
      auto kfn = vllm::moe::EpCombineRingScalarKernel<sycl_t>{
          reinterpret_cast<const sycl_t*>(expert_output.data_ptr<scalar_t>()),
          rank_output_ptrs.data_ptr<int64_t>(),
          topk_idx.data_ptr<int64_t>(),
          scatter_idx.data_ptr<int32_t>(),
          topk_weights.data_ptr<float>(),
          num_tokens_per_rank,
          hidden_size,
          topk,
          rank,
          world_size,
          base_experts,
          rem_experts,
          boundary};
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
            kfn);
      });
    }
  });

  return output;
}
