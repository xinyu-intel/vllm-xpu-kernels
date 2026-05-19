#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T>
struct EpDispatchRingScalarKernel {
  const int64_t* rank_ptrs;
  const int64_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  T* remap_ptr;
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

    const int64_t src_rank = (rank + step + 1) % world_size;
    const int64_t global_token_idx =
        src_rank * num_tokens_per_rank + local_token_idx;

    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const T val = src[local_token_idx * hidden_size + h];

    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert =
          static_cast<int32_t>(topk_idx_ptr[global_token_idx * topk + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) {
        const int64_t dst_row =
            static_cast<int64_t>(scatter_idx_ptr[global_token_idx * topk + k]);
        remap_ptr[dst_row * hidden_size + h] = val;
      }
    }
  }
};

template <typename scalar_t, int VEC_SIZE>
struct EpDispatchRingVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  const int64_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  scalar_t* remap_ptr;
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

    const int32_t src_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx =
        src_rank * num_tokens_per_rank + local_token_idx;

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

    const scalar_t* src =
        reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        const int32_t dst_row =
            scatter_idx_ptr[static_cast<int64_t>(global_token_idx) * topk + k];
        auto dst_vec = reinterpret_cast<vec_t*>(
            remap_ptr + static_cast<int64_t>(dst_row) * hidden_size);
        dst_vec[vec_h] = v;
      }
    }
  }
};

}  // namespace moe
}  // namespace vllm

at::Tensor ep_dispatch(
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "ep_dispatch: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "ep_dispatch: rank_buffers_ptr must be int64");
  TORCH_CHECK(topk_idx.dim() == 2, "ep_dispatch: topk_idx must be 2D");
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.size(0) == topk_idx.size(0) &&
          scatter_idx.size(1) == topk_idx.size(1),
      "ep_dispatch: scatter_idx must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "ep_dispatch: scatter_idx must be int32");
  TORCH_CHECK(
      remap_hidden_states.dim() == 2,
      "ep_dispatch: remap_hidden_states must be 2D");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "ep_dispatch: remap_hidden_states must be contiguous");

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = remap_hidden_states.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "ep_dispatch: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "ep_dispatch: remap_hidden_states first dim mismatch");

  const int64_t total_elems = num_tokens * topk * hidden_size;
  if (total_elems == 0) {
    return remap_hidden_states;
  }

  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  const at::DeviceGuard device_guard(remap_hidden_states.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      remap_hidden_states.scalar_type(), "ep_dispatch", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::EpDispatchRingVecKernel<sycl_t, VEC_SIZE>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              topk_idx.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
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
          auto kfn = vllm::moe::EpDispatchRingScalarKernel<sycl_t>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              topk_idx.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
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

  return remap_hidden_states;
}
