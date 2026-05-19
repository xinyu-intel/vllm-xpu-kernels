#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T>
struct LocalUnpermuteCopyScalarKernel {
  const T* expert_output_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  T* output_ptr;
  int32_t token_count;
  int32_t hidden_size;
  int32_t topk;
  int32_t num_tokens;
  int32_t token_offset;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = token_count * hidden_size;
    if (idx >= total) return;

    const int32_t h = idx % hidden_size;
    const int32_t local_idx = idx / hidden_size;
    const int32_t global_token_idx = token_offset + local_idx;

    float acc = 0.0f;
    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t src_row = scatter_idx_ptr[scatter_base + k];
      const float weight = topk_weights_ptr[scatter_base + k];
      acc +=
          weight * static_cast<float>(
                       expert_output_ptr
                           [static_cast<int64_t>(src_row) * hidden_size + h]);
    }
    output_ptr[static_cast<int64_t>(local_idx) * hidden_size + h] =
        static_cast<T>(acc);
  }
};

template <typename scalar_t, int VEC_SIZE>
struct LocalUnpermuteCopyVecKernel {
  const scalar_t* expert_output_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  scalar_t* output_ptr;
  int32_t token_count;
  int32_t hidden_size;
  int32_t topk;
  int32_t token_offset;
  int32_t hidden_vecs;
  int32_t hidden_vecs_mask;
  int32_t hidden_vecs_shift;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = token_count * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx & hidden_vecs_mask;
    const int32_t local_idx = idx >> hidden_vecs_shift;
    const int32_t global_token_idx = token_offset + local_idx;
    const int32_t h_start = vec_h * VEC_SIZE;

    float acc[VEC_SIZE] = {};

    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t src_row = scatter_idx_ptr[scatter_base + k];
      const float weight = topk_weights_ptr[scatter_base + k];
      const scalar_t* src = expert_output_ptr +
                            static_cast<int64_t>(src_row) * hidden_size +
                            h_start;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] += weight * static_cast<float>(src[i]);
      }
    }

    scalar_t* dst =
        output_ptr + static_cast<int64_t>(local_idx) * hidden_size + h_start;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      dst[i] = static_cast<scalar_t>(acc[i]);
    }
  }
};

template <typename T>
struct UnpermuteReduceScatterScalarKernel {
  const int64_t* rank_ptrs;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  T* output_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t local_token_idx = idx / hidden_size;
    const int64_t global_token_idx =
        rank * num_tokens_per_rank + local_token_idx;

    float acc = 0.0f;
    const int64_t scatter_base = global_token_idx * topk;
    for (int64_t r = 0; r < world_size; ++r) {
      const T* expert_out = reinterpret_cast<const T*>(rank_ptrs[r]);
      for (int64_t k = 0; k < topk; ++k) {
        const int32_t src_row = scatter_idx_ptr[scatter_base + k];
        const float weight = topk_weights_ptr[scatter_base + k];
        acc += weight *
               static_cast<float>(
                   expert_out[static_cast<int64_t>(src_row) * hidden_size + h]);
      }
    }
    output_ptr[local_token_idx * hidden_size + h] = static_cast<T>(acc);
  }
};

template <typename scalar_t, int VEC_SIZE>
struct UnpermuteReduceScatterVecKernel {
  const int64_t* rank_ptrs;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  scalar_t* output_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t local_token_idx = idx / hidden_vecs;
    const int32_t global_token_idx =
        rank * num_tokens_per_rank + local_token_idx;
    const int32_t h_start = vec_h * VEC_SIZE;

    float acc[VEC_SIZE] = {};

    const int64_t scatter_base = static_cast<int64_t>(global_token_idx) * topk;
    for (int32_t r = 0; r < world_size; ++r) {
      const scalar_t* expert_out =
          reinterpret_cast<const scalar_t*>(rank_ptrs[r]);
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t src_row = scatter_idx_ptr[scatter_base + k];
        const float weight = topk_weights_ptr[scatter_base + k];
        const scalar_t* src =
            expert_out + static_cast<int64_t>(src_row) * hidden_size + h_start;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src[i]);
        }
      }
    }

    scalar_t* dst = output_ptr +
                    static_cast<int64_t>(local_token_idx) * hidden_size +
                    h_start;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      dst[i] = static_cast<scalar_t>(acc[i]);
    }
  }
};

}  // namespace moe
}  // namespace vllm

static int32_t log2_po2_urs(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

at::Tensor local_unpermute_copy(
    const at::Tensor& expert_output,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    int64_t token_offset,
    int64_t token_count,
    at::Tensor output) {
  TORCH_CHECK(
      expert_output.dim() == 2,
      "local_unpermute_copy: expert_output must be 2D");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "local_unpermute_copy: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "local_unpermute_copy: scatter_idx must be int32");
  TORCH_CHECK(
      topk_weights.dim() == 2,
      "local_unpermute_copy: topk_weights must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "local_unpermute_copy: topk_weights must be float32");
  TORCH_CHECK(expert_output.is_contiguous());
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(output.dim() == 2);

  const int64_t hidden_size = expert_output.size(1);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);

  TORCH_CHECK(token_offset >= 0 && token_offset + token_count <= num_tokens);
  TORCH_CHECK(output.size(0) == token_count);
  TORCH_CHECK(output.size(1) == hidden_size);
  TORCH_CHECK(
      topk_weights.size(0) == num_tokens && topk_weights.size(1) == topk);

  if (token_count == 0) return output;

  const at::DeviceGuard device_guard(expert_output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  VLLM_DISPATCH_FLOATING_TYPES(
      expert_output.scalar_type(), "local_unpermute_copy", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const int32_t tc = static_cast<int32_t>(token_count);
        const int32_t h = static_cast<int32_t>(hidden_size);
        const int32_t tk = static_cast<int32_t>(topk);
        const int32_t off = static_cast<int32_t>(token_offset);

        if (hidden_size % VEC_SIZE == 0) {
          const int32_t hidden_vecs = h / VEC_SIZE;
          const int32_t hv_mask = hidden_vecs - 1;
          const int32_t hv_shift = log2_po2_urs(hidden_vecs);
          const int64_t total = static_cast<int64_t>(tc) * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::LocalUnpermuteCopyVecKernel<sycl_t, VEC_SIZE>{
              reinterpret_cast<const sycl_t*>(
                  expert_output.data_ptr<scalar_t>()),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
              tc,
              h,
              tk,
              off,
              hidden_vecs,
              hv_mask,
              hv_shift};
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                kfn);
          });
        } else {
          const int64_t total = static_cast<int64_t>(tc) * h;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::LocalUnpermuteCopyScalarKernel<sycl_t>{
              reinterpret_cast<const sycl_t*>(
                  expert_output.data_ptr<scalar_t>()),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
              tc,
              h,
              tk,
              static_cast<int32_t>(num_tokens),
              off};
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

at::Tensor unpermute_reduce_scatter(
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    at::Tensor output,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "unpermute_reduce_scatter: rank_buffers_ptr must be 1D with size == "
      "world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "unpermute_reduce_scatter: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "unpermute_reduce_scatter: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "unpermute_reduce_scatter: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(
      topk_weights.dim() == 2,
      "unpermute_reduce_scatter: topk_weights must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "unpermute_reduce_scatter: topk_weights must be float32");
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(output.dim() == 2, "unpermute_reduce_scatter: output must be 2D");
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size);

  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);
  const int64_t hidden_size = output.size(1);

  TORCH_CHECK(num_tokens % world_size == 0);
  const int64_t num_tokens_per_rank = num_tokens / world_size;
  TORCH_CHECK(output.size(0) == num_tokens_per_rank);
  TORCH_CHECK(
      topk_weights.size(0) == num_tokens && topk_weights.size(1) == topk);

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return output;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "unpermute_reduce_scatter", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn =
              vllm::moe::UnpermuteReduceScatterVecKernel<sycl_t, VEC_SIZE>{
                  rank_buffers_ptr.data_ptr<int64_t>(),
                  scatter_idx.data_ptr<int32_t>(),
                  topk_weights.data_ptr<float>(),
                  reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
                  static_cast<int32_t>(num_tokens_per_rank),
                  static_cast<int32_t>(hidden_size),
                  static_cast<int32_t>(topk),
                  static_cast<int32_t>(rank),
                  static_cast<int32_t>(world_size),
                  static_cast<int32_t>(hidden_vecs)};
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                kfn);
          });
        } else {
          const int64_t total = num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::UnpermuteReduceScatterScalarKernel<sycl_t>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
              num_tokens_per_rank,
              hidden_size,
              topk,
              rank,
              world_size};
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
