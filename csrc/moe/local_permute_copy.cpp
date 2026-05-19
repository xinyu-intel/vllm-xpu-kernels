#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T>
struct LocalPermuteCopyScalarKernel {
  const T* src_ptr;
  T* dst_ptr;
  const int32_t* scatter_idx_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t remote_token_offset;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int32_t h = idx % hidden_size;
    const int32_t local_token_idx = idx / hidden_size;
    const int32_t global_token_idx = remote_token_offset + local_token_idx;
    const T val = src_ptr[local_token_idx * hidden_size + h];
    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      int32_t dst_row = scatter_idx_ptr[scatter_base + k];
      dst_ptr[dst_row * hidden_size + h] = val;
    }
  }
};

template <typename scalar_t, int VEC_SIZE>
struct LocalPermuteCopyVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* src_ptr;
  scalar_t* dst_ptr;
  const int32_t* scatter_idx_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t remote_token_offset;
  int32_t hidden_vecs;
  int32_t hidden_vecs_mask;
  int32_t hidden_vecs_shift;
  int32_t total;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    if (idx >= total) return;

    const int32_t vec_h = idx & hidden_vecs_mask;
    const int32_t local_token_idx = idx >> hidden_vecs_shift;
    const int32_t global_token_idx = remote_token_offset + local_token_idx;

    auto src_vec =
        reinterpret_cast<const vec_t*>(src_ptr + local_token_idx * hidden_size);
    vec_t v = src_vec[vec_h];

    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      int32_t dst_row = scatter_idx_ptr[scatter_base + k];
      auto dst_vec = reinterpret_cast<vec_t*>(dst_ptr + dst_row * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

template <typename scalar_t, int VEC_SIZE>
struct LocalPermuteCopyFusedKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* src_ptr;
  scalar_t* dst_ptr;
  const int32_t* scatter_idx_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t hidden_vecs;
  int32_t hidden_vecs_mask;
  int32_t hidden_vecs_shift;
  int32_t tokens_per_rank_mask;
  int32_t tokens_per_rank_shift;
  int32_t total;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    if (idx >= total) return;

    const int32_t vec_h = idx & hidden_vecs_mask;
    const int32_t rank_and_token = idx >> hidden_vecs_shift;
    const int32_t local_token_idx = rank_and_token & tokens_per_rank_mask;
    const int32_t src_rank = rank_and_token >> tokens_per_rank_shift;

    const int32_t global_token_idx =
        src_rank * num_tokens_per_rank + local_token_idx;

    const int32_t src_offset =
        (src_rank * num_tokens_per_rank + local_token_idx) * hidden_size;
    auto src_vec = reinterpret_cast<const vec_t*>(src_ptr + src_offset);
    vec_t v = src_vec[vec_h];

    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t k = 0; k < topk; ++k) {
      int32_t dst_row = scatter_idx_ptr[scatter_base + k];
      auto dst_vec = reinterpret_cast<vec_t*>(dst_ptr + dst_row * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

template <typename T>
struct AllgatherPermuteRingScalarKernel {
  const int64_t* rank_ptrs;
  const int32_t* scatter_idx_ptr;
  T* remap_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t rank;
  int64_t world_size;

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
      const int32_t dst_row = scatter_idx_ptr[global_token_idx * topk + k];
      remap_ptr[static_cast<int64_t>(dst_row) * hidden_size + h] = val;
    }
  }
};

template <typename scalar_t, int VEC_SIZE>
struct AllgatherPermuteRingVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  const int32_t* scatter_idx_ptr;
  scalar_t* remap_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;

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

    const scalar_t* src =
        reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t dst_row = scatter_idx_ptr[topk_base + k];
      auto dst_vec = reinterpret_cast<vec_t*>(
          remap_ptr + static_cast<int64_t>(dst_row) * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

template <typename T>
struct AllgatherRingScalarKernel {
  const int64_t* rank_ptrs;
  T* dst_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    const int64_t src_rank = (rank + step + 1) % world_size;

    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const T val = src[local_token_idx * hidden_size + h];

    const int64_t dst_row = src_rank * num_tokens_per_rank + local_token_idx;
    dst_ptr[dst_row * hidden_size + h] = val;
  }
};

template <typename scalar_t, int VEC_SIZE>
struct AllgatherRingVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  scalar_t* dst_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * world_size * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t pair_idx = idx / hidden_vecs;
    const int32_t step = pair_idx % world_size;
    const int32_t local_token_idx = pair_idx / world_size;

    const int32_t src_rank = (rank + step + 1) % world_size;

    const scalar_t* src =
        reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    const int32_t dst_row = src_rank * num_tokens_per_rank + local_token_idx;
    auto dst_vec = reinterpret_cast<vec_t*>(
        dst_ptr + static_cast<int64_t>(dst_row) * hidden_size);
    dst_vec[vec_h] = v;
  }
};

}  // namespace moe
}  // namespace vllm

static int32_t log2_po2(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

at::Tensor local_permute_copy(
    const at::Tensor& src_hidden,
    const at::Tensor& scatter_idx,
    int64_t remote_token_offset,
    at::Tensor remap_hidden_states) {
  TORCH_CHECK(
      src_hidden.dim() == 2, "local_permute_copy: src_hidden must be 2D");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "local_permute_copy: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "local_permute_copy: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "local_permute_copy: scatter_idx must be contiguous");
  TORCH_CHECK(
      src_hidden.scalar_type() == remap_hidden_states.scalar_type(),
      "local_permute_copy: src and remap dtype must match");
  TORCH_CHECK(
      src_hidden.is_contiguous(),
      "local_permute_copy: src_hidden must be contiguous");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "local_permute_copy: remap_hidden_states must be contiguous");

  const int64_t num_tokens_per_rank = src_hidden.size(0);
  const int64_t hidden_size = src_hidden.size(1);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);

  TORCH_CHECK(
      remote_token_offset >= 0,
      "local_permute_copy: remote_token_offset must be >= 0");
  TORCH_CHECK(
      remote_token_offset + num_tokens_per_rank <= num_tokens,
      "local_permute_copy: remote token range out of bounds");
  TORCH_CHECK(
      remap_hidden_states.size(1) == hidden_size,
      "local_permute_copy: remap_hidden_states hidden size mismatch");

  if (num_tokens_per_rank == 0) {
    return remap_hidden_states;
  }

  TORCH_CHECK(
      remap_hidden_states.size(0) * hidden_size <= INT32_MAX,
      "local_permute_copy: total output elements exceed int32 range");

  const at::DeviceGuard device_guard(src_hidden.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 128;

  VLLM_DISPATCH_FLOATING_TYPES(
      src_hidden.scalar_type(), "local_permute_copy", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const int32_t n = static_cast<int32_t>(num_tokens_per_rank);
        const int32_t h = static_cast<int32_t>(hidden_size);
        const int32_t tk = static_cast<int32_t>(topk);
        const int32_t off = static_cast<int32_t>(remote_token_offset);

        if (hidden_size % VEC_SIZE == 0) {
          const int32_t hidden_vecs = h / VEC_SIZE;
          const int32_t hv_mask = hidden_vecs - 1;
          const int32_t hv_shift = log2_po2(hidden_vecs);
          const int64_t total = static_cast<int64_t>(n) * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::LocalPermuteCopyVecKernel<sycl_t, VEC_SIZE>{
              reinterpret_cast<const sycl_t*>(src_hidden.data_ptr<scalar_t>()),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
              scatter_idx.data_ptr<int32_t>(),
              n,
              h,
              tk,
              off,
              hidden_vecs,
              hv_mask,
              hv_shift,
              static_cast<int32_t>(total)};
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                kfn);
          });
        } else {
          const int64_t total = static_cast<int64_t>(n) * h;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::LocalPermuteCopyScalarKernel<sycl_t>{
              reinterpret_cast<const sycl_t*>(src_hidden.data_ptr<scalar_t>()),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
              scatter_idx.data_ptr<int32_t>(),
              n,
              h,
              tk,
              off};
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

at::Tensor local_permute_copy_fused(
    const at::Tensor& src_all,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states) {
  TORCH_CHECK(
      src_all.dim() == 3,
      "local_permute_copy_fused: src_all must be 3D [world_size, tokens, "
      "hidden]");
  TORCH_CHECK(
      src_all.is_contiguous(),
      "local_permute_copy_fused: src_all must be contiguous");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "local_permute_copy_fused: remap must be contiguous");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "local_permute_copy_fused: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "local_permute_copy_fused: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "local_permute_copy_fused: scatter_idx must be contiguous");

  const int64_t world_size = src_all.size(0);
  const int64_t num_tokens_per_rank = src_all.size(1);
  const int64_t hidden_size = src_all.size(2);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);

  TORCH_CHECK(num_tokens == world_size * num_tokens_per_rank);
  TORCH_CHECK(remap_hidden_states.size(1) == hidden_size);

  TORCH_CHECK(
      (num_tokens_per_rank & (num_tokens_per_rank - 1)) == 0,
      "local_permute_copy_fused: num_tokens_per_rank must be power of 2");
  TORCH_CHECK(remap_hidden_states.size(0) * hidden_size <= INT32_MAX);

  if (num_tokens_per_rank == 0) {
    return remap_hidden_states;
  }

  const at::DeviceGuard device_guard(src_all.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  VLLM_DISPATCH_FLOATING_TYPES(
      src_all.scalar_type(), "local_permute_copy_fused", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        const int32_t n = static_cast<int32_t>(num_tokens_per_rank);
        const int32_t h = static_cast<int32_t>(hidden_size);
        const int32_t tk = static_cast<int32_t>(topk);

        if (hidden_size % VEC_SIZE == 0) {
          const int32_t hidden_vecs = h / VEC_SIZE;
          const int32_t hv_mask = hidden_vecs - 1;
          const int32_t hv_shift = log2_po2(hidden_vecs);
          const int32_t tpr_mask = n - 1;
          const int32_t tpr_shift = log2_po2(n);
          const int32_t total =
              static_cast<int32_t>(world_size) * n * hidden_vecs;
          const int64_t blocks =
              (static_cast<int64_t>(total) + threads - 1) / threads;

          auto kfn = vllm::moe::LocalPermuteCopyFusedKernel<sycl_t, VEC_SIZE>{
              reinterpret_cast<const sycl_t*>(src_all.data_ptr<scalar_t>()),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
              scatter_idx.data_ptr<int32_t>(),
              n,
              h,
              tk,
              hidden_vecs,
              hv_mask,
              hv_shift,
              tpr_mask,
              tpr_shift,
              total};
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                kfn);
          });
        } else {
          for (int64_t r = 0; r < world_size; ++r) {
            local_permute_copy(
                src_all[r],
                scatter_idx,
                r * num_tokens_per_rank,
                remap_hidden_states);
          }
        }
      });

  return remap_hidden_states;
}

at::Tensor allgather_permute(
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather_permute: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "allgather_permute: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "allgather_permute: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "allgather_permute: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "allgather_permute: scatter_idx must be contiguous");
  TORCH_CHECK(
      remap_hidden_states.dim() == 2,
      "allgather_permute: remap_hidden_states must be 2D");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "allgather_permute: remap_hidden_states must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather_permute: rank must be in [0, world_size)");

  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);
  const int64_t hidden_size = remap_hidden_states.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "allgather_permute: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "allgather_permute: remap_hidden_states first dim must be num_tokens * "
      "topk");

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return remap_hidden_states;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  const at::DeviceGuard device_guard(remap_hidden_states.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      remap_hidden_states.scalar_type(), "allgather_permute", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::AllgatherPermuteRingVecKernel<sycl_t, VEC_SIZE>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
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
          const int64_t total = world_size * num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::AllgatherPermuteRingScalarKernel<sycl_t>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              reinterpret_cast<sycl_t*>(
                  remap_hidden_states.data_ptr<scalar_t>()),
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

  return remap_hidden_states;
}

at::Tensor allgather(
    const at::Tensor& rank_buffers_ptr,
    at::Tensor output,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "allgather: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      output.dim() == 2,
      "allgather: output must be 2D [num_tokens, hidden_size]");
  TORCH_CHECK(output.is_contiguous(), "allgather: output must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather: rank must be in [0, world_size)");

  const int64_t num_tokens = output.size(0);
  const int64_t hidden_size = output.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "allgather: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  if (num_tokens == 0 || hidden_size == 0) {
    return output;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(output.scalar_type(), "allgather", [&]() {
    using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
    if (hidden_size % VEC_SIZE == 0) {
      const int64_t hidden_vecs = hidden_size / VEC_SIZE;
      const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
      const int64_t blocks = (total + threads - 1) / threads;
      auto kfn = vllm::moe::AllgatherRingVecKernel<sycl_t, VEC_SIZE>{
          rank_buffers_ptr.data_ptr<int64_t>(),
          reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
          static_cast<int32_t>(num_tokens_per_rank),
          static_cast<int32_t>(hidden_size),
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
      const int64_t total = world_size * num_tokens_per_rank * hidden_size;
      const int64_t blocks = (total + threads - 1) / threads;
      auto kfn = vllm::moe::AllgatherRingScalarKernel<sycl_t>{
          rank_buffers_ptr.data_ptr<int64_t>(),
          reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
          num_tokens_per_rank,
          hidden_size,
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
