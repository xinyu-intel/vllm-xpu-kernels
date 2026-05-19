#include <sycl/sycl.hpp>

#include "../utils.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

template <typename T>
struct AllgatherWithSymmMemScalarKernel {
  const T* input_shard_ptr;
  const int64_t* rank_buffers_ptr;
  T* output_ptr;
  int64_t numel_per_rank;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * numel_per_rank;
    if (idx >= total) return;

    const int64_t step = idx / numel_per_rank;
    const int64_t elem = idx % numel_per_rank;

    const int64_t src_rank = (rank + step + 1) % world_size;

    T val;
    if (src_rank == rank) {
      val = input_shard_ptr[elem];
    } else {
      const T* src = reinterpret_cast<const T*>(rank_buffers_ptr[src_rank]);
      val = src[elem];
    }

    output_ptr[src_rank * numel_per_rank + elem] = val;
  }
};

template <typename scalar_t, int VEC_SIZE>
struct AllgatherWithSymmMemVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard_ptr;
  const int64_t* rank_buffers_ptr;
  scalar_t* output_ptr;
  int32_t numel_per_rank;
  int32_t rank;
  int32_t world_size;
  int32_t numel_vecs;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = world_size * numel_vecs;
    if (idx >= total) return;

    const int32_t vec_elem = idx % numel_vecs;
    const int32_t step = idx / numel_vecs;

    const int32_t src_rank = (rank + step + 1) % world_size;

    vec_t v;
    if (src_rank == rank) {
      auto src_vec = reinterpret_cast<const vec_t*>(input_shard_ptr);
      v = src_vec[vec_elem];
    } else {
      const scalar_t* src =
          reinterpret_cast<const scalar_t*>(rank_buffers_ptr[src_rank]);
      auto src_vec = reinterpret_cast<const vec_t*>(src);
      v = src_vec[vec_elem];
    }

    auto dst_vec = reinterpret_cast<vec_t*>(
        output_ptr + static_cast<int64_t>(src_rank) * numel_per_rank);
    dst_vec[vec_elem] = v;
  }
};

}  // namespace moe
}  // namespace vllm

at::Tensor allgather_with_symm_mem(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    at::Tensor output,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      input_shard.dim() == 1,
      "allgather_with_symm_mem: input_shard must be 1D");
  TORCH_CHECK(
      input_shard.is_contiguous(),
      "allgather_with_symm_mem: input_shard must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather_with_symm_mem: rank_buffers_ptr must be 1D with size == "
      "world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "allgather_with_symm_mem: rank_buffers_ptr must be int64");
  TORCH_CHECK(output.dim() == 1, "allgather_with_symm_mem: output must be 1D");
  TORCH_CHECK(
      output.is_contiguous(),
      "allgather_with_symm_mem: output must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather_with_symm_mem: rank must be in [0, world_size)");
  TORCH_CHECK(
      input_shard.scalar_type() == output.scalar_type(),
      "allgather_with_symm_mem: input_shard and output must have same dtype");

  const int64_t numel_per_rank = input_shard.numel();
  const int64_t total_numel = output.numel();

  TORCH_CHECK(
      total_numel == numel_per_rank * world_size,
      "allgather_with_symm_mem: output.numel() must equal input_shard.numel() "
      "* world_size");

  if (numel_per_rank == 0) {
    return output;
  }

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  const at::DeviceGuard device_guard(output.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      output.scalar_type(), "allgather_with_symm_mem", [&]() {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        if (numel_per_rank % VEC_SIZE == 0) {
          const int64_t numel_vecs = numel_per_rank / VEC_SIZE;
          const int64_t total = world_size * numel_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::AllgatherWithSymmMemVecKernel<sycl_t, VEC_SIZE>{
              reinterpret_cast<const sycl_t*>(input_shard.data_ptr<scalar_t>()),
              rank_buffers_ptr.data_ptr<int64_t>(),
              reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
              static_cast<int32_t>(numel_per_rank),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(numel_vecs)};
          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                kfn);
          });
        } else {
          const int64_t total = world_size * numel_per_rank;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = vllm::moe::AllgatherWithSymmMemScalarKernel<sycl_t>{
              reinterpret_cast<const sycl_t*>(input_shard.data_ptr<scalar_t>()),
              rank_buffers_ptr.data_ptr<int64_t>(),
              reinterpret_cast<sycl_t*>(output.data_ptr<scalar_t>()),
              numel_per_rank,
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
