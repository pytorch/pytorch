#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "sparse24_pack.h"
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Utils.h>

using namespace torch::sparse;

namespace {
__global__ void meta_shuffle_test_kernel(
    at::PackedTensorAccessor<int64_t, 3> local_meta,
    at::PackedTensorAccessor<int64_t, 3> final_meta,
    bool transpose) {
  uint32_t meta_ab = 0;
  uint32_t meta_cd = 0;
  for (int i = 0; i < 4; ++i) {
    meta_ab |= uint8b_t(uint32_t(local_meta[threadIdx.x][threadIdx.y][i]))
        << (8 * i);
    meta_cd |= uint8b_t(uint32_t(local_meta[threadIdx.x][threadIdx.y][4 + i]))
        << (8 * i);
  }
  final_meta[threadIdx.x][threadIdx.y][0] =
      warp_shuffle_meta(meta_ab, transpose);
  final_meta[threadIdx.x][threadIdx.y][1] =
      warp_shuffle_meta(meta_cd, transpose);
}

at::Tensor _sparse24_meta_shuffle_test(at::Tensor local_meta, bool transpose) {
  auto threads_grid = KernelTypes<cutlass::half_t>::Params::getThreadsGrid();

  TORCH_CHECK(local_meta.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(local_meta.dim() == 3);
  TORCH_CHECK(local_meta.size(0) == threads_grid.x);
  TORCH_CHECK(local_meta.size(1) == threads_grid.y);
  TORCH_CHECK(local_meta.size(2) == kThreadY);
  at::cuda::CUDAGuard device_guard(local_meta.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor final_meta =
      at::zeros({threads_grid.x, threads_grid.y, 2}, local_meta.options());
  size_t smem_bytes = 0;
  meta_shuffle_test_kernel<<<1, threads_grid, smem_bytes, stream>>>(
      local_meta.packed_accessor64<int64_t, 3>(),
      final_meta.packed_accessor64<int64_t, 3>(),
      transpose);
  return final_meta;
}
} // namespace

TORCH_LIBRARY_IMPL(sparse, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::_sparse24_meta_shuffle_test"),
      TORCH_FN(_sparse24_meta_shuffle_test));
}
