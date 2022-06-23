#include <ATen/native/sparse/ValidateCompressedIndicesCommon.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

namespace {

template <typename func_t>
struct CUDAKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};

}

void _validate_compressed_sparse_indices_cuda(
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  validate_compressed_sparse_indices_kernel<CUDAKernelLauncher>(
      cidx, idx, cdim, dim, nnz);
}

}}
