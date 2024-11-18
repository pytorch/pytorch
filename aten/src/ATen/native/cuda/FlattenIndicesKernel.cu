#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/AccumulateType.h>

namespace at::native {

namespace {

template <typename func_t>
struct CUDAKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};

Tensor flatten_indices_cuda_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<CUDAKernelLauncher>(indices, size);
}

}

REGISTER_CUDA_DISPATCH(flatten_indices_stub, &flatten_indices_cuda_kernel)

} // namespace at::native
