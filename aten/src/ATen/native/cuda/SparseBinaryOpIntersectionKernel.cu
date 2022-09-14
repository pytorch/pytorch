#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
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

struct MulOp {
  static Tensor apply(const Tensor& a, const Tensor& b) {
    return a.mul(b);
  }
};

void mul_sparse_sparse_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, MulOp>(
      result, x, y
  );
}

}

Tensor& _mul_sparse_sparse_out_cuda(
    const Tensor& x,
    const Tensor& y,
    Tensor& result) {
  _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, MulOp>(
      result, x, y
  );
  return result;
}

REGISTER_CUDA_DISPATCH(mul_sparse_sparse_stub, &mul_sparse_sparse_cuda_kernel);

}}
