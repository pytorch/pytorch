#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
#include <ATen/native/cuda/Loops.cuh>
#include <iostream>

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

}

Tensor& _mul_sparse_sparse_out_cuda(
    const Tensor& x,
    const Tensor& y,
    Tensor& result) {
  return _sparse_binary_op_intersection_kernel_out<CUDAKernelLauncher, MulOp>(
      result, x, y
  );
}

}}
