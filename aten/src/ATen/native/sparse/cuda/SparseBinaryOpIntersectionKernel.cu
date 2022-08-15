#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionStubs.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

namespace {

template <typename func_t>
struct CUDAKernel {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};


struct MulOp {
  static Tensor apply(const Tensor& a, const Tensor& b) {
    return a.mul(b);
  }
};

void mul_sparse_sparse_out_cuda_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  _sparse_binary_op_intersection_kernel_out<CUDAKernel, MulOp>(
      result, x, y
  );
}

}

REGISTER_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cuda_kernel);

}}
