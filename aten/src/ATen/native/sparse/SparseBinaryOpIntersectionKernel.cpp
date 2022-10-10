#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
#include <ATen/native/cpu/Loops.h>

namespace at {
namespace native {

namespace {

template <typename func_t>
struct CPUKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    cpu_kernel(iter, f);
  }
};


struct MulOp {
  static Tensor apply(const Tensor& a, const Tensor& b) {
    return a.mul(b);
  }
};

void mul_sparse_sparse_out_cpu_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, MulOp>(
      result, x, y
  );
}

}

REGISTER_ARCH_DISPATCH(mul_sparse_sparse_out_stub, DEFAULT, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_AVX512_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_AVX2_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_VSX_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);
REGISTER_ZVECTOR_DISPATCH(mul_sparse_sparse_out_stub, &mul_sparse_sparse_out_cpu_kernel);

}}
