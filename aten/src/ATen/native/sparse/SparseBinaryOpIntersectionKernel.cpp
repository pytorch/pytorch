#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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

}

Tensor& _mul_sparse_sparse_out_cpu(
    const Tensor& x,
    const Tensor& y,
    Tensor& result) {
  return _sparse_binary_op_intersection_kernel_out<CPUKernelLauncher, MulOp>(
      result, x, y
  );
}

}}
