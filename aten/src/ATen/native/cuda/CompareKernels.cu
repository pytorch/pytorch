#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native { namespace {

enum class OpType {GE, GT, LE, LT};

template<typename scalar_t>
struct CompareFunctor{
  constexpr CompareFunctor(OpType op): op_(op) {};
  OpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == OpType::GE) {
      return a >= b;
    } else if (op_ == OpType::GT) {
      return a > b;
    } else if (op_ == OpType::LE) {
      return a <= b;
    } else { //LT
      return a < b;
    }
  }
};

// Reflects the comparison operator, so reflect(op)(a, b) == op(b, a)
OpType reflect(OpType x) {
  switch (x) {
    case OpType::GE: return OpType::LE;
    case OpType::GT: return OpType::LT;
    case OpType::LE: return OpType::GE;
    case OpType::LT: return OpType::GT;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid OpType");
}

}  // namespace (anonymous)

template <typename scalar_t>
void compare_scalar_kernel(TensorIteratorBase &iter, OpType op, scalar_t rhs) {
  CompareFunctor<scalar_t> f(op);
  gpu_kernel(iter, [=] GPU_LAMBDA (scalar_t lhs) -> bool {
    return f(lhs, rhs);
  });
}

template <typename scalar_t>
void compare_kernel_impl(TensorIteratorBase &iter, OpType op) {
  // If either input is a cpu scalar, perform the equivalent comparison
  // where the scalar is on the right hand side. This saves us from
  // generating two otherwise identical kernels with mirrored
  // arguments.
  if (iter.is_cpu_scalar(1)) {
    const scalar_t lhs = iter.scalar_value<scalar_t>(1);
    iter.remove_operand(1);
    const DeviceGuard device_guard(iter.device(1));
    compare_scalar_kernel(iter, reflect(op), lhs);
  } else if (iter.is_cpu_scalar(2)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(2);
    iter.remove_operand(2);
    compare_scalar_kernel(iter, op, rhs);
  } else {
    CompareFunctor<scalar_t> f(op);
    gpu_kernel(iter, f);
  }
}

C10_NOINLINE void compare_kernel_with_scalars(TensorIteratorBase &iter, OpType op) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "compare_cuda", [&]() {
    compare_kernel_impl<scalar_t>(iter, op);
  });
}


void ge_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GE);
}

void gt_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GT);
}

void le_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LE);
}

void lt_kernel_cuda(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LT);
}

REGISTER_DISPATCH(ge_stub, &ge_kernel_cuda);
REGISTER_DISPATCH(gt_stub, &gt_kernel_cuda);
REGISTER_DISPATCH(le_stub, &le_kernel_cuda);
REGISTER_DISPATCH(lt_stub, &lt_kernel_cuda);

} // namespace at::native
