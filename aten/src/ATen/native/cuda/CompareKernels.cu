#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native { namespace {

enum class OpType {GE, GT, LE, LT};

template<typename scalar_t>
struct CompareFunctor{
  CompareFunctor(OpType op): op_(op) {};
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
}


void ge_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ge_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(OpType::GE));
  });
}

void gt_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "gt_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(OpType::GT));
  });
}

void le_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "le_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(OpType::LE));
  });
}

void lt_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "lt_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(OpType::LT));
  });
}

REGISTER_DISPATCH(ge_stub, &ge_kernel_cuda);
REGISTER_DISPATCH(gt_stub, &gt_kernel_cuda);
REGISTER_DISPATCH(le_stub, &le_kernel_cuda);
REGISTER_DISPATCH(lt_stub, &lt_kernel_cuda);

}} // namespace at::native
