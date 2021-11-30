#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t>
struct CompareFunctor{
  CompareFunctor(const int op): op_(op) {TORCH_INTERNAL_ASSERT_DEBUG_ONLY(op_>=0 && op_<=1);}
  const int op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == 0) {
      return a != b;
    } else if (op_ == 1) { //EQ
      return a == b;
    }
  }
};

void ne_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ne_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(0));
  });
}

void eq_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ne_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareFunctor<scalar_t>(1));
  });
}

REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda);
REGISTER_DISPATCH(eq_stub, &eq_kernel_cuda);

}} // namespace at::native
