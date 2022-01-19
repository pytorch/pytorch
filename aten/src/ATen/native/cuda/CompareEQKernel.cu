#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native { namespace {

enum class EqOpType {EQ, NE};

template<typename scalar_t>
struct CompareEqFunctor{
  CompareEqFunctor(EqOpType op): op_(op) {}
  const EqOpType op_;
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == EqOpType::EQ) {
      return a == b;
    } else { //NE
      return a != b;
    }

  }
 };
}

void eq_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "eq_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareEqFunctor<scalar_t>(EqOpType::EQ));
  });
}

void ne_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ne_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareEqFunctor<scalar_t>(EqOpType::NE));
  });
}

REGISTER_DISPATCH(eq_stub, &eq_kernel_cuda);
REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda);

}} // namespace at::native
