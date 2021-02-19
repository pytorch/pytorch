#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t>
struct CompareNEFunctor {
  __device__ __forceinline__ bool operator() (scalar_t a, scalar_t b) const {
    return a != b;
  }
};

void ne_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ne_cuda", [&]() {
    gpu_kernel_with_scalars(iter, CompareNEFunctor<scalar_t>());
  });
}

REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda);

}} // namespace at::native
