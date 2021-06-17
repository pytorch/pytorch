#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t, typename accscalar_t>
struct AddFunctor {
  AddFunctor(accscalar_t a): alpha(a) {}
  __device__ __forceinline__ scalar_t operator() (const scalar_t a, const scalar_t b) const {
    return a + alpha * b;
  }
  private:
    accscalar_t alpha;
};

void add_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    AddFunctor<scalar_t, accscalar_t> f(alpha_scalar.to<accscalar_t>());
    gpu_kernel_with_scalars(iter, f);
  });
}

static void sub_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);

}} // namespace at::native
