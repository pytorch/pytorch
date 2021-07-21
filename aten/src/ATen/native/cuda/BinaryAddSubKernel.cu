#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <c10/cuda/CUDAGuard.h>

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

template<typename scalar_t, typename accscalar_t, int SCALAR_ARG>
struct AddScalarFunctor {
  static_assert(SCALAR_ARG == 1 || SCALAR_ARG == 2, "SCALAR_ARG must be either 1 or 2");
  AddScalarFunctor(accscalar_t alpha, accscalar_t b): alpha(alpha), b(b) {}
  __device__ __forceinline__ scalar_t operator() (const scalar_t a) const {
    return static_cast<scalar_t>(SCALAR_ARG == 1 ? b + alpha * a : a + alpha * b);
  }
  private:
    accscalar_t alpha;
    accscalar_t b;
};

void add_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  if (!isIntegralType(iter.common_dtype(), /* includeBool */ true) && (iter.is_cpu_scalar(1) || iter.is_cpu_scalar(2))) {
    // if common dtype is half the scalar constant can overflow in half precision, and yet the result can
    // still be representable in the half dtype. Cast scalar to acc_type to have better accuracy.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      int scalar_arg = iter.is_cpu_scalar(1) ? 1 : 2;
      auto b = iter.scalar_value<accscalar_t>(scalar_arg);
      iter.remove_operand(scalar_arg);
      const cuda::OptionalCUDAGuard device_guard(device_of(iter.tensor(1)));
      if (scalar_arg == 1) {
        AddScalarFunctor<scalar_t, decltype(b), 1> f(alpha_scalar.to<accscalar_t>(), b);
        gpu_kernel(iter, f);
      } else {
        AddScalarFunctor<scalar_t, decltype(b), 2> f(alpha_scalar.to<accscalar_t>(), b);
        gpu_kernel(iter, f);
      }
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
      using accscalar_t = at::acc_type<scalar_t, true>;
      AddFunctor<scalar_t, accscalar_t> f(alpha_scalar.to<accscalar_t>());
      gpu_kernel_with_scalars(iter, f);
    });
  }
}

static void sub_kernel_cuda(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);

}} // namespace at::native
