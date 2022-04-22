#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

// We manually overload angle because std::arg does not work with types other than c10::complex.
template<typename scalar_t>
__host__ __device__ static inline scalar_t angle_wrapper(scalar_t v) {
  if (at::_isnan(v)){
    return v;
  }
  return v < 0 ? M_PI : 0;
}

template<typename T>
__host__ __device__ static inline c10::complex<T> angle_wrapper(c10::complex<T> v) {
  return std::arg(v);
}

void angle_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "angle_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return angle_wrapper(a);
    });
  });
}

// We manually overload real because std::real does not work types other than c10::complex.
template<typename scalar_t>
__host__ __device__ static inline scalar_t real_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline c10::complex<T> real_wrapper(c10::complex<T> v) {
  return v.real();
}

void real_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "real_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return real_wrapper(a);
    });
  });
}

// We manually overload imag because std::imag does not work types other than c10::complex.
template<typename scalar_t>
__host__ __device__ static inline scalar_t imag_wrapper(scalar_t v) {
  return 0;
}

template<typename T>
__host__ __device__ static inline c10::complex<T> imag_wrapper(c10::complex<T> v) {
  return v.imag();
}

void imag_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "imag_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return imag_wrapper(a);
    });
  });
}

// We manually overload conj because std::conj does not work types other than c10::complex.
template<typename scalar_t>
__host__ __device__ static inline scalar_t conj_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline c10::complex<T> conj_wrapper(c10::complex<T> v) {
  return std::conj(v);
}

// NB: Ignores the negative bit on tensors
const char conj_name[] = "conj_kernel";
void conj_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    #if AT_USE_JITERATOR()
      static const auto conj_string = jiterator_stringify(
        template <typename T>
        T conj_kernel(T z) {
          return std::conj(z);
        }
      );
      jitted_gpu_kernel<conj_name, scalar_t, scalar_t, 1>(iter, conj_string);
    #else
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return conj_wrapper(a);
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kBFloat16, kHalf, iter.common_dtype(), "conj_cuda", [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return conj_wrapper(a);
        });
    });
  }
}

REGISTER_DISPATCH(angle_stub, &angle_kernel_cuda);
REGISTER_DISPATCH(real_stub, &real_kernel_cuda);
REGISTER_DISPATCH(imag_stub, &imag_kernel_cuda);
REGISTER_DISPATCH(conj_physical_stub, &conj_kernel_cuda);

}} // namespace at::native
