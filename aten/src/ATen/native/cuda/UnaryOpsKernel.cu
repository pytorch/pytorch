#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/zmath.cuh>

namespace at { namespace native {

// We manually overload abs because std::abs does not work with thrust::complex types and ROCm.
template<typename scalar_t>
__host__ __device__ static inline scalar_t abs_wrapper(scalar_t v) {
  return ::abs(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> abs_wrapper(thrust::complex<T> v) {
  return thrust::abs(v);
}

__host__ __device__ static inline uint8_t abs_wrapper(uint8_t v) {
  return v;
}

__host__ __device__ static inline bool abs_wrapper(bool v) {
  return v;
}

void abs_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(ScalarType::Half, ScalarType::Bool, iter.dtype(), "abs_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return abs_wrapper(a);
    });
  });
}

// We manually overload angle because std::angle does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t angle_wrapper(scalar_t v) {
  return 0;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> angle_wrapper(thrust::complex<T> v) {
  return thrust::arg(v);
}

void angle_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "angle_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return angle_wrapper(a);
    });
  });
}

// We manually overload real because std::real does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t real_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> real_wrapper(thrust::complex<T> v) {
  return v.real();
}

void real_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "real_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return real_wrapper(a);
    });
  });
}

// We manually overload imag because std::imag does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t imag_wrapper(scalar_t v) {
  return 0;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> imag_wrapper(thrust::complex<T> v) {
  return v.imag();
}

void imag_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "imag_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return imag_wrapper(a);
    });
  });
}

// We manually overload conj because std::conj does not work with std::thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t conj_wrapper(scalar_t v) {
  return v;
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> conj_wrapper(thrust::complex<T> v) {
  return thrust::conj(v);
}

void conj_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "conj_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return conj_wrapper(a);
    });
  });
}

void bitwise_not_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a) {
      return !a;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ~a;
      });
    });
  }
}

void logical_not_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(1), "logical_not_cuda", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kHalf, iter.dtype(0), "logical_not_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

// We manually overload acos because std::acos does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t acos_wrapper(scalar_t v) {
  return ::acos(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> acos_wrapper(thrust::complex<T> v) {
  return thrust::acos(v);
}

void acos_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "acos_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return acos_wrapper(a);
    });
  });
}

// We manually overload asin because std::asin does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t asin_wrapper(scalar_t v) {
  return ::asin(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> asin_wrapper(thrust::complex<T> v) {
  return thrust::asin(v);
}

void asin_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "asin_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return asin_wrapper(a);
    });
  });
}

// We manually overload ceil because std::ceil does not work with std::complex types.
template <typename scalar_t>
__host__ __device__ static inline scalar_t ceil_wrapper(scalar_t a) {
  return std::ceil(a);
}

template<typename T>
__host__ __device__ static inline std::complex<T> ceil_wrapper(std::complex<T> v) {
  return std::complex<T>(std::ceil(v.real()), std::ceil(v.imag()));
}

void ceil_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "ceil_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ceil_wrapper(a);
    });
  });
}

void expm1_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "expm1_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::expm1(a);
    });
  });
}


void frac_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "frac_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a - ::trunc(a);
    });
  });
}

// We manually overload floor because std::floor does not work with std::complex types.
template <typename scalar_t>
__host__ __device__ static inline scalar_t floor_wrapper(scalar_t a) {
  return std::floor(a);
}

template<typename T>
__host__ __device__ static inline std::complex<T> floor_wrapper(std::complex<T> v) {
  return std::complex<T>(std::floor(v.real()), std::floor(v.imag()));
}

void floor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "floor_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return floor_wrapper(a);
    });
  });
}

// We manually overload log because std::log does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t log_wrapper(scalar_t v) {
  return ::log(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> log_wrapper(thrust::complex<T> v) {
  return thrust::log(v);
}

void log_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "log_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return log_wrapper(a);
    });
  });
}

// We manually overload log10 because std::log10 does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t log10_wrapper(scalar_t v) {
  return ::log10(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> log10_wrapper(thrust::complex<T> v) {
  return thrust::log10(v);
}

void log10_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "log10_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return log10_wrapper(a);
    });
  });
}

void log1p_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "log1p_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::log1p(a);
    });
  });
}

// We manually overload log2 because std::log2 does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t log2_wrapper(scalar_t v) {
  return ::log2(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> log2_wrapper(thrust::complex<T> v) {
  const thrust::complex<T> log2 = thrust::complex<T>(::log(2.0), 0.0);
  return thrust::log(v)/log2;
}

void log2_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "log2_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return log2_wrapper(a);
    });
  });
}

void neg_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(ScalarType::Half, iter.dtype(), "neg_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return -a;
    });
  });
}

// We manually overload nearbyint because std::nearbyint does not work with std::complex types and ROCm.
template <typename scalar_t>
__host__ __device__ static inline scalar_t nearbyint_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::nearbyintf(static_cast<float>(a)));
}

__host__ __device__ static inline double nearbyint_wrapper(double a) {
  return ::nearbyint(a);
}

__host__ __device__ static inline std::complex<float> nearbyint_wrapper(std::complex<float> a) {
  return std::complex<float>(::nearbyintf(static_cast<float>(a.real())), ::nearbyintf(static_cast<float>(a.imag())));
}

__host__ __device__ static inline std::complex<double> nearbyint_wrapper(std::complex<double> a) {
  return std::complex<double>(::nearbyint(static_cast<double>(a.real())), ::nearbyint(static_cast<double>(a.imag())));
}

void round_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "round_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      // We do not use std::round because we would like to round midway numbers to the nearest even integer.
      return nearbyint_wrapper(a);
    });
  });
}

// We manually overload trunc because std::trunc does not work with std::complex types and ROCm.
template <typename scalar_t>
__host__ __device__ static inline scalar_t trunc_wrapper(scalar_t a) {
  return static_cast<scalar_t>(::truncf(static_cast<float>(a)));
}

__host__ __device__ static inline double trunc_wrapper(double a) {
  return ::trunc(a);
}

__host__ __device__ static inline std::complex<float> trunc_wrapper(std::complex<float> a) {
  return std::complex<float>(::truncf(static_cast<float>(a.real())), ::truncf(static_cast<float>(a.imag())));
}

__host__ __device__ static inline std::complex<double> trunc_wrapper(std::complex<double> a) {
  return std::complex<double>(::trunc(static_cast<double>(a.real())), ::trunc(static_cast<double>(a.imag())));
}

void trunc_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "trunc_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return trunc_wrapper(a);
    });
  });
}

// We manually overload rsqrt because std::rsqrt does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t rsqrt_wrapper(scalar_t v) {
  return ::rsqrt(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> rsqrt_wrapper(thrust::complex<T> v) {
  const thrust::complex<T> one = thrust::complex<T>(1.0, 0);
  return one/thrust::sqrt(v);
}

void reciprocal_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "reciprocal_cuda", [&]() {
    using acc_t = acc_type<scalar_t, /*is_cuda=*/true>;
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return static_cast<acc_t>(1) / a;
    });
  });
}

void rsqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "rsqrt_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      // In CUDA, ::rsqrt is overloaded for float and at::Half here is implicitly cast to float.
      return rsqrt_wrapper(a);
    });
  });
}

void sign_kernel_cuda(TensorIterator& iter){
    if (iter.dtype() == ScalarType::Bool) {
      gpu_kernel(iter, []GPU_LAMBDA(bool a){
        return a;
      });
    } else {
      AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, iter.dtype(), "sign_cuda", [&]() {
          gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
              scalar_t zero = scalar_t(0);
              return (zero < a) - (a < zero);
          });
      });
    }
}

// We manually overload sin because std::sin does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t sin_wrapper(scalar_t v) {
  return ::sin(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> sin_wrapper(thrust::complex<T> v) {
  return thrust::sin(v);
}

void sin_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sin_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return sin_wrapper(a);
    });
  });
}

// We manually overload sinh because std::sinh does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t sinh_wrapper(scalar_t v) {
  return ::sinh(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> sinh_wrapper(thrust::complex<T> v) {
  return thrust::sinh(v);
}

void sinh_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sinh_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return sinh_wrapper(a);
    });
  });
}

// We manually overload sqrt because std::sqrt does not work with thrust::complex types.
template<typename scalar_t>
__host__ __device__ static inline scalar_t sqrt_wrapper(scalar_t v) {
  return ::sqrt(v);
}

template<typename T>
__host__ __device__ static inline thrust::complex<T> sqrt_wrapper(thrust::complex<T> v) {
  return thrust::sqrt(v);
}

void sqrt_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(ScalarType::Half, iter.dtype(), "sqrt_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    gpu_kernel(iter, []GPU_LAMBDA(thrust_t a) -> thrust_t {
      return sqrt_wrapper(a);
    });
  });
}

void sigmoid_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "sigmoid_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      scalar_t one = scalar_t(1);
      return  one / (one + std::exp(- a));
    });
  });
}

void erfinv_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "erfinv_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erfinv(a);
    });
  });
}

void digamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "digamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_digamma(a);
    });
  });
}

void trigamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "trigamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_trigamma(a);
    });
  });
}

void polygamma_kernel_cuda(TensorIterator& iter, int64_t n) {
  switch (n) {
    case 0: digamma_kernel_cuda(iter); break;
    case 1: trigamma_kernel_cuda(iter); break;
    default: TORCH_CHECK(false, "polygamma(n,x) is not implemented for n>=2, but was ", n);
  }
}

void lgamma_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "lgamma_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::lgamma(a);
    });
  });
}

REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);
REGISTER_DISPATCH(angle_stub, &angle_kernel_cuda);
REGISTER_DISPATCH(real_stub, &real_kernel_cuda);
REGISTER_DISPATCH(imag_stub, &imag_kernel_cuda);
REGISTER_DISPATCH(conj_stub, &conj_kernel_cuda);
REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_cuda);
REGISTER_DISPATCH(acos_stub, &acos_kernel_cuda);
REGISTER_DISPATCH(asin_stub, &asin_kernel_cuda);
REGISTER_DISPATCH(ceil_stub, &ceil_kernel_cuda);
REGISTER_DISPATCH(expm1_stub, &expm1_kernel_cuda);
REGISTER_DISPATCH(frac_stub, &frac_kernel_cuda);
REGISTER_DISPATCH(floor_stub, &floor_kernel_cuda);
REGISTER_DISPATCH(log_stub, &log_kernel_cuda);
REGISTER_DISPATCH(log10_stub, &log10_kernel_cuda);
REGISTER_DISPATCH(log2_stub, &log2_kernel_cuda);
REGISTER_DISPATCH(log1p_stub, &log1p_kernel_cuda);
REGISTER_DISPATCH(neg_stub, &neg_kernel_cuda);
REGISTER_DISPATCH(reciprocal_stub, &reciprocal_kernel_cuda);
REGISTER_DISPATCH(round_stub, &round_kernel_cuda);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel_cuda);
REGISTER_DISPATCH(sign_stub, &sign_kernel_cuda);
REGISTER_DISPATCH(sin_stub, &sin_kernel_cuda);
REGISTER_DISPATCH(sinh_stub, &sinh_kernel_cuda);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel_cuda);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel_cuda);
REGISTER_DISPATCH(trunc_stub, &trunc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(digamma_stub, &digamma_kernel_cuda);
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_cuda);
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_cuda);
}}
