#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>

namespace at { namespace native {

namespace {


// SFINAE doesn't work well with NVCC under Windows for math functions like pow and sqrt.
// So we need to define the functions with the explicit function signatures.
// As for pow, the following signatures are defined as the device function:
//   pow(float, int)
//   pow(double, int)
//   pow(float, float)
//   pow(double, double)
// As for sqrt, the following signatures are defined as the device function:
//   sqrt(float)
//   sqrt(double)
// As for inverse sqrt, we must define it explicitly in MSVC, otherwise the static cast will be
// applied to the result of the inline function, and thus the result is incorrect.
//   e.g. if we use 1.0 / sqrt(2) for 2 ^ (-0.5) in MSVC, we get
//          int(2 ^ (-0.5)) = int(1.0 / sqrt(2)) = int(1.0 / int(1.414)) = int(1.0 / 1) = 1
//        However, the correct result is
//          int(2 ^ (-0.5)) = int(1.0 / 1.414) = 0
#ifdef _MSC_VER
// Functions for pow
// pow for at::Half
static inline __host__ __device__ at::Half pow_(at::Half base, at::Half exp) {
  return static_cast<at::Half>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow for at::BFloat16
static inline __host__ __device__ at::BFloat16 pow_(at::BFloat16 base, at::BFloat16 exp) {
  return static_cast<at::BFloat16>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}
// pow (floating, floating/int)
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ typename std::enable_if<std::is_floating_point<Base_type>::value && (std::is_same<Base_type, Exp_type>::value || std::is_same<Exp_type, int>::value), Base_type>::type
  pow_(Base_type base, Exp_type exp) {
  return std::pow(base, exp);
}
// pow (integral, integral)
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ typename std::enable_if<std::is_integral<Base_type>::value && std::is_same<Base_type, Exp_type>::value, Base_type>::type
  pow_(Base_type base, Exp_type exp) {
  return native::powi(base, exp);
}
// pow (Otherwise)
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ typename std::enable_if<!std::is_same<Base_type, Exp_type>::value && !std::is_same<Exp_type, int>::value, Base_type>::type
  pow_(Base_type base, Exp_type exp) {
  return static_cast<Base_type>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
}
// pow (Complex)
template<typename B, typename E>
static inline __host__ __device__ B complex_pow_(B base, E exp) {
  return std::pow(base, exp);
}
// Functions for sqrt
// sqrt (floating)
template <typename T>
static inline __host__ __device__ typename std::enable_if<std::is_floating_point<T>::value, T>::type sqrt_(T x) {
  return std::sqrt(x);
}
// sqrt (integral)
template <typename T>
static inline __host__ __device__ typename std::enable_if<!std::is_floating_point<T>::value, T>::type sqrt_(T x) {
  return static_cast<T>(std::sqrt(static_cast<double>(x)));
}
// Function for inverse sqrt
// invsqrt (floating)
template <typename T>
static inline __host__ __device__ typename std::enable_if<std::is_floating_point<T>::value, T>::type invsqrt_(T x) {
  return 1.0 / std::sqrt(x);
}
// invsqrt (integral)
template <typename T>
static inline __host__ __device__ typename std::enable_if<!std::is_floating_point<T>::value, T>::type invsqrt_(T x) {
  return static_cast<T>(1.0 / std::sqrt(static_cast<double>(x)));
}
#else
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ Base_type pow_(Base_type base, Exp_type exp) {
  return ::pow(base, exp);
}
template <typename T>
static inline __host__ __device__ T sqrt_(T x) {
  return ::sqrt(x);
}
template <typename T>
static inline __host__ __device__ T invsqrt_(T x) {
  return 1.0 / ::sqrt(x);
}
// pow (Otherwise)
template<typename B, typename E>
static inline __host__ __device__ B complex_pow_(B base, E exp) {
  return std::pow(base, exp);
}
#endif

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return complex_pow_(base, exp);
      });
    });
  } else if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return pow_(base, exp);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return native::powi(base, exp);
      });
    });
  }
}

template<typename Base_type, typename Exp_type>
void pow_tensor_scalar_kernel_impl(TensorIterator& iter,
                                                 Exp_type exp) {
  const auto d_exp = static_cast<double>(exp);
  if (d_exp == 0.5) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return sqrt_(base);
    });
  } else if (d_exp == 2) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return base * base;
    });
  } else if (d_exp == 3) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return base * base * base;
    });
  } else if (d_exp == -0.5) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return invsqrt_(base);
    });
  } else if (d_exp == -1) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return 1.0 / base;
    });
  } else if (d_exp == -2) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return 1.0 / (base * base);
    });
  } else {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return pow_(base, exp);
    });
  }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, const Scalar& exp_scalar) {
  if (isComplexType(iter.dtype()) || exp_scalar.isComplex()) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "pow_cuda", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
        return complex_pow_(base, exp);
      });
    });
  } else if (isFloatingType(iter.dtype()) || exp_scalar.isIntegral(false)) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "pow_cuda", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      pow_tensor_scalar_kernel_impl<scalar_t>(iter, exp);
    });
  } else {
    const auto exp = exp_scalar.to<float>();
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow_cuda", [&]() {
      pow_tensor_scalar_kernel_impl<scalar_t>(iter, exp);
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
