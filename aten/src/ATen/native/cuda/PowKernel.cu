#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>

namespace at { namespace native {

namespace {

template <typename T>
static inline __host__ __device__ T powi(T a, T b) {
  T result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

namespace {
  // SFINAE doesn't work well with NVCC under Windows for pow.
  // So we need to define the functions with the explicit function signatures.
  #ifdef _MSC_VER
    // Functions for pow
    // pow for at::Half
    inline __host__ __device__ at::Half pow_(at::Half base, at::Half exp) {
      return static_cast<at::Half>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
    }
    // pow (floating, floating/int)
    template <typename Base_type, typename Exp_type>
    inline __host__ __device__ typename std::enable_if<std::is_floating_point<Base_type>::value && (std::is_same<Base_type, Exp_type>::value || std::is_same<Exp_type, int>::value), Base_type>::type
      pow_(Base_type base, Exp_type exp) {
      return std::pow(base, exp);
    }
    // pow (integral, integral)
    template <typename Base_type, typename Exp_type>
    inline __host__ __device__ typename std::enable_if<std::is_integral<Base_type>::value && std::is_same<Base_type, Exp_type>::value, Base_type>::type
      pow_(Base_type base, Exp_type exp) {
      return powi(base, exp);
    }
    // pow (Otherwise)
    template <typename Base_type, typename Exp_type>
    inline __host__ __device__ typename std::enable_if<!std::is_same<Base_type, Exp_type>::value && !std::is_same<Exp_type, int>::value, Base_type>::type
      pow_(Base_type base, Exp_type exp) {
      return static_cast<Base_type>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
    }
    // Functions for sqrt
    // sqrt (floating)
    template <typename T>
    static inline __host__ __device__ typename std::enable_if<std::is_floating_point<T>::value, T>::type sqrt(T x) {
      return std::sqrt(x);
    }
    // sqrt (integral)
    template <typename T>
    static inline __host__ __device__ typename std::enable_if<!std::is_floating_point<T>::value, T>::type sqrt(T x) {
      return static_cast<T>(std::sqrt(static_cast<double>(x)));
    }
  #else
    template <typename Base_type, typename Exp_type>
    inline __host__ __device__ Base_type pow_(Base_type base, Exp_type exp) {
      return std::pow(base, exp);
    }
    template <typename T>
    static inline __host__ __device__ T sqrt(T x) {
      return std::sqrt(x);
    }
  #endif
}

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return std::pow(base, exp);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
        return powi(base, exp);
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
      return 1.0 / sqrt_(base);
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

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  if (isFloatingType(iter.dtype()) || exp_scalar.isIntegral(false)) {
    AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "pow_cuda", [&]() {
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
