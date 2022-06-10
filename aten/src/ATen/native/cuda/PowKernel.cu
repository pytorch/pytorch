#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <c10/core/Scalar.h>

namespace at { namespace native {

// Forward declare some unary kernels
void rsqrt_kernel_cuda(TensorIteratorBase& iter);
void sqrt_kernel_cuda(TensorIteratorBase& iter);
void reciprocal_kernel_cuda(TensorIteratorBase& iter);

namespace {


// SFINAE doesn't work well with NVCC under Windows for math functions like pow and sqrt.
// So we need to define the functions with the explicit function signatures.
// As for pow, the following signatures are defined as the device function:
//   pow(float, int)
//   pow(double, int)
//   pow(float, float)
//   pow(double, double)
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
// pow (Otherwise)
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ typename std::enable_if<!std::is_same<Base_type, Exp_type>::value && !std::is_same<Exp_type, int>::value, Base_type>::type
  pow_(Base_type base, Exp_type exp) {
  return static_cast<Base_type>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
}
#else
template <typename Base_type, typename Exp_type>
static inline __host__ __device__ Base_type pow_(Base_type base, Exp_type exp) {
  return ::pow(base, exp);
}
#endif

template <typename T>
static inline __host__ __device__ std::enable_if_t<std::is_integral<T>::value, T> pow_(
    T base, T exp) {
  return at::native::powi(base, exp);
}

template <typename T>
static inline __host__ __device__ c10::complex<T> pow_(c10::complex<T> base, c10::complex<T> exp) {
  return c10_complex_math::pow(base, exp);
}

void pow_tensor_scalar_kernel(TensorIteratorBase& iter, const Scalar& exp_scalar);

template <typename scalar_t>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, scalar_t base) {
  gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t exp) -> scalar_t {
    return pow_(base, exp);
  });
}

template <typename value_t>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, c10::complex<value_t> base) {
  // For complex, thrust::pow uses the identity
  // pow(a, b) = exp(log(a) * b)
  const auto fct = std::log(base);
  gpu_kernel(iter, [=]GPU_LAMBDA(c10::complex<value_t> exp) -> c10::complex<value_t> {
    return std::exp(fct * exp);
  });
}

/* complex<Half> support impl */
const char pow_scalar_base_name[] = "pow_scalar_base_kernel";
template <>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, c10::complex<at::Half> base) {
  using scalar_t = c10::complex<at::Half>;
  using opmath_t = at::opmath_type<scalar_t>;
  // For complex, thrust::pow uses the identity
  // pow(a, b) = exp(log(a) * b)
  const auto fct = std::log(opmath_t{base});
#if AT_USE_JITERATOR()
  static const auto pow_kernel_string =
      jiterator_stringify(template <typename T> T pow_scalar_base_kernel(T exp, T fct) {
        return std::exp(fct * exp);
      });
  jitted_gpu_kernel<pow_scalar_base_name, scalar_t, scalar_t, 1>(
      iter,
      pow_kernel_string,
      /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
      /*scalar_val=*/0,
      /*extra_args=*/std::make_tuple(fct));
#else
  gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t exp) -> scalar_t {
    return std::exp(fct * opmath_t{exp});
  });
#endif
}

namespace {

#if AT_USE_JITERATOR()
/* complex<Half> support impl */
const char pow_name[] = "pow_kernel";
static const auto pow_kernel_string =
    jiterator_stringify(template <typename T> T pow_kernel(T base, T exp) {
      return std::pow(base, exp);
    });
#endif

/* complex<Half> support impl */
void pow_chalf_tensor_scalar_impl(TensorIteratorBase& iter, const Scalar& exp_scalar) {
  using scalar_t = c10::complex<at::Half>;
  using opmath_t = at::opmath_type<scalar_t>;
  auto exp = exp_scalar.to<opmath_t>();
#if AT_USE_JITERATOR()
  jitted_gpu_kernel<pow_name, scalar_t, scalar_t, 1>(
      iter,
      pow_kernel_string,
      /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
      /*scalar_val=*/0,
      /*extra_args=*/std::make_tuple(exp));
#else
  gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base) -> scalar_t {
    return std::pow(opmath_t{base}, exp);
  });
#endif
}

}  // anonymous namespace

void pow_tensor_tensor_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    if (iter.is_cpu_scalar(1)) {
      const auto base = iter.scalar_value<scalar_t>(1);
      iter.remove_operand(1);
      pow_scalar_tensor_impl(iter, base);
    } else if (iter.is_cpu_scalar(2)) {
      const auto exp = iter.scalar_value<scalar_t>(2);
      iter.remove_operand(2);
      pow_chalf_tensor_scalar_impl(iter, exp);
    } else {
      using opmath_t = at::opmath_type<scalar_t>;
      TORCH_INTERNAL_ASSERT(!iter.is_cpu_scalar(1) && !iter.is_cpu_scalar(2));
#if AT_USE_JITERATOR()
      jitted_gpu_kernel<pow_name, scalar_t, scalar_t, 2>(
          iter, pow_kernel_string);
#else
      gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return pow_(opmath_t{base}, opmath_t{exp});
          });
#endif
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "pow_cuda", [&] {
      if (iter.is_cpu_scalar(1)) {
        const auto base = iter.scalar_value<scalar_t>(1);
        iter.remove_operand(1);
        pow_scalar_tensor_impl(iter, base);
      } else if (iter.is_cpu_scalar(2)) {
        const auto exp = iter.scalar_value<scalar_t>(2);
        iter.remove_operand(2);
        pow_tensor_scalar_kernel(iter, exp);
      } else {
        gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base, scalar_t exp) -> scalar_t {
          return pow_(base, exp);
        });
      }
    });
  }
}


template<typename Base_type, typename Exp_type>
void pow_tensor_scalar_kernel_impl(TensorIteratorBase& iter,
                                                 Exp_type exp) {
  const auto d_exp = static_cast<double>(exp);
  // .5 (sqrt), -.5 (rsqrt) and -1 (reciprocal) specializations are handled
  // in pow_tensor_scalar_kernel
  if (d_exp == 2) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return base * base;
    });
  } else if (d_exp == 3) {
    gpu_kernel(iter, [=]GPU_LAMBDA(Base_type base) -> Base_type {
      return base * base * base;
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

void pow_tensor_scalar_kernel(TensorIteratorBase& iter, const Scalar& exp_scalar) {
  // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
  if (!exp_scalar.isComplex()) {
    if (exp_scalar.equal(.5)) {
      return sqrt_kernel_cuda(iter);
    } else if (exp_scalar.equal(-0.5)) {
      return rsqrt_kernel_cuda(iter);
    } else if (exp_scalar.equal(-1.0)) {
      return reciprocal_kernel_cuda(iter);
    }
  }
  if (isComplexType(iter.common_dtype()) || exp_scalar.isComplex()) {
    if (iter.common_dtype() == kComplexHalf) {
      using scalar_t = c10::complex<at::Half>;
      pow_chalf_tensor_scalar_impl(iter, exp_scalar);
      return;
    }
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "pow_cuda", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t base) -> scalar_t {
        return pow_(base, exp);
      });
    });
  } else if (isFloatingType(iter.common_dtype()) || exp_scalar.isIntegral(false)) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "pow_cuda", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      pow_tensor_scalar_kernel_impl<scalar_t>(iter, exp);
    });
  } else {
    TORCH_INTERNAL_ASSERT(false, "invalid combination of type in Pow function, common dtype:", iter.common_dtype(),
                                 "exp is integral?", exp_scalar.isIntegral(false));
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
