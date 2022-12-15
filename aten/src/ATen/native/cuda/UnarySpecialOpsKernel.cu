#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
namespace native {

const char exp2_name[] = "exp2_kernel";
void exp2_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "exp2_cuda", [&]() {
      jitted_gpu_kernel</*name=*/exp2_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, exp2_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16,
        iter.common_dtype(), "exp2_cuda",
        [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ::exp2(a);
          });
        });
  #endif
}

const char i0_name[] = "i0";
void i0_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0_cuda", [&]() {
      jitted_gpu_kernel</*name=*/i0_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, i0_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        // implicit conversion of a to opmath_t will happen here,
        //   but as far as TI is concerned, it's still a no-dynamic-cast kernel because lambda input is scalar_t
        return calc_i0<opmath_t>(a);
      });
    });
  #endif
}

// See note [Jiterator]
const char i0e_name[] = "calc_i0e";
void i0e_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0e_cuda", [&]() {
      jitted_gpu_kernel</*name=*/i0e_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, i0e_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0e_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        return calc_i0e<opmath_t>(a);
      });
    });
  #endif
}

// See note [Jiterator]

const char i1_name[] = "i1";
void i1_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1_cuda", [&]() {
      jitted_gpu_kernel</*name=*/i1_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, i1_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return calc_i1(a);
      });
    });
  #endif // AT_USE_JITERATOR()
}

const char i1e_name[] = "i1e";
void i1e_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1e_cuda", [&]() {
      jitted_gpu_kernel</*name=*/i1e_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, i1e_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1e_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return calc_i1e(a);
      });
    });
  #endif
}

const char sigmoid_name[] = "sigmoid";
void sigmoid_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    // only jiterate for complex-dtype
    #if AT_USE_JITERATOR()
      static const auto sigmoid_string = jiterator_stringify(
        template <typename T>
        T sigmoid(T x) {
          return T{1} / (T{1} + std::exp(-x));
        }
      ); // sigmoid_string
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sigmoid_cuda", [&]() {
        jitted_gpu_kernel<
            /*name=*/sigmoid_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/1>(iter, sigmoid_string);
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sigmoid_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const auto one = opmath_t{1};
          return static_cast<scalar_t>(one / (one + std::exp(-opmath_t{a})));
        });
      });
    #endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, common_dtype, "sigmoid_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto one = opmath_t{1};
        return static_cast<scalar_t>(one/(one + std::exp(-opmath_t{a})));
      });
    });
  }
}

const char sinc_name[] = "sinc";
void sinc_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "sinc_cuda",
      [&]() {
        jitted_gpu_kernel</*name=*/sinc_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, sinc_string);
      });
  #else
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16,
        iter.common_dtype(), "sinc_cuda",
        [&]() {
          gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
            if (a == scalar_t(0)) {
              return scalar_t(1);
            } else {
              // NVCC says constexpr var is not accessible from device
              using opmath_t = at::opmath_type<scalar_t>;
              opmath_t product = c10::detail::pi<opmath_t>() * opmath_t{a};
              return static_cast<scalar_t>(std::sin(product) / product);
            }
          });
        });
  #endif
}

void logit_kernel_cuda(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "logit_cuda",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
            const T_ACC x_acc = static_cast<T_ACC>(x);
            return c10::cuda::compat::log(x_acc / (T_ACC(1) - x_acc));
          });
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          gpu_kernel(
              iter, [lo, hi] GPU_LAMBDA(scalar_t x) -> scalar_t {
                const T_ACC x_acc = static_cast<T_ACC>(x);
                T_ACC z = x_acc < lo ? lo : (x_acc > hi ? hi : x_acc);
                return c10::cuda::compat::log(z / (T_ACC(1) - z));
              });
        }
      });
}

const char ndtri_name[] = "ndtri";
void ndtri_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ndtri_cuda", [&]() {
      jitted_gpu_kernel</*name=*/ndtri_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, ndtri_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ndtri_cuda", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return calc_ndtri(a); });
      });
  #endif
}

const char log_ndtr_name[] = "log_ndtr";
void log_ndtr_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "log_ndtr_cuda", [&]() {
      jitted_gpu_kernel</*name=*/log_ndtr_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, log_ndtr_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "log_ndtr_cuda", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return calc_log_ndtr(a); });
      });
  #endif
}

void erf_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "erf_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erf(a);
    });
  });
}

const char erfc_name[] = "erfc_kernel";
void erfc_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "erfc_cuda", [&]() {
      jitted_gpu_kernel</*name=*/erfc_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, erfc_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16,
        iter.common_dtype(), "erfc_cuda", [&]() {
          gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ::erfc(a);
          });
        });
  #endif
}

const char erfinv_name[] = "erfinv_kernel";
void erfinv_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "erfinv_cuda", [&]() {
      jitted_gpu_kernel</*name=*/erfinv_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, erfinv_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "erfinv_cuda", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::erfinv(a);
      });
    });
  #endif
}

const char erfcx_name[] = "erfcx";
void erfcx_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "erfcx_cuda", [&]() {
      jitted_gpu_kernel</*name=*/erfcx_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, erfcx_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "erfcx_cuda", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return calc_erfcx(a); });
    });
  #endif
}

const char kaiser_window_name[] = "kaiser_window";
void kaiser_window_kernel_cuda(TensorIteratorBase& iter, int64_t window_length, double beta_){
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "kaiser_window_cuda", [&](){
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t inv_alpha = static_cast<opmath_t>(2.0 / (window_length - 1));
        const opmath_t beta = static_cast<opmath_t>(beta_);
        const opmath_t inv_i0_beta = 1.0 / calc_i0(beta);
        jitted_gpu_kernel<
            /*name=*/kaiser_window_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/1>(
            iter,
            kaiser_window_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(inv_alpha, beta, inv_i0_beta));
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "kaiser_window_cuda", [&](){
      using opmath_t = at::opmath_type<scalar_t>;
      const opmath_t inv_alpha = static_cast<opmath_t>(2.0 / (window_length - 1));
      const opmath_t beta = static_cast<opmath_t>(beta_);
      const opmath_t inv_i0_beta = 1.0 / calc_i0(beta);
      gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t a) -> scalar_t {
        opmath_t x = static_cast<opmath_t>(a) * inv_alpha - 1;
        opmath_t y = std::max<opmath_t>(0, 1 - x * x);
        return calc_i0(beta * ::sqrt(y)) * inv_i0_beta;
      });
    });
  #endif
}

const char entr_name[] = "entr";
void entr_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "entr_cuda", [&]() {
      jitted_gpu_kernel</*name=*/entr_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 1>(iter, entr_string);
      });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        iter.common_dtype(),
        "entr_cuda",
        [&]() {
          gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t x) -> scalar_t {
            if (at::_isnan(x)) {
              return x;
            } else if (x > 0) {
              return -x * std::log(x);
            } else if (x == 0) {
              return 0;
            }
            return static_cast<scalar_t>(-INFINITY);
          });
        });
  #endif
}

REGISTER_DISPATCH(exp2_stub, &exp2_kernel_cuda);
REGISTER_DISPATCH(i0_stub, &i0_kernel_cuda);
REGISTER_DISPATCH(special_i0e_stub, &i0e_kernel_cuda);
REGISTER_DISPATCH(special_i1_stub, &i1_kernel_cuda);
REGISTER_DISPATCH(special_i1e_stub, &i1e_kernel_cuda);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel_cuda);
REGISTER_DISPATCH(sinc_stub, &sinc_kernel_cuda);
REGISTER_DISPATCH(logit_stub, &logit_kernel_cuda);
REGISTER_DISPATCH(erf_stub, &erf_kernel_cuda);
REGISTER_DISPATCH(erfc_stub, &erfc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(kaiser_window_stub, &kaiser_window_kernel_cuda);
REGISTER_DISPATCH(special_entr_stub, &entr_kernel_cuda);
REGISTER_DISPATCH(special_ndtri_stub, &ndtri_kernel_cuda);
REGISTER_DISPATCH(special_log_ndtr_stub, &log_ndtr_kernel_cuda);
REGISTER_DISPATCH(special_erfcx_stub, &erfcx_kernel_cuda);

} // namespace native
} // namespace at
