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
        return scalar_t{1} / (scalar_t{1} + std::exp(-a));
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
              scalar_t product = c10::detail::pi<scalar_t>() * a;
              return std::sin(product) / product;
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

// BESSEL FUNCTIONS

const char bessel_j0_name[] = "bessel_j0";

void bessel_j0_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j0_cuda", [&]() {
            jitted_gpu_kernel<bessel_j0_name, scalar_t, scalar_t, 1>(iterator, bessel_j0_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j0_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return bessel_j0(x);
            });
        });
    #endif
} // bessel_j0_kernel_cuda

const char bessel_j1_name[] = "bessel_j1";

void bessel_j1_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j1_cuda", [&]() {
            jitted_gpu_kernel<bessel_j1_name, scalar_t, scalar_t, 1>(iterator, bessel_j1_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j1_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return bessel_j1(x);
            });
        });
    #endif
} // bessel_j1_kernel_cuda

const char bessel_y0_name[] = "bessel_y0";

void bessel_y0_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y0_cuda", [&]() {
            jitted_gpu_kernel<bessel_y0_name, scalar_t, scalar_t, 1>(iterator, bessel_y0_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y0_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return bessel_y0(x);
            });
        });
    #endif
} // bessel_y0_kernel_cuda

const char bessel_y1_name[] = "bessel_y1";

void bessel_y1_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y1_cuda", [&]() {
            jitted_gpu_kernel<bessel_y1_name, scalar_t, scalar_t, 1>(iterator, bessel_y1_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y1_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return bessel_y1(x);
            });
        });
    #endif
} // bessel_y1_kernel_cuda

// MODIFIED BESSEL FUNCTIONS

const char modified_bessel_i0_name[] = "modified_bessel_i0";

void modified_bessel_i0_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i0_cuda", [&]() {
            jitted_gpu_kernel<modified_bessel_i0_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_i0_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i0_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return modified_bessel_i0(x);
            });
        });
    #endif
} // modified_bessel_i0_kernel_cuda

const char modified_bessel_i1_name[] = "modified_bessel_i1";

void modified_bessel_i1_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i1_cuda", [&]() {
            jitted_gpu_kernel<modified_bessel_i1_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_i1_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i1_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return modified_bessel_i1(x);
            });
        });
    #endif
} // modified_bessel_i1_kernel_cuda

const char modified_bessel_k0_name[] = "modified_bessel_k0";

void modified_bessel_k0_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k0_cuda", [&]() {
            jitted_gpu_kernel<modified_bessel_k0_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_k0_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k0_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return modified_bessel_k0(x);
            });
        });
    #endif
} // modified_bessel_k0_kernel_cuda

const char modified_bessel_k1_name[] = "modified_bessel_k1";

void modified_bessel_k1_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k1_cuda", [&]() {
            jitted_gpu_kernel<modified_bessel_k1_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_k1_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k1_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return modified_bessel_k1(x);
            });
        });
    #endif
} // modified_bessel_k1_kernel_cuda

// SCALED MODIFIED BESSEL FUNCTIONS

const char scaled_modified_bessel_i0_name[] = "scaled_modified_bessel_i0";

void scaled_modified_bessel_i0_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_i0_cuda", [&]() {
            jitted_gpu_kernel<scaled_modified_bessel_i0_name, scalar_t, scalar_t, 1>(iterator, scaled_modified_bessel_i0_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_i0_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return scaled_modified_bessel_i0(x);
            });
        });
    #endif
} // scaled_modified_bessel_i0_kernel_cuda

const char scaled_modified_bessel_i1_name[] = "scaled_modified_bessel_i1";

void scaled_modified_bessel_i1_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_i1_cuda", [&]() {
            jitted_gpu_kernel<scaled_modified_bessel_i1_name, scalar_t, scalar_t, 1>(iterator, scaled_modified_bessel_i1_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_i1_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return scaled_modified_bessel_i1(x);
            });
        });
    #endif
} // scaled_modified_bessel_i1_kernel_cuda

const char scaled_modified_bessel_k0_name[] = "scaled_modified_bessel_k0";

void scaled_modified_bessel_k0_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_k0_cuda", [&]() {
            jitted_gpu_kernel<scaled_modified_bessel_k0_name, scalar_t, scalar_t, 1>(iterator, scaled_modified_bessel_k0_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_k0_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return scaled_modified_bessel_k0(x);
            });
        });
    #endif
} // scaled_modified_bessel_k0_kernel_cuda

const char scaled_modified_bessel_k1_name[] = "scaled_modified_bessel_k1";

void scaled_modified_bessel_k1_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_k1_cuda", [&]() {
            jitted_gpu_kernel<scaled_modified_bessel_k1_name, scalar_t, scalar_t, 1>(iterator, scaled_modified_bessel_k1_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_k1_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return scaled_modified_bessel_k1(x);
            });
        });
    #endif
} // scaled_modified_bessel_k1_kernel_cuda

// AIRY FUNCTIONS

const char airy_ai_name[] = "airy_ai";

void airy_ai_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda", [&]() {
            jitted_gpu_kernel<airy_ai_name, scalar_t, scalar_t, 1>(iterator, airy_ai_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return airy_ai(x);
            });
        });
    #endif
} // airy_ai_kernel_cuda

const char airy_bi_name[] = "airy_bi";

void airy_bi_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cuda", [&]() {
            jitted_gpu_kernel<airy_bi_name, scalar_t, scalar_t, 1>(iterator, airy_bi_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return airy_bi(x);
            });
        });
    #endif
} // airy_bi_kernel_cuda

// AIRY DERIVATIVES

const char airy_derivative_ai_name[] = "airy_derivative_ai";

void airy_derivative_ai_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_derivative_ai_cuda", [&]() {
            jitted_gpu_kernel<airy_derivative_ai_name, scalar_t, scalar_t, 1>(iterator, airy_derivative_ai_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_derivative_ai_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return airy_derivative_ai(x);
            });
        });
    #endif
} // airy_derivative_ai_kernel_cuda

const char airy_derivative_bi_name[] = "airy_derivative_bi";

void airy_derivative_bi_kernel_cuda(TensorIteratorBase& iterator) {
    #ifdef USE_JITERATOR
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_derivative_bi_cuda", [&]() {
            jitted_gpu_kernel<airy_derivative_bi_name, scalar_t, scalar_t, 1>(iterator, airy_derivative_bi_string);
        });
    #else
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_derivative_bi_cuda", [&]() {
            gpu_kernel(iterator, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
                return airy_derivative_bi(x);
            });
        });
    #endif
} // airy_derivative_bi_kernel_cuda

REGISTER_DISPATCH(erf_stub, &erf_kernel_cuda);
REGISTER_DISPATCH(erfc_stub, &erfc_kernel_cuda);
REGISTER_DISPATCH(erfinv_stub, &erfinv_kernel_cuda);
REGISTER_DISPATCH(exp2_stub, &exp2_kernel_cuda);
REGISTER_DISPATCH(i0_stub, &i0_kernel_cuda);
REGISTER_DISPATCH(kaiser_window_stub, &kaiser_window_kernel_cuda);
REGISTER_DISPATCH(logit_stub, &logit_kernel_cuda);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel_cuda);
REGISTER_DISPATCH(sinc_stub, &sinc_kernel_cuda);
REGISTER_DISPATCH(special_airy_ai_stub, &airy_ai_kernel_cuda);
REGISTER_DISPATCH(special_airy_bi_stub, &airy_bi_kernel_cuda);
REGISTER_DISPATCH(special_airy_derivative_ai_stub, &airy_derivative_ai_kernel_cuda);
REGISTER_DISPATCH(special_airy_derivative_bi_stub, &airy_derivative_bi_kernel_cuda);
REGISTER_DISPATCH(special_bessel_j0_stub, &bessel_j0_kernel_cuda);
REGISTER_DISPATCH(special_bessel_j1_stub, &bessel_j1_kernel_cuda);
REGISTER_DISPATCH(special_bessel_y0_stub, &bessel_y0_kernel_cuda);
REGISTER_DISPATCH(special_bessel_y1_stub, &bessel_y1_kernel_cuda);
REGISTER_DISPATCH(special_entr_stub, &entr_kernel_cuda);
REGISTER_DISPATCH(special_erfcx_stub, &erfcx_kernel_cuda);
REGISTER_DISPATCH(special_i0e_stub, &i0e_kernel_cuda);
REGISTER_DISPATCH(special_i1_stub, &i1_kernel_cuda);
REGISTER_DISPATCH(special_i1e_stub, &i1e_kernel_cuda);
REGISTER_DISPATCH(special_log_ndtr_stub, &log_ndtr_kernel_cuda);
REGISTER_DISPATCH(special_modified_bessel_i0_stub, &modified_bessel_i0_kernel_cuda);
REGISTER_DISPATCH(special_modified_bessel_i1_stub, &modified_bessel_i1_kernel_cuda);
REGISTER_DISPATCH(special_modified_bessel_k0_stub, &modified_bessel_k0_kernel_cuda);
REGISTER_DISPATCH(special_modified_bessel_k1_stub, &modified_bessel_k1_kernel_cuda);
REGISTER_DISPATCH(special_ndtri_stub, &ndtri_kernel_cuda);

} // namespace native
} // namespace at
