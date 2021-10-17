#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
namespace native {

void exp2_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "exp2_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::exp2(a);
        });
      });
}

// TODO: do we want this as a string or a resource string or ... ?
#define stringify(...) std::string(#__VA_ARGS__);
const auto i0_string = stringify(
  template <typename T>
  C10_HOST_DEVICE inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_A() {
    /* Chebyshev coefficients for exp(-x) I0(x)
    * in the interval [0,8].
    *
    * lim(x->0){ exp(-x) I0(x) } = 1.
    */
    static const T coefficients[] = {
        -4.41534164647933937950E-18, 3.33079451882223809783E-17,
        -2.43127984654795469359E-16, 1.71539128555513303061E-15,
        -1.16853328779934516808E-14, 7.67618549860493561688E-14,
        -4.85644678311192946090E-13, 2.95505266312963983461E-12,
        -1.72682629144155570723E-11, 9.67580903537323691224E-11,
        -5.18979560163526290666E-10, 2.65982372468238665035E-9,
        -1.30002500998624804212E-8,  6.04699502254191894932E-8,
        -2.67079385394061173391E-7,  1.11738753912010371815E-6,
        -4.41673835845875056359E-6,  1.64484480707288970893E-5,
        -5.75419501008210370398E-5,  1.88502885095841655729E-4,
        -5.76375574538582365885E-4,  1.63947561694133579842E-3,
        -4.32430999505057594430E-3,  1.05464603945949983183E-2,
        -2.37374148058994688156E-2,  4.93052842396707084878E-2,
        -9.49010970480476444210E-2,  1.71620901522208775349E-1,
        -3.04682672343198398683E-1,  6.76795274409476084995E-1};

    return std::make_tuple(coefficients, 30);
  }

  template <typename T>
  C10_HOST_DEVICE inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_B() {
    /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
    * in the inverted interval [8,infinity].
    *
    * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
    */
    static const T coefficients[] = {
        -7.23318048787475395456E-18, -4.83050448594418207126E-18,
        4.46562142029675999901E-17,  3.46122286769746109310E-17,
        -2.82762398051658348494E-16, -3.42548561967721913462E-16,
        1.77256013305652638360E-15,  3.81168066935262242075E-15,
        -9.55484669882830764870E-15, -4.15056934728722208663E-14,
        1.54008621752140982691E-14,  3.85277838274214270114E-13,
        7.18012445138366623367E-13,  -1.79417853150680611778E-12,
        -1.32158118404477131188E-11, -3.14991652796324136454E-11,
        1.18891471078464383424E-11,  4.94060238822496958910E-10,
        3.39623202570838634515E-9,   2.26666899049817806459E-8,
        2.04891858946906374183E-7,   2.89137052083475648297E-6,
        6.88975834691682398426E-5,   3.36911647825569408990E-3,
        8.04490411014108831608E-1};

    return std::make_tuple(coefficients, 25);
  }

  template <typename scalar_t>
  static inline C10_HOST_DEVICE scalar_t
  chbevl(scalar_t _x, const scalar_t array[], size_t len) {
    using accscalar_t = at::acc_type<scalar_t, true>;

    accscalar_t x = static_cast<accscalar_t>(_x);
    accscalar_t b0, b1, b2;

    b0 = static_cast<accscalar_t>(array[0]);
    b1 = 0;

    for (size_t i = 1; i < len; ++i)  {
      b2 = b1;
      b1 = b0;
      b0 = x * b1 - b2 + static_cast<accscalar_t>(array[i]);
    }

    return static_cast<scalar_t>(0.5 * (b0 - b2));
  }

  template <typename scalar_t>
  static inline C10_HOST_DEVICE scalar_t calc_i0(scalar_t _x) {
    using accscalar_t = at::acc_type<scalar_t, true>;

    // Upcast input for numerical accuracy purposes
    // Needed for accurate results if input is bfloat16 or float16
    accscalar_t x = ::abs(static_cast<accscalar_t>(_x));

    if (x <= accscalar_t{8.0}) {
      auto coeff_pair = chebyshev_coefficients_i0e_A<accscalar_t>();
      auto A = std::get<0>(coeff_pair);
      auto len = std::get<1>(coeff_pair);
      accscalar_t y = (x / accscalar_t{2.0}) - accscalar_t{2.0};
      return static_cast<scalar_t>(::exp(x) * chbevl(y, A, len));
    }

    auto coeff_pair = chebyshev_coefficients_i0e_B<accscalar_t>();
    auto B = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    return static_cast<scalar_t>(::exp(x) * chbevl(accscalar_t{32.0} / x - accscalar_t{2.0}, B, len) / ::sqrt(x));
  }
); // stringify
#undef stringify

void i0_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0_cuda", [&]() {
    jitted_gpu_kernel</*return_dtype=*/ scalar_t, 
                      /*common_dtype=*/ scalar_t,
                      /*arity=*/ 1>(iter, i0_string);
    // gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
    //   return calc_i0(a);
    // });
  });
}

void i0e_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "i0e_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_i0e(a);
    });
  });
}

void i1_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_i1(a);
    });
  });
}

void i1e_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1e_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return calc_i1e(a);
    });
  });
}

void sigmoid_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "sigmoid_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-a));
    });
  });
}

void sinc_kernel_cuda(TensorIteratorBase& iter) {
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

void ndtri_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ndtri_cuda", [&]() {
    gpu_kernel(
        iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return calc_ndtri(a); });
  });
}

void erf_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "erf_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erf(a);
    });
  });
}

void erfc_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      iter.common_dtype(), "erfc_cuda",
      [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::erfc(a);
        });
      });
}

void erfinv_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "erfinv_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::erfinv(a);
    });
  });
}

void erfcx_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "erfcx_cuda", [&]() {
    gpu_kernel(
        iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return calc_erfcx(a); });
  });
}

void kaiser_window_kernel_cuda(TensorIteratorBase& iter, int64_t window_length, double beta_){
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "kaiser_window_cuda", [&](){
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC inv_alpha = static_cast<T_ACC>(2.0 / (window_length - 1));
    const T_ACC beta = static_cast<T_ACC>(beta_);
    const T_ACC inv_i0_beta = 1.0 / calc_i0(beta);
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t a) -> scalar_t {
      T_ACC x = static_cast<T_ACC>(a) * inv_alpha - 1;
      T_ACC y = std::max<T_ACC>(0, 1 - x * x);
      return calc_i0(beta * ::sqrt(y)) * inv_i0_beta;
    });
  });
}

void entr_kernel_cuda(TensorIteratorBase& iter) {
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
REGISTER_DISPATCH(special_erfcx_stub, &erfcx_kernel_cuda);

} // namespace native
} // namespace at
