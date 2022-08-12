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
const auto airy_ai_string = jiterator_stringify(
  template<typename T1>
  T1 airy_ai(T1 a) {
    return a;
  } // T1 airy_ai(T1 a)
); // airy_ai_string

const char airy_ai_name[] = "airy_ai";

void airy_ai_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<airy_ai_name, scalar_t>(iterator, airy_ai_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void airy_ai_cuda_kernel(TensorIteratorBase &iterator)

const auto airy_bi_string = jiterator_stringify(
  template<typename T1>
  T1 airy_bi(T1 a) {
    return a;
  } // T1 airy_bi(T1 a)
); // airy_bi_string

const char airy_bi_name[] = "airy_bi";

void airy_bi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<airy_bi_name, scalar_t>(iterator, airy_bi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void airy_bi_cuda_kernel(TensorIteratorBase &iterator)

const auto bernoulli_number_string = jiterator_stringify(
  template<typename T1>
  T1 bernoulli_number(T1 a) {
    return a;
  } // T1 bernoulli_number(T1 a)
); // bernoulli_number_string

const char bernoulli_number_name[] = "bernoulli_number";

void bernoulli_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_number_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bernoulli_number_name, scalar_t>(iterator, bernoulli_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void bernoulli_number_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_j_0_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j_0(T1 a) {
    return a;
  } // T1 bessel_j_0(T1 a)
); // bessel_j_0_string

const char bessel_j_0_name[] = "bessel_j_0";

void bessel_j_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_0_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bessel_j_0_name, scalar_t>(iterator, bessel_j_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void bessel_j_0_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_j_1_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j_1(T1 a) {
    return a;
  } // T1 bessel_j_1(T1 a)
); // bessel_j_1_string

const char bessel_j_1_name[] = "bessel_j_1";

void bessel_j_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bessel_j_1_name, scalar_t>(iterator, bessel_j_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void bessel_j_1_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_y_0_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_y_0(T1 a) {
    return a;
  } // T1 bessel_y_0(T1 a)
); // bessel_y_0_string

const char bessel_y_0_name[] = "bessel_y_0";

void bessel_y_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bessel_y_0_name, scalar_t>(iterator, bessel_y_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void bessel_y_0_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_y_1_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_y_1(T1 a) {
    return a;
  } // T1 bessel_y_1(T1 a)
); // bessel_y_1_string

const char bessel_y_1_name[] = "bessel_y_1";

void bessel_y_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bessel_y_1_name, scalar_t>(iterator, bessel_y_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void bessel_y_1_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_elliptic_integral_e_string = jiterator_stringify(
  template<typename T1>
  T1 complete_elliptic_integral_e(T1 a) {
    return a;
  } // T1 complete_elliptic_integral_e(T1 a)
); // complete_elliptic_integral_e_string

const char complete_elliptic_integral_e_name[] = "complete_elliptic_integral_e";

void complete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<complete_elliptic_integral_e_name, scalar_t>(iterator, complete_elliptic_integral_e_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void complete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_elliptic_integral_k_string = jiterator_stringify(
  template<typename T1>
  T1 complete_elliptic_integral_k(T1 a) {
    return a;
  } // T1 complete_elliptic_integral_k(T1 a)
); // complete_elliptic_integral_k_string

const char complete_elliptic_integral_k_name[] = "complete_elliptic_integral_k";

void complete_elliptic_integral_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_k_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<complete_elliptic_integral_k_name, scalar_t>(iterator, complete_elliptic_integral_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void complete_elliptic_integral_k_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_legendre_elliptic_integral_d_string = jiterator_stringify(
  template<typename T1>
  T1 complete_legendre_elliptic_integral_d(T1 a) {
    return a;
  } // T1 complete_legendre_elliptic_integral_d(T1 a)
); // complete_legendre_elliptic_integral_d_string

const char complete_legendre_elliptic_integral_d_name[] = "complete_legendre_elliptic_integral_d";

void complete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<complete_legendre_elliptic_integral_d_name, scalar_t>(iterator, complete_legendre_elliptic_integral_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void complete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator)

const auto cos_pi_string = jiterator_stringify(
  template<typename T1>
  T1 cos_pi(T1 a) {
    return a;
  } // T1 cos_pi(T1 a)
); // cos_pi_string

const char cos_pi_name[] = "cos_pi";

void cos_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cos_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<cos_pi_name, scalar_t>(iterator, cos_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cos_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void cos_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto cosh_pi_string = jiterator_stringify(
  template<typename T1>
  T1 cosh_pi(T1 a) {
    return a;
  } // T1 cosh_pi(T1 a)
); // cosh_pi_string

const char cosh_pi_name[] = "cosh_pi";

void cosh_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosh_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<cosh_pi_name, scalar_t>(iterator, cosh_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosh_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void cosh_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto cosine_integral_ci_string = jiterator_stringify(
  template<typename T1>
  T1 cosine_integral_ci(T1 a) {
    return a;
  } // T1 cosine_integral_ci(T1 a)
); // cosine_integral_ci_string

const char cosine_integral_ci_name[] = "cosine_integral_ci";

void cosine_integral_ci_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosine_integral_ci_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<cosine_integral_ci_name, scalar_t>(iterator, cosine_integral_ci_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosine_integral_ci_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void cosine_integral_ci_cuda_kernel(TensorIteratorBase &iterator)

const auto digamma_string = jiterator_stringify(
  template<typename T1>
  T1 digamma(T1 a) {
    return a;
  } // T1 digamma(T1 a)
); // digamma_string

const char digamma_name[] = "digamma";

void digamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "digamma_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<digamma_name, scalar_t>(iterator, digamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "digamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void digamma_cuda_kernel(TensorIteratorBase &iterator)

const auto dilogarithm_li_2_string = jiterator_stringify(
  template<typename T1>
  T1 dilogarithm_li_2(T1 a) {
    return a;
  } // T1 dilogarithm_li_2(T1 a)
); // dilogarithm_li_2_string

const char dilogarithm_li_2_name[] = "dilogarithm_li_2";

void dilogarithm_li_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dilogarithm_li_2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<dilogarithm_li_2_name, scalar_t>(iterator, dilogarithm_li_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dilogarithm_li_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void dilogarithm_li_2_cuda_kernel(TensorIteratorBase &iterator)

const auto dirichlet_beta_string = jiterator_stringify(
  template<typename T1>
  T1 dirichlet_beta(T1 a) {
    return a;
  } // T1 dirichlet_beta(T1 a)
); // dirichlet_beta_string

const char dirichlet_beta_name[] = "dirichlet_beta";

void dirichlet_beta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_beta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<dirichlet_beta_name, scalar_t>(iterator, dirichlet_beta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_beta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void dirichlet_beta_cuda_kernel(TensorIteratorBase &iterator)

const auto dirichlet_eta_string = jiterator_stringify(
  template<typename T1>
  T1 dirichlet_eta(T1 a) {
    return a;
  } // T1 dirichlet_eta(T1 a)
); // dirichlet_eta_string

const char dirichlet_eta_name[] = "dirichlet_eta";

void dirichlet_eta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_eta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<dirichlet_eta_name, scalar_t>(iterator, dirichlet_eta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_eta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void dirichlet_eta_cuda_kernel(TensorIteratorBase &iterator)

const auto dirichlet_lambda_string = jiterator_stringify(
  template<typename T1>
  T1 dirichlet_lambda(T1 a) {
    return a;
  } // T1 dirichlet_lambda(T1 a)
); // dirichlet_lambda_string

const char dirichlet_lambda_name[] = "dirichlet_lambda";

void dirichlet_lambda_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_lambda_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<dirichlet_lambda_name, scalar_t>(iterator, dirichlet_lambda_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_lambda_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void dirichlet_lambda_cuda_kernel(TensorIteratorBase &iterator)

const auto double_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 double_factorial(T1 a) {
    return a;
  } // T1 double_factorial(T1 a)
); // double_factorial_string

const char double_factorial_name[] = "double_factorial";

void double_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "double_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<double_factorial_name, scalar_t>(iterator, double_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "double_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void double_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_airy_ai_string = jiterator_stringify(
  template<typename T1>
  T1 exp_airy_ai(T1 a) {
    return a;
  } // T1 exp_airy_ai(T1 a)
); // exp_airy_ai_string

const char exp_airy_ai_name[] = "exp_airy_ai";

void exp_airy_ai_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_ai_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exp_airy_ai_name, scalar_t>(iterator, exp_airy_ai_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_ai_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void exp_airy_ai_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_airy_bi_string = jiterator_stringify(
  template<typename T1>
  T1 exp_airy_bi(T1 a) {
    return a;
  } // T1 exp_airy_bi(T1 a)
); // exp_airy_bi_string

const char exp_airy_bi_name[] = "exp_airy_bi";

void exp_airy_bi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_bi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exp_airy_bi_name, scalar_t>(iterator, exp_airy_bi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_bi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void exp_airy_bi_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_k_0_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_k_0(T1 a) {
    return a;
  } // T1 exp_modified_bessel_k_0(T1 a)
); // exp_modified_bessel_k_0_string

const char exp_modified_bessel_k_0_name[] = "exp_modified_bessel_k_0";

void exp_modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_0_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exp_modified_bessel_k_0_name, scalar_t>(iterator, exp_modified_bessel_k_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void exp_modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_k_1_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_k_1(T1 a) {
    return a;
  } // T1 exp_modified_bessel_k_1(T1 a)
); // exp_modified_bessel_k_1_string

const char exp_modified_bessel_k_1_name[] = "exp_modified_bessel_k_1";

void exp_modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exp_modified_bessel_k_1_name, scalar_t>(iterator, exp_modified_bessel_k_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void exp_modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator)

const auto exponential_integral_ei_string = jiterator_stringify(
  template<typename T1>
  T1 exponential_integral_ei(T1 a) {
    return a;
  } // T1 exponential_integral_ei(T1 a)
); // exponential_integral_ei_string

const char exponential_integral_ei_name[] = "exponential_integral_ei";

void exponential_integral_ei_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_ei_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exponential_integral_ei_name, scalar_t>(iterator, exponential_integral_ei_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_ei_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void exponential_integral_ei_cuda_kernel(TensorIteratorBase &iterator)

const auto factorial_string = jiterator_stringify(
  template<typename T1>
  T1 factorial(T1 a) {
    return a;
  } // T1 factorial(T1 a)
); // factorial_string

const char factorial_name[] = "factorial";

void factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<factorial_name, scalar_t>(iterator, factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto fresnel_integral_c_string = jiterator_stringify(
  template<typename T1>
  T1 fresnel_integral_c(T1 a) {
    return a;
  } // T1 fresnel_integral_c(T1 a)
); // fresnel_integral_c_string

const char fresnel_integral_c_name[] = "fresnel_integral_c";

void fresnel_integral_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_c_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<fresnel_integral_c_name, scalar_t>(iterator, fresnel_integral_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void fresnel_integral_c_cuda_kernel(TensorIteratorBase &iterator)

const auto fresnel_integral_s_string = jiterator_stringify(
  template<typename T1>
  T1 fresnel_integral_s(T1 a) {
    return a;
  } // T1 fresnel_integral_s(T1 a)
); // fresnel_integral_s_string

const char fresnel_integral_s_name[] = "fresnel_integral_s";

void fresnel_integral_s_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_s_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<fresnel_integral_s_name, scalar_t>(iterator, fresnel_integral_s_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_s_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void fresnel_integral_s_cuda_kernel(TensorIteratorBase &iterator)

const auto harmonic_number_string = jiterator_stringify(
  template<typename T1>
  T1 harmonic_number(T1 a) {
    return a;
  } // T1 harmonic_number(T1 a)
); // harmonic_number_string

const char harmonic_number_name[] = "harmonic_number";

void harmonic_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<harmonic_number_name, scalar_t>(iterator, harmonic_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void harmonic_number_cuda_kernel(TensorIteratorBase &iterator)

const auto hyperbolic_cosine_integral_chi_string = jiterator_stringify(
  template<typename T1>
  T1 hyperbolic_cosine_integral_chi(T1 a) {
    return a;
  } // T1 hyperbolic_cosine_integral_chi(T1 a)
); // hyperbolic_cosine_integral_chi_string

const char hyperbolic_cosine_integral_chi_name[] = "hyperbolic_cosine_integral_chi";

void hyperbolic_cosine_integral_chi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_cosine_integral_chi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hyperbolic_cosine_integral_chi_name, scalar_t>(iterator, hyperbolic_cosine_integral_chi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_cosine_integral_chi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void hyperbolic_cosine_integral_chi_cuda_kernel(TensorIteratorBase &iterator)

const auto hyperbolic_sine_integral_shi_string = jiterator_stringify(
  template<typename T1>
  T1 hyperbolic_sine_integral_shi(T1 a) {
    return a;
  } // T1 hyperbolic_sine_integral_shi(T1 a)
); // hyperbolic_sine_integral_shi_string

const char hyperbolic_sine_integral_shi_name[] = "hyperbolic_sine_integral_shi";

void hyperbolic_sine_integral_shi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_sine_integral_shi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hyperbolic_sine_integral_shi_name, scalar_t>(iterator, hyperbolic_sine_integral_shi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_sine_integral_shi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void hyperbolic_sine_integral_shi_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_double_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_double_factorial(T1 a) {
    return a;
  } // T1 ln_double_factorial(T1 a)
); // ln_double_factorial_string

const char ln_double_factorial_name[] = "ln_double_factorial";

void ln_double_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_double_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_double_factorial_name, scalar_t>(iterator, ln_double_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_double_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_double_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_factorial(T1 a) {
    return a;
  } // T1 ln_factorial(T1 a)
); // ln_factorial_string

const char ln_factorial_name[] = "ln_factorial";

void ln_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_factorial_name, scalar_t>(iterator, ln_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_gamma_sign_string = jiterator_stringify(
  template<typename T1>
  T1 ln_gamma_sign(T1 a) {
    return a;
  } // T1 ln_gamma_sign(T1 a)
); // ln_gamma_sign_string

const char ln_gamma_sign_name[] = "ln_gamma_sign";

void ln_gamma_sign_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_sign_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_gamma_sign_name, scalar_t>(iterator, ln_gamma_sign_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_sign_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_gamma_sign_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 ln_gamma(T1 a) {
    return a;
  } // T1 ln_gamma(T1 a)
); // ln_gamma_string

const char ln_gamma_name[] = "ln_gamma";

void ln_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_gamma_name, scalar_t>(iterator, ln_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto logarithmic_integral_li_string = jiterator_stringify(
  template<typename T1>
  T1 logarithmic_integral_li(T1 a) {
    return a;
  } // T1 logarithmic_integral_li(T1 a)
); // logarithmic_integral_li_string

const char logarithmic_integral_li_name[] = "logarithmic_integral_li";

void logarithmic_integral_li_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "logarithmic_integral_li_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<logarithmic_integral_li_name, scalar_t>(iterator, logarithmic_integral_li_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "logarithmic_integral_li_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void logarithmic_integral_li_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_i_0_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_i_0(T1 a) {
    return a;
  } // T1 modified_bessel_i_0(T1 a)
); // modified_bessel_i_0_string

const char modified_bessel_i_0_name[] = "modified_bessel_i_0";

void modified_bessel_i_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<modified_bessel_i_0_name, scalar_t>(iterator, modified_bessel_i_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void modified_bessel_i_0_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_i_1_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_i_1(T1 a) {
    return a;
  } // T1 modified_bessel_i_1(T1 a)
); // modified_bessel_i_1_string

const char modified_bessel_i_1_name[] = "modified_bessel_i_1";

void modified_bessel_i_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<modified_bessel_i_1_name, scalar_t>(iterator, modified_bessel_i_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void modified_bessel_i_1_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_k_0_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k_0(T1 a) {
    return a;
  } // T1 modified_bessel_k_0(T1 a)
); // modified_bessel_k_0_string

const char modified_bessel_k_0_name[] = "modified_bessel_k_0";

void modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<modified_bessel_k_0_name, scalar_t>(iterator, modified_bessel_k_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_k_1_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k_1(T1 a) {
    return a;
  } // T1 modified_bessel_k_1(T1 a)
); // modified_bessel_k_1_string

const char modified_bessel_k_1_name[] = "modified_bessel_k_1";

void modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<modified_bessel_k_1_name, scalar_t>(iterator, modified_bessel_k_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator)

const auto nome_q_string = jiterator_stringify(
  template<typename T1>
  T1 nome_q(T1 a) {
    return a;
  } // T1 nome_q(T1 a)
); // nome_q_string

const char nome_q_name[] = "nome_q";

void nome_q_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "nome_q_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<nome_q_name, scalar_t>(iterator, nome_q_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "nome_q_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void nome_q_cuda_kernel(TensorIteratorBase &iterator)

const auto prime_number_string = jiterator_stringify(
  template<typename T1>
  T1 prime_number(T1 a) {
    return a;
  } // T1 prime_number(T1 a)
); // prime_number_string

const char prime_number_name[] = "prime_number";

void prime_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "prime_number_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<prime_number_name, scalar_t>(iterator, prime_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "prime_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void prime_number_cuda_kernel(TensorIteratorBase &iterator)

const auto reciprocal_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 reciprocal_gamma(T1 a) {
    return a;
  } // T1 reciprocal_gamma(T1 a)
); // reciprocal_gamma_string

const char reciprocal_gamma_name[] = "reciprocal_gamma";

void reciprocal_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "reciprocal_gamma_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<reciprocal_gamma_name, scalar_t>(iterator, reciprocal_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "reciprocal_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void reciprocal_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto riemann_zeta_string = jiterator_stringify(
  template<typename T1>
  T1 riemann_zeta(T1 a) {
    return a;
  } // T1 riemann_zeta(T1 a)
); // riemann_zeta_string

const char riemann_zeta_name[] = "riemann_zeta";

void riemann_zeta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "riemann_zeta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<riemann_zeta_name, scalar_t>(iterator, riemann_zeta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "riemann_zeta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void riemann_zeta_cuda_kernel(TensorIteratorBase &iterator)

const auto sin_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sin_pi(T1 a) {
    return a;
  } // T1 sin_pi(T1 a)
); // sin_pi_string

const char sin_pi_name[] = "sin_pi";

void sin_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sin_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<sin_pi_name, scalar_t>(iterator, sin_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sin_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void sin_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinc_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sinc_pi(T1 a) {
    return a;
  } // T1 sinc_pi(T1 a)
); // sinc_pi_string

const char sinc_pi_name[] = "sinc_pi";

void sinc_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinc_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<sinc_pi_name, scalar_t>(iterator, sinc_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinc_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void sinc_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinh_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sinh_pi(T1 a) {
    return a;
  } // T1 sinh_pi(T1 a)
); // sinh_pi_string

const char sinh_pi_name[] = "sinh_pi";

void sinh_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinh_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<sinh_pi_name, scalar_t>(iterator, sinh_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinh_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void sinh_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinhc_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sinhc_pi(T1 a) {
    return a;
  } // T1 sinhc_pi(T1 a)
); // sinhc_pi_string

const char sinhc_pi_name[] = "sinhc_pi";

void sinhc_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<sinhc_pi_name, scalar_t>(iterator, sinhc_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void sinhc_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinhc_string = jiterator_stringify(
  template<typename T1>
  T1 sinhc(T1 a) {
    return a;
  } // T1 sinhc(T1 a)
); // sinhc_string

const char sinhc_name[] = "sinhc";

void sinhc_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<sinhc_name, scalar_t>(iterator, sinhc_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void sinhc_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_bessel_j_0_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_j_0(T1 a) {
    return a;
  } // T1 spherical_bessel_j_0(T1 a)
); // spherical_bessel_j_0_string

const char spherical_bessel_j_0_name[] = "spherical_bessel_j_0";

void spherical_bessel_j_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_bessel_j_0_name, scalar_t>(iterator, spherical_bessel_j_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_bessel_j_0_cuda_kernel(TensorIteratorBase &iterator)

const auto tan_pi_string = jiterator_stringify(
  template<typename T1>
  T1 tan_pi(T1 a) {
    return a;
  } // T1 tan_pi(T1 a)
); // tan_pi_string

const char tan_pi_name[] = "tan_pi";

void tan_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tan_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<tan_pi_name, scalar_t>(iterator, tan_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tan_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void tan_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto tanh_pi_string = jiterator_stringify(
  template<typename T1>
  T1 tanh_pi(T1 a) {
    return a;
  } // T1 tanh_pi(T1 a)
); // tanh_pi_string

const char tanh_pi_name[] = "tanh_pi";

void tanh_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tanh_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<tanh_pi_name, scalar_t>(iterator, tanh_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tanh_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void tanh_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto bell_polynomial_b_string = jiterator_stringify(
  template<typename T1>
  T1 bell_polynomial_b(T1 a, T1 b) {
    return a;
  } // T1 bell_polynomial_b(T1 a, T1 b)
); // bell_polynomial_b_string

const char bell_polynomial_b_name[] = "bell_polynomial_b";

void bell_polynomial_b_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bell_polynomial_b_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bell_polynomial_b_name, scalar_t, scalar_t>(iterator, bell_polynomial_b_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bell_polynomial_b_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void bell_polynomial_b_cuda_kernel(TensorIteratorBase &iterator)

const auto bernoulli_polynomial_b_string = jiterator_stringify(
  template<typename T1>
  T1 bernoulli_polynomial_b(T1 a, T1 b) {
    return a;
  } // T1 bernoulli_polynomial_b(T1 a, T1 b)
); // bernoulli_polynomial_b_string

const char bernoulli_polynomial_b_name[] = "bernoulli_polynomial_b";

void bernoulli_polynomial_b_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_polynomial_b_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bernoulli_polynomial_b_name, scalar_t, scalar_t>(iterator, bernoulli_polynomial_b_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_polynomial_b_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void bernoulli_polynomial_b_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_j_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j(T1 a, T1 b) {
    return a;
  } // T1 bessel_j(T1 a, T1 b)
); // bessel_j_string

const char bessel_j_name[] = "bessel_j";

void bessel_j_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bessel_j_name, scalar_t, scalar_t>(iterator, bessel_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void bessel_j_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_y_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_y(T1 a, T1 b) {
    return a;
  } // T1 bessel_y(T1 a, T1 b)
); // bessel_y_string

const char bessel_y_name[] = "bessel_y";

void bessel_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bessel_y_name, scalar_t, scalar_t>(iterator, bessel_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void bessel_y_cuda_kernel(TensorIteratorBase &iterator)

const auto beta_string = jiterator_stringify(
  template<typename T1>
  T1 beta(T1 a, T1 b) {
    return a;
  } // T1 beta(T1 a, T1 b)
); // beta_string

const char beta_name[] = "beta";

void beta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<beta_name, scalar_t, scalar_t>(iterator, beta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void beta_cuda_kernel(TensorIteratorBase &iterator)

const auto binomial_coefficient_string = jiterator_stringify(
  template<typename T1>
  T1 binomial_coefficient(T1 a, T1 b) {
    return a;
  } // T1 binomial_coefficient(T1 a, T1 b)
); // binomial_coefficient_string

const char binomial_coefficient_name[] = "binomial_coefficient";

void binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "binomial_coefficient_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<binomial_coefficient_name, scalar_t, scalar_t>(iterator, binomial_coefficient_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "binomial_coefficient_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator)

const auto bose_einstein_integral_g_string = jiterator_stringify(
  template<typename T1>
  T1 bose_einstein_integral_g(T1 a, T1 b) {
    return a;
  } // T1 bose_einstein_integral_g(T1 a, T1 b)
); // bose_einstein_integral_g_string

const char bose_einstein_integral_g_name[] = "bose_einstein_integral_g";

void bose_einstein_integral_g_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bose_einstein_integral_g_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bose_einstein_integral_g_name, scalar_t, scalar_t>(iterator, bose_einstein_integral_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bose_einstein_integral_g_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void bose_einstein_integral_g_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_el1_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el1(T1 a, T1 b) {
    return a;
  } // T1 bulirsch_elliptic_integral_el1(T1 a, T1 b)
); // bulirsch_elliptic_integral_el1_string

const char bulirsch_elliptic_integral_el1_name[] = "bulirsch_elliptic_integral_el1";

void bulirsch_elliptic_integral_el1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bulirsch_elliptic_integral_el1_name, scalar_t, scalar_t>(iterator, bulirsch_elliptic_integral_el1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void bulirsch_elliptic_integral_el1_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_c_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_c(T1 a, T1 b) {
    return a;
  } // T1 carlson_elliptic_r_c(T1 a, T1 b)
); // carlson_elliptic_r_c_string

const char carlson_elliptic_r_c_name[] = "carlson_elliptic_r_c";

void carlson_elliptic_r_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<carlson_elliptic_r_c_name, scalar_t, scalar_t>(iterator, carlson_elliptic_r_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void carlson_elliptic_r_c_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_t_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_t(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_t(T1 a, T1 b)
); // chebyshev_polynomial_t_string

const char chebyshev_polynomial_t_name[] = "chebyshev_polynomial_t";

void chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<chebyshev_polynomial_t_name, scalar_t, scalar_t>(iterator, chebyshev_polynomial_t_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_u_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_u(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_u(T1 a, T1 b)
); // chebyshev_polynomial_u_string

const char chebyshev_polynomial_u_name[] = "chebyshev_polynomial_u";

void chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_u_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<chebyshev_polynomial_u_name, scalar_t, scalar_t>(iterator, chebyshev_polynomial_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_u_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_v_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_v(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_v(T1 a, T1 b)
); // chebyshev_polynomial_v_string

const char chebyshev_polynomial_v_name[] = "chebyshev_polynomial_v";

void chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_v_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<chebyshev_polynomial_v_name, scalar_t, scalar_t>(iterator, chebyshev_polynomial_v_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_v_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_w_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_w(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_w(T1 a, T1 b)
); // chebyshev_polynomial_w_string

const char chebyshev_polynomial_w_name[] = "chebyshev_polynomial_w";

void chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<chebyshev_polynomial_w_name, scalar_t, scalar_t>(iterator, chebyshev_polynomial_w_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator)

const auto clausen_cl_string = jiterator_stringify(
  template<typename T1>
  T1 clausen_cl(T1 a, T1 b) {
    return a;
  } // T1 clausen_cl(T1 a, T1 b)
); // clausen_cl_string

const char clausen_cl_name[] = "clausen_cl";

void clausen_cl_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_cl_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<clausen_cl_name, scalar_t, scalar_t>(iterator, clausen_cl_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_cl_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void clausen_cl_cuda_kernel(TensorIteratorBase &iterator)

const auto clausen_sl_string = jiterator_stringify(
  template<typename T1>
  T1 clausen_sl(T1 a, T1 b) {
    return a;
  } // T1 clausen_sl(T1 a, T1 b)
); // clausen_sl_string

const char clausen_sl_name[] = "clausen_sl";

void clausen_sl_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_sl_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<clausen_sl_name, scalar_t, scalar_t>(iterator, clausen_sl_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_sl_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void clausen_sl_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_carlson_elliptic_r_f_string = jiterator_stringify(
  template<typename T1>
  T1 complete_carlson_elliptic_r_f(T1 a, T1 b) {
    return a;
  } // T1 complete_carlson_elliptic_r_f(T1 a, T1 b)
); // complete_carlson_elliptic_r_f_string

const char complete_carlson_elliptic_r_f_name[] = "complete_carlson_elliptic_r_f";

void complete_carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_f_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<complete_carlson_elliptic_r_f_name, scalar_t, scalar_t>(iterator, complete_carlson_elliptic_r_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void complete_carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_carlson_elliptic_r_g_string = jiterator_stringify(
  template<typename T1>
  T1 complete_carlson_elliptic_r_g(T1 a, T1 b) {
    return a;
  } // T1 complete_carlson_elliptic_r_g(T1 a, T1 b)
); // complete_carlson_elliptic_r_g_string

const char complete_carlson_elliptic_r_g_name[] = "complete_carlson_elliptic_r_g";

void complete_carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<complete_carlson_elliptic_r_g_name, scalar_t, scalar_t>(iterator, complete_carlson_elliptic_r_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void complete_carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_elliptic_integral_pi_string = jiterator_stringify(
  template<typename T1>
  T1 complete_elliptic_integral_pi(T1 a, T1 b) {
    return a;
  } // T1 complete_elliptic_integral_pi(T1 a, T1 b)
); // complete_elliptic_integral_pi_string

const char complete_elliptic_integral_pi_name[] = "complete_elliptic_integral_pi";

void complete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<complete_elliptic_integral_pi_name, scalar_t, scalar_t>(iterator, complete_elliptic_integral_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void complete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto confluent_hypergeometric_0_f_1_string = jiterator_stringify(
  template<typename T1>
  T1 confluent_hypergeometric_0_f_1(T1 a, T1 b) {
    return a;
  } // T1 confluent_hypergeometric_0_f_1(T1 a, T1 b)
); // confluent_hypergeometric_0_f_1_string

const char confluent_hypergeometric_0_f_1_name[] = "confluent_hypergeometric_0_f_1";

void confluent_hypergeometric_0_f_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "confluent_hypergeometric_0_f_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<confluent_hypergeometric_0_f_1_name, scalar_t, scalar_t>(iterator, confluent_hypergeometric_0_f_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "confluent_hypergeometric_0_f_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void confluent_hypergeometric_0_f_1_cuda_kernel(TensorIteratorBase &iterator)

const auto debye_d_string = jiterator_stringify(
  template<typename T1>
  T1 debye_d(T1 a, T1 b) {
    return a;
  } // T1 debye_d(T1 a, T1 b)
); // debye_d_string

const char debye_d_name[] = "debye_d";

void debye_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "debye_d_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<debye_d_name, scalar_t, scalar_t>(iterator, debye_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "debye_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void debye_d_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_i_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_i(T1 a, T1 b) {
    return a;
  } // T1 exp_modified_bessel_i(T1 a, T1 b)
); // exp_modified_bessel_i_string

const char exp_modified_bessel_i_name[] = "exp_modified_bessel_i";

void exp_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_i_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exp_modified_bessel_i_name, scalar_t, scalar_t>(iterator, exp_modified_bessel_i_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_i_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void exp_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_k(T1 a, T1 b) {
    return a;
  } // T1 exp_modified_bessel_k(T1 a, T1 b)
); // exp_modified_bessel_k_string

const char exp_modified_bessel_k_name[] = "exp_modified_bessel_k";

void exp_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exp_modified_bessel_k_name, scalar_t, scalar_t>(iterator, exp_modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void exp_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator)

const auto exponential_integral_e_string = jiterator_stringify(
  template<typename T1>
  T1 exponential_integral_e(T1 a, T1 b) {
    return a;
  } // T1 exponential_integral_e(T1 a, T1 b)
); // exponential_integral_e_string

const char exponential_integral_e_name[] = "exponential_integral_e";

void exponential_integral_e_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_e_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<exponential_integral_e_name, scalar_t, scalar_t>(iterator, exponential_integral_e_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_e_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void exponential_integral_e_cuda_kernel(TensorIteratorBase &iterator)

const auto falling_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 falling_factorial(T1 a, T1 b) {
    return a;
  } // T1 falling_factorial(T1 a, T1 b)
); // falling_factorial_string

const char falling_factorial_name[] = "falling_factorial";

void falling_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "falling_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<falling_factorial_name, scalar_t, scalar_t>(iterator, falling_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "falling_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void falling_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto fermi_dirac_integral_f_string = jiterator_stringify(
  template<typename T1>
  T1 fermi_dirac_integral_f(T1 a, T1 b) {
    return a;
  } // T1 fermi_dirac_integral_f(T1 a, T1 b)
); // fermi_dirac_integral_f_string

const char fermi_dirac_integral_f_name[] = "fermi_dirac_integral_f";

void fermi_dirac_integral_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fermi_dirac_integral_f_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<fermi_dirac_integral_f_name, scalar_t, scalar_t>(iterator, fermi_dirac_integral_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fermi_dirac_integral_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void fermi_dirac_integral_f_cuda_kernel(TensorIteratorBase &iterator)

const auto hankel_h_1_string = jiterator_stringify(
  template<typename T1>
  T1 hankel_h_1(T1 a, T1 b) {
    return a;
  } // T1 hankel_h_1(T1 a, T1 b)
); // hankel_h_1_string

const char hankel_h_1_name[] = "hankel_h_1";

void hankel_h_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hankel_h_1_name, scalar_t, scalar_t>(iterator, hankel_h_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void hankel_h_1_cuda_kernel(TensorIteratorBase &iterator)

const auto hankel_h_2_string = jiterator_stringify(
  template<typename T1>
  T1 hankel_h_2(T1 a, T1 b) {
    return a;
  } // T1 hankel_h_2(T1 a, T1 b)
); // hankel_h_2_string

const char hankel_h_2_name[] = "hankel_h_2";

void hankel_h_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hankel_h_2_name, scalar_t, scalar_t>(iterator, hankel_h_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void hankel_h_2_cuda_kernel(TensorIteratorBase &iterator)

const auto hermite_polynomial_h_string = jiterator_stringify(
  template<typename T1>
  T1 hermite_polynomial_h(T1 a, T1 b) {
    return a;
  } // T1 hermite_polynomial_h(T1 a, T1 b)
); // hermite_polynomial_h_string

const char hermite_polynomial_h_name[] = "hermite_polynomial_h";

void hermite_polynomial_h_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hermite_polynomial_h_name, scalar_t, scalar_t>(iterator, hermite_polynomial_h_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void hermite_polynomial_h_cuda_kernel(TensorIteratorBase &iterator)

const auto hermite_polynomial_he_string = jiterator_stringify(
  template<typename T1>
  T1 hermite_polynomial_he(T1 a, T1 b) {
    return a;
  } // T1 hermite_polynomial_he(T1 a, T1 b)
); // hermite_polynomial_he_string

const char hermite_polynomial_he_name[] = "hermite_polynomial_he";

void hermite_polynomial_he_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hermite_polynomial_he_name, scalar_t, scalar_t>(iterator, hermite_polynomial_he_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void hermite_polynomial_he_cuda_kernel(TensorIteratorBase &iterator)

const auto heuman_lambda_string = jiterator_stringify(
  template<typename T1>
  T1 heuman_lambda(T1 a, T1 b) {
    return a;
  } // T1 heuman_lambda(T1 a, T1 b)
); // heuman_lambda_string

const char heuman_lambda_name[] = "heuman_lambda";

void heuman_lambda_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "heuman_lambda_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<heuman_lambda_name, scalar_t, scalar_t>(iterator, heuman_lambda_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "heuman_lambda_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void heuman_lambda_cuda_kernel(TensorIteratorBase &iterator)

const auto hurwitz_zeta_string = jiterator_stringify(
  template<typename T1>
  T1 hurwitz_zeta(T1 a, T1 b) {
    return a;
  } // T1 hurwitz_zeta(T1 a, T1 b)
); // hurwitz_zeta_string

const char hurwitz_zeta_name[] = "hurwitz_zeta";

void hurwitz_zeta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hurwitz_zeta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<hurwitz_zeta_name, scalar_t, scalar_t>(iterator, hurwitz_zeta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hurwitz_zeta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void hurwitz_zeta_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_elliptic_integral_e_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_elliptic_integral_e(T1 a, T1 b) {
    return a;
  } // T1 incomplete_elliptic_integral_e(T1 a, T1 b)
); // incomplete_elliptic_integral_e_string

const char incomplete_elliptic_integral_e_name[] = "incomplete_elliptic_integral_e";

void incomplete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_e_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<incomplete_elliptic_integral_e_name, scalar_t, scalar_t>(iterator, incomplete_elliptic_integral_e_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_e_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void incomplete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_elliptic_integral_f_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_elliptic_integral_f(T1 a, T1 b) {
    return a;
  } // T1 incomplete_elliptic_integral_f(T1 a, T1 b)
); // incomplete_elliptic_integral_f_string

const char incomplete_elliptic_integral_f_name[] = "incomplete_elliptic_integral_f";

void incomplete_elliptic_integral_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_f_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<incomplete_elliptic_integral_f_name, scalar_t, scalar_t>(iterator, incomplete_elliptic_integral_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void incomplete_elliptic_integral_f_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_legendre_elliptic_integral_d_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_legendre_elliptic_integral_d(T1 a, T1 b) {
    return a;
  } // T1 incomplete_legendre_elliptic_integral_d(T1 a, T1 b)
); // incomplete_legendre_elliptic_integral_d_string

const char incomplete_legendre_elliptic_integral_d_name[] = "incomplete_legendre_elliptic_integral_d";

void incomplete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<incomplete_legendre_elliptic_integral_d_name, scalar_t, scalar_t>(iterator, incomplete_legendre_elliptic_integral_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void incomplete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_1_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_1(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_1(T1 a, T1 b)
); // jacobi_theta_1_string

const char jacobi_theta_1_name[] = "jacobi_theta_1";

void jacobi_theta_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<jacobi_theta_1_name, scalar_t, scalar_t>(iterator, jacobi_theta_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void jacobi_theta_1_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_2_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_2(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_2(T1 a, T1 b)
); // jacobi_theta_2_string

const char jacobi_theta_2_name[] = "jacobi_theta_2";

void jacobi_theta_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<jacobi_theta_2_name, scalar_t, scalar_t>(iterator, jacobi_theta_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void jacobi_theta_2_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_3_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_3(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_3(T1 a, T1 b)
); // jacobi_theta_3_string

const char jacobi_theta_3_name[] = "jacobi_theta_3";

void jacobi_theta_3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_3_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<jacobi_theta_3_name, scalar_t, scalar_t>(iterator, jacobi_theta_3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void jacobi_theta_3_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_4_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_4(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_4(T1 a, T1 b)
); // jacobi_theta_4_string

const char jacobi_theta_4_name[] = "jacobi_theta_4";

void jacobi_theta_4_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_4_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<jacobi_theta_4_name, scalar_t, scalar_t>(iterator, jacobi_theta_4_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_4_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void jacobi_theta_4_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_zeta_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_zeta(T1 a, T1 b) {
    return a;
  } // T1 jacobi_zeta(T1 a, T1 b)
); // jacobi_zeta_string

const char jacobi_zeta_name[] = "jacobi_zeta";

void jacobi_zeta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_zeta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<jacobi_zeta_name, scalar_t, scalar_t>(iterator, jacobi_zeta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_zeta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void jacobi_zeta_cuda_kernel(TensorIteratorBase &iterator)

const auto laguerre_polynomial_l_string = jiterator_stringify(
  template<typename T1>
  T1 laguerre_polynomial_l(T1 a, T1 b) {
    return a;
  } // T1 laguerre_polynomial_l(T1 a, T1 b)
); // laguerre_polynomial_l_string

const char laguerre_polynomial_l_name[] = "laguerre_polynomial_l";

void laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<laguerre_polynomial_l_name, scalar_t, scalar_t>(iterator, laguerre_polynomial_l_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator)

const auto lah_number_string = jiterator_stringify(
  template<typename T1>
  T1 lah_number(T1 a, T1 b) {
    return a;
  } // T1 lah_number(T1 a, T1 b)
); // lah_number_string

const char lah_number_name[] = "lah_number";

void lah_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lah_number_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<lah_number_name, scalar_t, scalar_t>(iterator, lah_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lah_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void lah_number_cuda_kernel(TensorIteratorBase &iterator)

const auto legendre_polynomial_p_string = jiterator_stringify(
  template<typename T1>
  T1 legendre_polynomial_p(T1 a, T1 b) {
    return a;
  } // T1 legendre_polynomial_p(T1 a, T1 b)
); // legendre_polynomial_p_string

const char legendre_polynomial_p_name[] = "legendre_polynomial_p";

void legendre_polynomial_p_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<legendre_polynomial_p_name, scalar_t, scalar_t>(iterator, legendre_polynomial_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void legendre_polynomial_p_cuda_kernel(TensorIteratorBase &iterator)

const auto legendre_q_string = jiterator_stringify(
  template<typename T1>
  T1 legendre_q(T1 a, T1 b) {
    return a;
  } // T1 legendre_q(T1 a, T1 b)
); // legendre_q_string

const char legendre_q_name[] = "legendre_q";

void legendre_q_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_q_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<legendre_q_name, scalar_t, scalar_t>(iterator, legendre_q_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_q_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void legendre_q_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_binomial_coefficient_string = jiterator_stringify(
  template<typename T1>
  T1 ln_binomial_coefficient(T1 a, T1 b) {
    return a;
  } // T1 ln_binomial_coefficient(T1 a, T1 b)
); // ln_binomial_coefficient_string

const char ln_binomial_coefficient_name[] = "ln_binomial_coefficient";

void ln_binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_binomial_coefficient_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_binomial_coefficient_name, scalar_t, scalar_t>(iterator, ln_binomial_coefficient_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_binomial_coefficient_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_falling_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_falling_factorial(T1 a, T1 b) {
    return a;
  } // T1 ln_falling_factorial(T1 a, T1 b)
); // ln_falling_factorial_string

const char ln_falling_factorial_name[] = "ln_falling_factorial";

void ln_falling_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_falling_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_falling_factorial_name, scalar_t, scalar_t>(iterator, ln_falling_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_falling_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_falling_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_rising_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_rising_factorial(T1 a, T1 b) {
    return a;
  } // T1 ln_rising_factorial(T1 a, T1 b)
); // ln_rising_factorial_string

const char ln_rising_factorial_name[] = "ln_rising_factorial";

void ln_rising_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_rising_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<ln_rising_factorial_name, scalar_t, scalar_t>(iterator, ln_rising_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_rising_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void ln_rising_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto lower_incomplete_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 lower_incomplete_gamma(T1 a, T1 b) {
    return a;
  } // T1 lower_incomplete_gamma(T1 a, T1 b)
); // lower_incomplete_gamma_string

const char lower_incomplete_gamma_name[] = "lower_incomplete_gamma";

void lower_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lower_incomplete_gamma_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<lower_incomplete_gamma_name, scalar_t, scalar_t>(iterator, lower_incomplete_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lower_incomplete_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void lower_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_i_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_i(T1 a, T1 b) {
    return a;
  } // T1 modified_bessel_i(T1 a, T1 b)
); // modified_bessel_i_string

const char modified_bessel_i_name[] = "modified_bessel_i";

void modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<modified_bessel_i_name, scalar_t, scalar_t>(iterator, modified_bessel_i_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k(T1 a, T1 b) {
    return a;
  } // T1 modified_bessel_k(T1 a, T1 b)
); // modified_bessel_k_string

const char modified_bessel_k_name[] = "modified_bessel_k";

void modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<modified_bessel_k_name, scalar_t, scalar_t>(iterator, modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_c_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_c(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_c(T1 a, T1 b)
); // neville_theta_c_string

const char neville_theta_c_name[] = "neville_theta_c";

void neville_theta_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_c_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<neville_theta_c_name, scalar_t, scalar_t>(iterator, neville_theta_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void neville_theta_c_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_d_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_d(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_d(T1 a, T1 b)
); // neville_theta_d_string

const char neville_theta_d_name[] = "neville_theta_d";

void neville_theta_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_d_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<neville_theta_d_name, scalar_t, scalar_t>(iterator, neville_theta_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void neville_theta_d_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_n_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_n(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_n(T1 a, T1 b)
); // neville_theta_n_string

const char neville_theta_n_name[] = "neville_theta_n";

void neville_theta_n_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_n_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<neville_theta_n_name, scalar_t, scalar_t>(iterator, neville_theta_n_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_n_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void neville_theta_n_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_s_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_s(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_s(T1 a, T1 b)
); // neville_theta_s_string

const char neville_theta_s_name[] = "neville_theta_s";

void neville_theta_s_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_s_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<neville_theta_s_name, scalar_t, scalar_t>(iterator, neville_theta_s_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_s_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void neville_theta_s_cuda_kernel(TensorIteratorBase &iterator)

const auto owens_t_string = jiterator_stringify(
  template<typename T1>
  T1 owens_t(T1 a, T1 b) {
    return a;
  } // T1 owens_t(T1 a, T1 b)
); // owens_t_string

const char owens_t_name[] = "owens_t";

void owens_t_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "owens_t_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<owens_t_name, scalar_t, scalar_t>(iterator, owens_t_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "owens_t_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void owens_t_cuda_kernel(TensorIteratorBase &iterator)

const auto polar_pi_string = jiterator_stringify(
  template<typename T1>
  T1 polar_pi(T1 a, T1 b) {
    return a;
  } // T1 polar_pi(T1 a, T1 b)
); // polar_pi_string

const char polar_pi_name[] = "polar_pi";

void polar_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polar_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<polar_pi_name, scalar_t, scalar_t>(iterator, polar_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polar_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void polar_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto polylogarithm_li_string = jiterator_stringify(
  template<typename T1>
  T1 polylogarithm_li(T1 a, T1 b) {
    return a;
  } // T1 polylogarithm_li(T1 a, T1 b)
); // polylogarithm_li_string

const char polylogarithm_li_name[] = "polylogarithm_li";

void polylogarithm_li_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polylogarithm_li_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<polylogarithm_li_name, scalar_t, scalar_t>(iterator, polylogarithm_li_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polylogarithm_li_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void polylogarithm_li_cuda_kernel(TensorIteratorBase &iterator)

const auto rising_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 rising_factorial(T1 a, T1 b) {
    return a;
  } // T1 rising_factorial(T1 a, T1 b)
); // rising_factorial_string

const char rising_factorial_name[] = "rising_factorial";

void rising_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "rising_factorial_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<rising_factorial_name, scalar_t, scalar_t>(iterator, rising_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "rising_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void rising_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_t_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_t(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_t(T1 a, T1 b)
); // shifted_chebyshev_polynomial_t_string

const char shifted_chebyshev_polynomial_t_name[] = "shifted_chebyshev_polynomial_t";

void shifted_chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_t_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_t_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void shifted_chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_u_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_u(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_u(T1 a, T1 b)
); // shifted_chebyshev_polynomial_u_string

const char shifted_chebyshev_polynomial_u_name[] = "shifted_chebyshev_polynomial_u";

void shifted_chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_u_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void shifted_chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_v_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_v(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_v(T1 a, T1 b)
); // shifted_chebyshev_polynomial_v_string

const char shifted_chebyshev_polynomial_v_name[] = "shifted_chebyshev_polynomial_v";

void shifted_chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_v_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_v_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void shifted_chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_w_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_w(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_w(T1 a, T1 b)
); // shifted_chebyshev_polynomial_w_string

const char shifted_chebyshev_polynomial_w_name[] = "shifted_chebyshev_polynomial_w";

void shifted_chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_w_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_w_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void shifted_chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_bessel_j_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_j(T1 a, T1 b) {
    return a;
  } // T1 spherical_bessel_j(T1 a, T1 b)
); // spherical_bessel_j_string

const char spherical_bessel_j_name[] = "spherical_bessel_j";

void spherical_bessel_j_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_bessel_j_name, scalar_t, scalar_t>(iterator, spherical_bessel_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_bessel_j_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_bessel_y_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_y(T1 a, T1 b) {
    return a;
  } // T1 spherical_bessel_y(T1 a, T1 b)
); // spherical_bessel_y_string

const char spherical_bessel_y_name[] = "spherical_bessel_y";

void spherical_bessel_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_y_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_bessel_y_name, scalar_t, scalar_t>(iterator, spherical_bessel_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_bessel_y_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_hankel_h_1_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_hankel_h_1(T1 a, T1 b) {
    return a;
  } // T1 spherical_hankel_h_1(T1 a, T1 b)
); // spherical_hankel_h_1_string

const char spherical_hankel_h_1_name[] = "spherical_hankel_h_1";

void spherical_hankel_h_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_hankel_h_1_name, scalar_t, scalar_t>(iterator, spherical_hankel_h_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_hankel_h_1_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_hankel_h_2_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_hankel_h_2(T1 a, T1 b) {
    return a;
  } // T1 spherical_hankel_h_2(T1 a, T1 b)
); // spherical_hankel_h_2_string

const char spherical_hankel_h_2_name[] = "spherical_hankel_h_2";

void spherical_hankel_h_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_hankel_h_2_name, scalar_t, scalar_t>(iterator, spherical_hankel_h_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_hankel_h_2_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_modified_bessel_i_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_modified_bessel_i(T1 a, T1 b) {
    return a;
  } // T1 spherical_modified_bessel_i(T1 a, T1 b)
); // spherical_modified_bessel_i_string

const char spherical_modified_bessel_i_name[] = "spherical_modified_bessel_i";

void spherical_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_i_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_modified_bessel_i_name, scalar_t, scalar_t>(iterator, spherical_modified_bessel_i_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_i_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_modified_bessel_k(T1 a, T1 b) {
    return a;
  } // T1 spherical_modified_bessel_k(T1 a, T1 b)
); // spherical_modified_bessel_k_string

const char spherical_modified_bessel_k_name[] = "spherical_modified_bessel_k";

void spherical_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_k_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_modified_bessel_k_name, scalar_t, scalar_t>(iterator, spherical_modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator)

const auto stirling_number_1_string = jiterator_stringify(
  template<typename T1>
  T1 stirling_number_1(T1 a, T1 b) {
    return a;
  } // T1 stirling_number_1(T1 a, T1 b)
); // stirling_number_1_string

const char stirling_number_1_name[] = "stirling_number_1";

void stirling_number_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<stirling_number_1_name, scalar_t, scalar_t>(iterator, stirling_number_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void stirling_number_1_cuda_kernel(TensorIteratorBase &iterator)

const auto stirling_number_2_string = jiterator_stringify(
  template<typename T1>
  T1 stirling_number_2(T1 a, T1 b) {
    return a;
  } // T1 stirling_number_2(T1 a, T1 b)
); // stirling_number_2_string

const char stirling_number_2_name[] = "stirling_number_2";

void stirling_number_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<stirling_number_2_name, scalar_t, scalar_t>(iterator, stirling_number_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void stirling_number_2_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_1_string = jiterator_stringify(
  template<typename T1>
  T1 theta_1(T1 a, T1 b) {
    return a;
  } // T1 theta_1(T1 a, T1 b)
); // theta_1_string

const char theta_1_name[] = "theta_1";

void theta_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<theta_1_name, scalar_t, scalar_t>(iterator, theta_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void theta_1_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_2_string = jiterator_stringify(
  template<typename T1>
  T1 theta_2(T1 a, T1 b) {
    return a;
  } // T1 theta_2(T1 a, T1 b)
); // theta_2_string

const char theta_2_name[] = "theta_2";

void theta_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<theta_2_name, scalar_t, scalar_t>(iterator, theta_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void theta_2_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_3_string = jiterator_stringify(
  template<typename T1>
  T1 theta_3(T1 a, T1 b) {
    return a;
  } // T1 theta_3(T1 a, T1 b)
); // theta_3_string

const char theta_3_name[] = "theta_3";

void theta_3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_3_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<theta_3_name, scalar_t, scalar_t>(iterator, theta_3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void theta_3_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_4_string = jiterator_stringify(
  template<typename T1>
  T1 theta_4(T1 a, T1 b) {
    return a;
  } // T1 theta_4(T1 a, T1 b)
); // theta_4_string

const char theta_4_name[] = "theta_4";

void theta_4_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_4_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<theta_4_name, scalar_t, scalar_t>(iterator, theta_4_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_4_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void theta_4_cuda_kernel(TensorIteratorBase &iterator)

const auto upper_incomplete_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 upper_incomplete_gamma(T1 a, T1 b) {
    return a;
  } // T1 upper_incomplete_gamma(T1 a, T1 b)
); // upper_incomplete_gamma_string

const char upper_incomplete_gamma_name[] = "upper_incomplete_gamma";

void upper_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "upper_incomplete_gamma_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<upper_incomplete_gamma_name, scalar_t, scalar_t>(iterator, upper_incomplete_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "upper_incomplete_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void upper_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto associated_laguerre_polynomial_l_string = jiterator_stringify(
  template<typename T1>
  T1 associated_laguerre_polynomial_l(T1 a, T1 b, T1 c) {
    return a;
  } // T1 associated_laguerre_polynomial_l(T1 a, T1 b, T1 c)
); // associated_laguerre_polynomial_l_string

const char associated_laguerre_polynomial_l_name[] = "associated_laguerre_polynomial_l";

void associated_laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_laguerre_polynomial_l_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<associated_laguerre_polynomial_l_name, scalar_t, scalar_t, scalar_t>(iterator, associated_laguerre_polynomial_l_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_laguerre_polynomial_l_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void associated_laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator)

const auto associated_legendre_p_string = jiterator_stringify(
  template<typename T1>
  T1 associated_legendre_p(T1 a, T1 b, T1 c) {
    return a;
  } // T1 associated_legendre_p(T1 a, T1 b, T1 c)
); // associated_legendre_p_string

const char associated_legendre_p_name[] = "associated_legendre_p";

void associated_legendre_p_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_p_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<associated_legendre_p_name, scalar_t, scalar_t, scalar_t>(iterator, associated_legendre_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_p_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void associated_legendre_p_cuda_kernel(TensorIteratorBase &iterator)

const auto associated_legendre_q_string = jiterator_stringify(
  template<typename T1>
  T1 associated_legendre_q(T1 a, T1 b, T1 c) {
    return a;
  } // T1 associated_legendre_q(T1 a, T1 b, T1 c)
); // associated_legendre_q_string

const char associated_legendre_q_name[] = "associated_legendre_q";

void associated_legendre_q_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_q_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<associated_legendre_q_name, scalar_t, scalar_t, scalar_t>(iterator, associated_legendre_q_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_q_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void associated_legendre_q_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_el3_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el3(T1 a, T1 b, T1 c) {
    return a;
  } // T1 bulirsch_elliptic_integral_el3(T1 a, T1 b, T1 c)
); // bulirsch_elliptic_integral_el3_string

const char bulirsch_elliptic_integral_el3_name[] = "bulirsch_elliptic_integral_el3";

void bulirsch_elliptic_integral_el3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el3_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bulirsch_elliptic_integral_el3_name, scalar_t, scalar_t, scalar_t>(iterator, bulirsch_elliptic_integral_el3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void bulirsch_elliptic_integral_el3_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_d_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_d(T1 a, T1 b, T1 c) {
    return a;
  } // T1 carlson_elliptic_r_d(T1 a, T1 b, T1 c)
); // carlson_elliptic_r_d_string

const char carlson_elliptic_r_d_name[] = "carlson_elliptic_r_d";

void carlson_elliptic_r_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_d_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<carlson_elliptic_r_d_name, scalar_t, scalar_t, scalar_t>(iterator, carlson_elliptic_r_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void carlson_elliptic_r_d_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_f_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_f(T1 a, T1 b, T1 c) {
    return a;
  } // T1 carlson_elliptic_r_f(T1 a, T1 b, T1 c)
); // carlson_elliptic_r_f_string

const char carlson_elliptic_r_f_name[] = "carlson_elliptic_r_f";

void carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_f_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<carlson_elliptic_r_f_name, scalar_t, scalar_t, scalar_t>(iterator, carlson_elliptic_r_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_g_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_g(T1 a, T1 b, T1 c) {
    return a;
  } // T1 carlson_elliptic_r_g(T1 a, T1 b, T1 c)
); // carlson_elliptic_r_g_string

const char carlson_elliptic_r_g_name[] = "carlson_elliptic_r_g";

void carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_g_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<carlson_elliptic_r_g_name, scalar_t, scalar_t, scalar_t>(iterator, carlson_elliptic_r_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_g_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator)

const auto gegenbauer_polynomial_c_string = jiterator_stringify(
  template<typename T1>
  T1 gegenbauer_polynomial_c(T1 a, T1 b, T1 c) {
    return a;
  } // T1 gegenbauer_polynomial_c(T1 a, T1 b, T1 c)
); // gegenbauer_polynomial_c_string

const char gegenbauer_polynomial_c_name[] = "gegenbauer_polynomial_c";

void gegenbauer_polynomial_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gegenbauer_polynomial_c_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<gegenbauer_polynomial_c_name, scalar_t, scalar_t, scalar_t>(iterator, gegenbauer_polynomial_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gegenbauer_polynomial_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void gegenbauer_polynomial_c_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_beta_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_beta(T1 a, T1 b, T1 c) {
    return a;
  } // T1 incomplete_beta(T1 a, T1 b, T1 c)
); // incomplete_beta_string

const char incomplete_beta_name[] = "incomplete_beta";

void incomplete_beta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_beta_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<incomplete_beta_name, scalar_t, scalar_t, scalar_t>(iterator, incomplete_beta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_beta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void incomplete_beta_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_elliptic_integral_pi_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_elliptic_integral_pi(T1 a, T1 b, T1 c) {
    return a;
  } // T1 incomplete_elliptic_integral_pi(T1 a, T1 b, T1 c)
); // incomplete_elliptic_integral_pi_string

const char incomplete_elliptic_integral_pi_name[] = "incomplete_elliptic_integral_pi";

void incomplete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_pi_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<incomplete_elliptic_integral_pi_name, scalar_t, scalar_t, scalar_t>(iterator, incomplete_elliptic_integral_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void incomplete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto kummer_confluent_hypergeometric_1_f_1_string = jiterator_stringify(
  template<typename T1>
  T1 kummer_confluent_hypergeometric_1_f_1(T1 a, T1 b, T1 c) {
    return a;
  } // T1 kummer_confluent_hypergeometric_1_f_1(T1 a, T1 b, T1 c)
); // kummer_confluent_hypergeometric_1_f_1_string

const char kummer_confluent_hypergeometric_1_f_1_name[] = "kummer_confluent_hypergeometric_1_f_1";

void kummer_confluent_hypergeometric_1_f_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "kummer_confluent_hypergeometric_1_f_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<kummer_confluent_hypergeometric_1_f_1_name, scalar_t, scalar_t, scalar_t>(iterator, kummer_confluent_hypergeometric_1_f_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "kummer_confluent_hypergeometric_1_f_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void kummer_confluent_hypergeometric_1_f_1_cuda_kernel(TensorIteratorBase &iterator)

const auto radial_polynomial_r_string = jiterator_stringify(
  template<typename T1>
  T1 radial_polynomial_r(T1 a, T1 b, T1 c) {
    return a;
  } // T1 radial_polynomial_r(T1 a, T1 b, T1 c)
); // radial_polynomial_r_string

const char radial_polynomial_r_name[] = "radial_polynomial_r";

void radial_polynomial_r_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "radial_polynomial_r_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<radial_polynomial_r_name, scalar_t, scalar_t, scalar_t>(iterator, radial_polynomial_r_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "radial_polynomial_r_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void radial_polynomial_r_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_legendre_y_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_legendre_y(T1 a, T1 b, T1 c) {
    return a;
  } // T1 spherical_legendre_y(T1 a, T1 b, T1 c)
); // spherical_legendre_y_string

const char spherical_legendre_y_name[] = "spherical_legendre_y";

void spherical_legendre_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_legendre_y_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_legendre_y_name, scalar_t, scalar_t, scalar_t>(iterator, spherical_legendre_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_legendre_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_legendre_y_cuda_kernel(TensorIteratorBase &iterator)

const auto tricomi_confluent_hypergeometric_u_string = jiterator_stringify(
  template<typename T1>
  T1 tricomi_confluent_hypergeometric_u(T1 a, T1 b, T1 c) {
    return a;
  } // T1 tricomi_confluent_hypergeometric_u(T1 a, T1 b, T1 c)
); // tricomi_confluent_hypergeometric_u_string

const char tricomi_confluent_hypergeometric_u_name[] = "tricomi_confluent_hypergeometric_u";

void tricomi_confluent_hypergeometric_u_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tricomi_confluent_hypergeometric_u_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<tricomi_confluent_hypergeometric_u_name, scalar_t, scalar_t, scalar_t>(iterator, tricomi_confluent_hypergeometric_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tricomi_confluent_hypergeometric_u_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void tricomi_confluent_hypergeometric_u_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_cel_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_cel(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 bulirsch_elliptic_integral_cel(T1 a, T1 b, T1 c, T1 d)
); // bulirsch_elliptic_integral_cel_string

const char bulirsch_elliptic_integral_cel_name[] = "bulirsch_elliptic_integral_cel";

void bulirsch_elliptic_integral_cel_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_cel_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bulirsch_elliptic_integral_cel_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, bulirsch_elliptic_integral_cel_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_cel_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void bulirsch_elliptic_integral_cel_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_el2_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el2(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 bulirsch_elliptic_integral_el2(T1 a, T1 b, T1 c, T1 d)
); // bulirsch_elliptic_integral_el2_string

const char bulirsch_elliptic_integral_el2_name[] = "bulirsch_elliptic_integral_el2";

void bulirsch_elliptic_integral_el2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el2_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<bulirsch_elliptic_integral_el2_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, bulirsch_elliptic_integral_el2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void bulirsch_elliptic_integral_el2_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_j_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_j(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 carlson_elliptic_r_j(T1 a, T1 b, T1 c, T1 d)
); // carlson_elliptic_r_j_string

const char carlson_elliptic_r_j_name[] = "carlson_elliptic_r_j";

void carlson_elliptic_r_j_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_j_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<carlson_elliptic_r_j_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, carlson_elliptic_r_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_j_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void carlson_elliptic_r_j_cuda_kernel(TensorIteratorBase &iterator)

const auto gauss_hypergeometric_2_f_1_string = jiterator_stringify(
  template<typename T1>
  T1 gauss_hypergeometric_2_f_1(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 gauss_hypergeometric_2_f_1(T1 a, T1 b, T1 c, T1 d)
); // gauss_hypergeometric_2_f_1_string

const char gauss_hypergeometric_2_f_1_name[] = "gauss_hypergeometric_2_f_1";

void gauss_hypergeometric_2_f_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gauss_hypergeometric_2_f_1_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<gauss_hypergeometric_2_f_1_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, gauss_hypergeometric_2_f_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gauss_hypergeometric_2_f_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void gauss_hypergeometric_2_f_1_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_polynomial_p_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_polynomial_p(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 jacobi_polynomial_p(T1 a, T1 b, T1 c, T1 d)
); // jacobi_polynomial_p_string

const char jacobi_polynomial_p_name[] = "jacobi_polynomial_p";

void jacobi_polynomial_p_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_polynomial_p_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<jacobi_polynomial_p_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, jacobi_polynomial_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_polynomial_p_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void jacobi_polynomial_p_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_harmonic_y_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_harmonic_y(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 spherical_harmonic_y(T1 a, T1 b, T1 c, T1 d)
); // spherical_harmonic_y_string

const char spherical_harmonic_y_name[] = "spherical_harmonic_y";

void spherical_harmonic_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_harmonic_y_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<spherical_harmonic_y_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, spherical_harmonic_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_harmonic_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void spherical_harmonic_y_cuda_kernel(TensorIteratorBase &iterator)

const auto zernike_polynomial_z_string = jiterator_stringify(
  template<typename T1>
  T1 zernike_polynomial_z(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 zernike_polynomial_z(T1 a, T1 b, T1 c, T1 d)
); // zernike_polynomial_z_string

const char zernike_polynomial_z_name[] = "zernike_polynomial_z";

void zernike_polynomial_z_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "zernike_polynomial_z_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<zernike_polynomial_z_name, scalar_t, scalar_t, scalar_t, scalar_t>(iterator, zernike_polynomial_z_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "zernike_polynomial_z_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void zernike_polynomial_z_cuda_kernel(TensorIteratorBase &iterator)
} // namespace native
} // namespace at
