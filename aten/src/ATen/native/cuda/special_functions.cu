#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special_functions.h>

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
namespace {
const auto airy_ai_string = jiterator_stringify(
  template<typename T1>
  T1 airy_ai(T1 a) {
    return a;
  } // T1 airy_ai(T1 a)
); // airy_ai_string

const char airy_ai_name[] = "airy_ai";

void special_airy_ai_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda_kernel", [&]() {
    jitted_gpu_kernel<airy_ai_name, scalar_t, scalar_t, 1>(iterator, airy_ai_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_airy_ai_cuda_kernel(TensorIteratorBase &iterator)

const auto airy_bi_string = jiterator_stringify(
  template<typename T1>
  T1 airy_bi(T1 a) {
    return a;
  } // T1 airy_bi(T1 a)
); // airy_bi_string

const char airy_bi_name[] = "airy_bi";

void special_airy_bi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cuda_kernel", [&]() {
    jitted_gpu_kernel<airy_bi_name, scalar_t, scalar_t, 1>(iterator, airy_bi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_airy_bi_cuda_kernel(TensorIteratorBase &iterator)

const auto bernoulli_number_string = jiterator_stringify(
  template<typename T1>
  T1 bernoulli_number(T1 a) {
    return a;
  } // T1 bernoulli_number(T1 a)
); // bernoulli_number_string

const char bernoulli_number_name[] = "bernoulli_number";

void special_bernoulli_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_number_cuda_kernel", [&]() {
    jitted_gpu_kernel<bernoulli_number_name, scalar_t, scalar_t, 1>(iterator, bernoulli_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bernoulli_number_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_j_0_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j_0(T1 a) {
    return a;
  } // T1 bessel_j_0(T1 a)
); // bessel_j_0_string

const char bessel_j_0_name[] = "bessel_j_0";

void special_bessel_j_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_0_cuda_kernel", [&]() {
    jitted_gpu_kernel<bessel_j_0_name, scalar_t, scalar_t, 1>(iterator, bessel_j_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bessel_j_0_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_j_1_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j_1(T1 a) {
    return a;
  } // T1 bessel_j_1(T1 a)
); // bessel_j_1_string

const char bessel_j_1_name[] = "bessel_j_1";

void special_bessel_j_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<bessel_j_1_name, scalar_t, scalar_t, 1>(iterator, bessel_j_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bessel_j_1_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_y_0_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_y_0(T1 a) {
    return a;
  } // T1 bessel_y_0(T1 a)
); // bessel_y_0_string

const char bessel_y_0_name[] = "bessel_y_0";

void special_bessel_y_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cuda_kernel", [&]() {
    jitted_gpu_kernel<bessel_y_0_name, scalar_t, scalar_t, 1>(iterator, bessel_y_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bessel_y_0_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_y_1_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_y_1(T1 a) {
    return a;
  } // T1 bessel_y_1(T1 a)
); // bessel_y_1_string

const char bessel_y_1_name[] = "bessel_y_1";

void special_bessel_y_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<bessel_y_1_name, scalar_t, scalar_t, 1>(iterator, bessel_y_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bessel_y_1_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_elliptic_integral_e_string = jiterator_stringify(
  template<typename T1>
  T1 complete_elliptic_integral_e(T1 a) {
    return a;
  } // T1 complete_elliptic_integral_e(T1 a)
); // complete_elliptic_integral_e_string

const char complete_elliptic_integral_e_name[] = "complete_elliptic_integral_e";

void special_complete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cuda_kernel", [&]() {
    jitted_gpu_kernel<complete_elliptic_integral_e_name, scalar_t, scalar_t, 1>(iterator, complete_elliptic_integral_e_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_complete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_elliptic_integral_k_string = jiterator_stringify(
  template<typename T1>
  T1 complete_elliptic_integral_k(T1 a) {
    return a;
  } // T1 complete_elliptic_integral_k(T1 a)
); // complete_elliptic_integral_k_string

const char complete_elliptic_integral_k_name[] = "complete_elliptic_integral_k";

void special_complete_elliptic_integral_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_k_cuda_kernel", [&]() {
    jitted_gpu_kernel<complete_elliptic_integral_k_name, scalar_t, scalar_t, 1>(iterator, complete_elliptic_integral_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_complete_elliptic_integral_k_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_legendre_elliptic_integral_d_string = jiterator_stringify(
  template<typename T1>
  T1 complete_legendre_elliptic_integral_d(T1 a) {
    return a;
  } // T1 complete_legendre_elliptic_integral_d(T1 a)
); // complete_legendre_elliptic_integral_d_string

const char complete_legendre_elliptic_integral_d_name[] = "complete_legendre_elliptic_integral_d";

void special_complete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    jitted_gpu_kernel<complete_legendre_elliptic_integral_d_name, scalar_t, scalar_t, 1>(iterator, complete_legendre_elliptic_integral_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_complete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator)

const auto cos_pi_string = jiterator_stringify(
  template<typename T1>
  T1 cos_pi(T1 a) {
    return a;
  } // T1 cos_pi(T1 a)
); // cos_pi_string

const char cos_pi_name[] = "cos_pi";

void special_cos_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cos_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<cos_pi_name, scalar_t, scalar_t, 1>(iterator, cos_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cos_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_cos_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto cosh_pi_string = jiterator_stringify(
  template<typename T1>
  T1 cosh_pi(T1 a) {
    return a;
  } // T1 cosh_pi(T1 a)
); // cosh_pi_string

const char cosh_pi_name[] = "cosh_pi";

void special_cosh_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosh_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<cosh_pi_name, scalar_t, scalar_t, 1>(iterator, cosh_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosh_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_cosh_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto cosine_integral_ci_string = jiterator_stringify(
  template<typename T1>
  T1 cosine_integral_ci(T1 a) {
    return a;
  } // T1 cosine_integral_ci(T1 a)
); // cosine_integral_ci_string

const char cosine_integral_ci_name[] = "cosine_integral_ci";

void special_cosine_integral_ci_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosine_integral_ci_cuda_kernel", [&]() {
    jitted_gpu_kernel<cosine_integral_ci_name, scalar_t, scalar_t, 1>(iterator, cosine_integral_ci_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosine_integral_ci_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_cosine_integral_ci_cuda_kernel(TensorIteratorBase &iterator)

const auto dilogarithm_li_2_string = jiterator_stringify(
  template<typename T1>
  T1 dilogarithm_li_2(T1 a) {
    return a;
  } // T1 dilogarithm_li_2(T1 a)
); // dilogarithm_li_2_string

const char dilogarithm_li_2_name[] = "dilogarithm_li_2";

void special_dilogarithm_li_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dilogarithm_li_2_cuda_kernel", [&]() {
    jitted_gpu_kernel<dilogarithm_li_2_name, scalar_t, scalar_t, 1>(iterator, dilogarithm_li_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dilogarithm_li_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_dilogarithm_li_2_cuda_kernel(TensorIteratorBase &iterator)

const auto dirichlet_beta_string = jiterator_stringify(
  template<typename T1>
  T1 dirichlet_beta(T1 a) {
    return a;
  } // T1 dirichlet_beta(T1 a)
); // dirichlet_beta_string

const char dirichlet_beta_name[] = "dirichlet_beta";

void special_dirichlet_beta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_beta_cuda_kernel", [&]() {
    jitted_gpu_kernel<dirichlet_beta_name, scalar_t, scalar_t, 1>(iterator, dirichlet_beta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_beta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_dirichlet_beta_cuda_kernel(TensorIteratorBase &iterator)

const auto dirichlet_eta_string = jiterator_stringify(
  template<typename T1>
  T1 dirichlet_eta(T1 a) {
    return a;
  } // T1 dirichlet_eta(T1 a)
); // dirichlet_eta_string

const char dirichlet_eta_name[] = "dirichlet_eta";

void special_dirichlet_eta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_eta_cuda_kernel", [&]() {
    jitted_gpu_kernel<dirichlet_eta_name, scalar_t, scalar_t, 1>(iterator, dirichlet_eta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_eta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_dirichlet_eta_cuda_kernel(TensorIteratorBase &iterator)

const auto dirichlet_lambda_string = jiterator_stringify(
  template<typename T1>
  T1 dirichlet_lambda(T1 a) {
    return a;
  } // T1 dirichlet_lambda(T1 a)
); // dirichlet_lambda_string

const char dirichlet_lambda_name[] = "dirichlet_lambda";

void special_dirichlet_lambda_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_lambda_cuda_kernel", [&]() {
    jitted_gpu_kernel<dirichlet_lambda_name, scalar_t, scalar_t, 1>(iterator, dirichlet_lambda_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_lambda_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_dirichlet_lambda_cuda_kernel(TensorIteratorBase &iterator)

const auto double_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 double_factorial(T1 a) {
    return a;
  } // T1 double_factorial(T1 a)
); // double_factorial_string

const char double_factorial_name[] = "double_factorial";

void special_double_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "double_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<double_factorial_name, scalar_t, scalar_t, 1>(iterator, double_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "double_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_double_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_airy_ai_string = jiterator_stringify(
  template<typename T1>
  T1 exp_airy_ai(T1 a) {
    return a;
  } // T1 exp_airy_ai(T1 a)
); // exp_airy_ai_string

const char exp_airy_ai_name[] = "exp_airy_ai";

void special_exp_airy_ai_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_ai_cuda_kernel", [&]() {
    jitted_gpu_kernel<exp_airy_ai_name, scalar_t, scalar_t, 1>(iterator, exp_airy_ai_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_ai_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exp_airy_ai_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_airy_bi_string = jiterator_stringify(
  template<typename T1>
  T1 exp_airy_bi(T1 a) {
    return a;
  } // T1 exp_airy_bi(T1 a)
); // exp_airy_bi_string

const char exp_airy_bi_name[] = "exp_airy_bi";

void special_exp_airy_bi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_bi_cuda_kernel", [&]() {
    jitted_gpu_kernel<exp_airy_bi_name, scalar_t, scalar_t, 1>(iterator, exp_airy_bi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_bi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exp_airy_bi_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_k_0_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_k_0(T1 a) {
    return a;
  } // T1 exp_modified_bessel_k_0(T1 a)
); // exp_modified_bessel_k_0_string

const char exp_modified_bessel_k_0_name[] = "exp_modified_bessel_k_0";

void special_exp_modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_0_cuda_kernel", [&]() {
    jitted_gpu_kernel<exp_modified_bessel_k_0_name, scalar_t, scalar_t, 1>(iterator, exp_modified_bessel_k_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exp_modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_k_1_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_k_1(T1 a) {
    return a;
  } // T1 exp_modified_bessel_k_1(T1 a)
); // exp_modified_bessel_k_1_string

const char exp_modified_bessel_k_1_name[] = "exp_modified_bessel_k_1";

void special_exp_modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<exp_modified_bessel_k_1_name, scalar_t, scalar_t, 1>(iterator, exp_modified_bessel_k_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exp_modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator)

const auto exponential_integral_ei_string = jiterator_stringify(
  template<typename T1>
  T1 exponential_integral_ei(T1 a) {
    return a;
  } // T1 exponential_integral_ei(T1 a)
); // exponential_integral_ei_string

const char exponential_integral_ei_name[] = "exponential_integral_ei";

void special_exponential_integral_ei_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_ei_cuda_kernel", [&]() {
    jitted_gpu_kernel<exponential_integral_ei_name, scalar_t, scalar_t, 1>(iterator, exponential_integral_ei_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_ei_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exponential_integral_ei_cuda_kernel(TensorIteratorBase &iterator)

const auto factorial_string = jiterator_stringify(
  template<typename T1>
  T1 factorial(T1 a) {
    return a;
  } // T1 factorial(T1 a)
); // factorial_string

const char factorial_name[] = "factorial";

void special_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<factorial_name, scalar_t, scalar_t, 1>(iterator, factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto fresnel_integral_c_string = jiterator_stringify(
  template<typename T1>
  T1 fresnel_integral_c(T1 a) {
    return a;
  } // T1 fresnel_integral_c(T1 a)
); // fresnel_integral_c_string

const char fresnel_integral_c_name[] = "fresnel_integral_c";

void special_fresnel_integral_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_c_cuda_kernel", [&]() {
    jitted_gpu_kernel<fresnel_integral_c_name, scalar_t, scalar_t, 1>(iterator, fresnel_integral_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_fresnel_integral_c_cuda_kernel(TensorIteratorBase &iterator)

const auto fresnel_integral_s_string = jiterator_stringify(
  template<typename T1>
  T1 fresnel_integral_s(T1 a) {
    return a;
  } // T1 fresnel_integral_s(T1 a)
); // fresnel_integral_s_string

const char fresnel_integral_s_name[] = "fresnel_integral_s";

void special_fresnel_integral_s_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_s_cuda_kernel", [&]() {
    jitted_gpu_kernel<fresnel_integral_s_name, scalar_t, scalar_t, 1>(iterator, fresnel_integral_s_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_s_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_fresnel_integral_s_cuda_kernel(TensorIteratorBase &iterator)

const auto harmonic_number_string = jiterator_stringify(
  template<typename T1>
  T1 harmonic_number(T1 a) {
    return a;
  } // T1 harmonic_number(T1 a)
); // harmonic_number_string

const char harmonic_number_name[] = "harmonic_number";

void special_harmonic_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cuda_kernel", [&]() {
    jitted_gpu_kernel<harmonic_number_name, scalar_t, scalar_t, 1>(iterator, harmonic_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_harmonic_number_cuda_kernel(TensorIteratorBase &iterator)

const auto hyperbolic_cosine_integral_chi_string = jiterator_stringify(
  template<typename T1>
  T1 hyperbolic_cosine_integral_chi(T1 a) {
    return a;
  } // T1 hyperbolic_cosine_integral_chi(T1 a)
); // hyperbolic_cosine_integral_chi_string

const char hyperbolic_cosine_integral_chi_name[] = "hyperbolic_cosine_integral_chi";

void special_hyperbolic_cosine_integral_chi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_cosine_integral_chi_cuda_kernel", [&]() {
    jitted_gpu_kernel<hyperbolic_cosine_integral_chi_name, scalar_t, scalar_t, 1>(iterator, hyperbolic_cosine_integral_chi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_cosine_integral_chi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hyperbolic_cosine_integral_chi_cuda_kernel(TensorIteratorBase &iterator)

const auto hyperbolic_sine_integral_shi_string = jiterator_stringify(
  template<typename T1>
  T1 hyperbolic_sine_integral_shi(T1 a) {
    return a;
  } // T1 hyperbolic_sine_integral_shi(T1 a)
); // hyperbolic_sine_integral_shi_string

const char hyperbolic_sine_integral_shi_name[] = "hyperbolic_sine_integral_shi";

void special_hyperbolic_sine_integral_shi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_sine_integral_shi_cuda_kernel", [&]() {
    jitted_gpu_kernel<hyperbolic_sine_integral_shi_name, scalar_t, scalar_t, 1>(iterator, hyperbolic_sine_integral_shi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_sine_integral_shi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hyperbolic_sine_integral_shi_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_double_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_double_factorial(T1 a) {
    return a;
  } // T1 ln_double_factorial(T1 a)
); // ln_double_factorial_string

const char ln_double_factorial_name[] = "ln_double_factorial";

void special_ln_double_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_double_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_double_factorial_name, scalar_t, scalar_t, 1>(iterator, ln_double_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_double_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_double_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_factorial(T1 a) {
    return a;
  } // T1 ln_factorial(T1 a)
); // ln_factorial_string

const char ln_factorial_name[] = "ln_factorial";

void special_ln_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_factorial_name, scalar_t, scalar_t, 1>(iterator, ln_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_gamma_sign_string = jiterator_stringify(
  template<typename T1>
  T1 ln_gamma_sign(T1 a) {
    return a;
  } // T1 ln_gamma_sign(T1 a)
); // ln_gamma_sign_string

const char ln_gamma_sign_name[] = "ln_gamma_sign";

void special_ln_gamma_sign_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_sign_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_gamma_sign_name, scalar_t, scalar_t, 1>(iterator, ln_gamma_sign_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_sign_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_gamma_sign_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 ln_gamma(T1 a) {
    return a;
  } // T1 ln_gamma(T1 a)
); // ln_gamma_string

const char ln_gamma_name[] = "ln_gamma";

void special_ln_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_gamma_name, scalar_t, scalar_t, 1>(iterator, ln_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto logarithmic_integral_li_string = jiterator_stringify(
  template<typename T1>
  T1 logarithmic_integral_li(T1 a) {
    return a;
  } // T1 logarithmic_integral_li(T1 a)
); // logarithmic_integral_li_string

const char logarithmic_integral_li_name[] = "logarithmic_integral_li";

void special_logarithmic_integral_li_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "logarithmic_integral_li_cuda_kernel", [&]() {
    jitted_gpu_kernel<logarithmic_integral_li_name, scalar_t, scalar_t, 1>(iterator, logarithmic_integral_li_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "logarithmic_integral_li_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_logarithmic_integral_li_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_i_0_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_i_0(T1 a) {
    return a;
  } // T1 modified_bessel_i_0(T1 a)
); // modified_bessel_i_0_string

const char modified_bessel_i_0_name[] = "modified_bessel_i_0";

void special_modified_bessel_i_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cuda_kernel", [&]() {
    jitted_gpu_kernel<modified_bessel_i_0_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_i_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_modified_bessel_i_0_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_i_1_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_i_1(T1 a) {
    return a;
  } // T1 modified_bessel_i_1(T1 a)
); // modified_bessel_i_1_string

const char modified_bessel_i_1_name[] = "modified_bessel_i_1";

void special_modified_bessel_i_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<modified_bessel_i_1_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_i_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_modified_bessel_i_1_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_k_0_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k_0(T1 a) {
    return a;
  } // T1 modified_bessel_k_0(T1 a)
); // modified_bessel_k_0_string

const char modified_bessel_k_0_name[] = "modified_bessel_k_0";

void special_modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cuda_kernel", [&]() {
    jitted_gpu_kernel<modified_bessel_k_0_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_k_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_modified_bessel_k_0_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_k_1_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k_1(T1 a) {
    return a;
  } // T1 modified_bessel_k_1(T1 a)
); // modified_bessel_k_1_string

const char modified_bessel_k_1_name[] = "modified_bessel_k_1";

void special_modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<modified_bessel_k_1_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_k_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_modified_bessel_k_1_cuda_kernel(TensorIteratorBase &iterator)

const auto nome_q_string = jiterator_stringify(
  template<typename T1>
  T1 nome_q(T1 a) {
    return a;
  } // T1 nome_q(T1 a)
); // nome_q_string

const char nome_q_name[] = "nome_q";

void special_nome_q_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "nome_q_cuda_kernel", [&]() {
    jitted_gpu_kernel<nome_q_name, scalar_t, scalar_t, 1>(iterator, nome_q_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "nome_q_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_nome_q_cuda_kernel(TensorIteratorBase &iterator)

const auto prime_number_string = jiterator_stringify(
  template<typename T1>
  T1 prime_number(T1 a) {
    return a;
  } // T1 prime_number(T1 a)
); // prime_number_string

const char prime_number_name[] = "prime_number";

void special_prime_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "prime_number_cuda_kernel", [&]() {
    jitted_gpu_kernel<prime_number_name, scalar_t, scalar_t, 1>(iterator, prime_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "prime_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_prime_number_cuda_kernel(TensorIteratorBase &iterator)

const auto reciprocal_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 reciprocal_gamma(T1 a) {
    return a;
  } // T1 reciprocal_gamma(T1 a)
); // reciprocal_gamma_string

const char reciprocal_gamma_name[] = "reciprocal_gamma";

void special_reciprocal_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "reciprocal_gamma_cuda_kernel", [&]() {
    jitted_gpu_kernel<reciprocal_gamma_name, scalar_t, scalar_t, 1>(iterator, reciprocal_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "reciprocal_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_reciprocal_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto riemann_zeta_string = jiterator_stringify(
  template<typename T1>
  T1 riemann_zeta(T1 a) {
    return a;
  } // T1 riemann_zeta(T1 a)
); // riemann_zeta_string

const char riemann_zeta_name[] = "riemann_zeta";

void special_riemann_zeta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "riemann_zeta_cuda_kernel", [&]() {
    jitted_gpu_kernel<riemann_zeta_name, scalar_t, scalar_t, 1>(iterator, riemann_zeta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "riemann_zeta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_riemann_zeta_cuda_kernel(TensorIteratorBase &iterator)

const auto sin_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sin_pi(T1 a) {
    return a;
  } // T1 sin_pi(T1 a)
); // sin_pi_string

const char sin_pi_name[] = "sin_pi";

void special_sin_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sin_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<sin_pi_name, scalar_t, scalar_t, 1>(iterator, sin_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sin_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_sin_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinc_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sinc_pi(T1 a) {
    return a;
  } // T1 sinc_pi(T1 a)
); // sinc_pi_string

const char sinc_pi_name[] = "sinc_pi";

void special_sinc_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinc_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<sinc_pi_name, scalar_t, scalar_t, 1>(iterator, sinc_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinc_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_sinc_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinh_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sinh_pi(T1 a) {
    return a;
  } // T1 sinh_pi(T1 a)
); // sinh_pi_string

const char sinh_pi_name[] = "sinh_pi";

void special_sinh_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinh_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<sinh_pi_name, scalar_t, scalar_t, 1>(iterator, sinh_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinh_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_sinh_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinhc_pi_string = jiterator_stringify(
  template<typename T1>
  T1 sinhc_pi(T1 a) {
    return a;
  } // T1 sinhc_pi(T1 a)
); // sinhc_pi_string

const char sinhc_pi_name[] = "sinhc_pi";

void special_sinhc_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<sinhc_pi_name, scalar_t, scalar_t, 1>(iterator, sinhc_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_sinhc_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto sinhc_string = jiterator_stringify(
  template<typename T1>
  T1 sinhc(T1 a) {
    return a;
  } // T1 sinhc(T1 a)
); // sinhc_string

const char sinhc_name[] = "sinhc";

void special_sinhc_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_cuda_kernel", [&]() {
    jitted_gpu_kernel<sinhc_name, scalar_t, scalar_t, 1>(iterator, sinhc_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_sinhc_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_bessel_j_0_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_j_0(T1 a) {
    return a;
  } // T1 spherical_bessel_j_0(T1 a)
); // spherical_bessel_j_0_string

const char spherical_bessel_j_0_name[] = "spherical_bessel_j_0";

void special_spherical_bessel_j_0_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_bessel_j_0_name, scalar_t, scalar_t, 1>(iterator, spherical_bessel_j_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_bessel_j_0_cuda_kernel(TensorIteratorBase &iterator)

const auto tan_pi_string = jiterator_stringify(
  template<typename T1>
  T1 tan_pi(T1 a) {
    return a;
  } // T1 tan_pi(T1 a)
); // tan_pi_string

const char tan_pi_name[] = "tan_pi";

void special_tan_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tan_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<tan_pi_name, scalar_t, scalar_t, 1>(iterator, tan_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tan_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_tan_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto tanh_pi_string = jiterator_stringify(
  template<typename T1>
  T1 tanh_pi(T1 a) {
    return a;
  } // T1 tanh_pi(T1 a)
); // tanh_pi_string

const char tanh_pi_name[] = "tanh_pi";

void special_tanh_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tanh_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<tanh_pi_name, scalar_t, scalar_t, 1>(iterator, tanh_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tanh_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_tanh_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto bell_polynomial_b_string = jiterator_stringify(
  template<typename T1>
  T1 bell_polynomial_b(T1 a, T1 b) {
    return a;
  } // T1 bell_polynomial_b(T1 a, T1 b)
); // bell_polynomial_b_string

const char bell_polynomial_b_name[] = "bell_polynomial_b";

void special_bell_polynomial_b_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bell_polynomial_b_cuda_kernel", [&]() {
    jitted_gpu_kernel<bell_polynomial_b_name, scalar_t, scalar_t, 2>(iterator, bell_polynomial_b_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bell_polynomial_b_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bell_polynomial_b_cuda_kernel(TensorIteratorBase &iterator)

const auto bernoulli_polynomial_b_string = jiterator_stringify(
  template<typename T1>
  T1 bernoulli_polynomial_b(T1 a, T1 b) {
    return a;
  } // T1 bernoulli_polynomial_b(T1 a, T1 b)
); // bernoulli_polynomial_b_string

const char bernoulli_polynomial_b_name[] = "bernoulli_polynomial_b";

void special_bernoulli_polynomial_b_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_polynomial_b_cuda_kernel", [&]() {
    jitted_gpu_kernel<bernoulli_polynomial_b_name, scalar_t, scalar_t, 2>(iterator, bernoulli_polynomial_b_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_polynomial_b_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bernoulli_polynomial_b_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_j_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_j(T1 a, T1 b) {
    return a;
  } // T1 bessel_j(T1 a, T1 b)
); // bessel_j_string

const char bessel_j_name[] = "bessel_j";

void special_bessel_j_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cuda_kernel", [&]() {
    jitted_gpu_kernel<bessel_j_name, scalar_t, scalar_t, 2>(iterator, bessel_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bessel_j_cuda_kernel(TensorIteratorBase &iterator)

const auto bessel_y_string = jiterator_stringify(
  template<typename T1>
  T1 bessel_y(T1 a, T1 b) {
    return a;
  } // T1 bessel_y(T1 a, T1 b)
); // bessel_y_string

const char bessel_y_name[] = "bessel_y";

void special_bessel_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_cuda_kernel", [&]() {
    jitted_gpu_kernel<bessel_y_name, scalar_t, scalar_t, 2>(iterator, bessel_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bessel_y_cuda_kernel(TensorIteratorBase &iterator)

const auto beta_string = jiterator_stringify(
  template<typename T1>
  T1 beta(T1 a, T1 b) {
    return a;
  } // T1 beta(T1 a, T1 b)
); // beta_string

const char beta_name[] = "beta";

void special_beta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cuda_kernel", [&]() {
    jitted_gpu_kernel<beta_name, scalar_t, scalar_t, 2>(iterator, beta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_beta_cuda_kernel(TensorIteratorBase &iterator)

const auto binomial_coefficient_string = jiterator_stringify(
  template<typename T1>
  T1 binomial_coefficient(T1 a, T1 b) {
    return a;
  } // T1 binomial_coefficient(T1 a, T1 b)
); // binomial_coefficient_string

const char binomial_coefficient_name[] = "binomial_coefficient";

void special_binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "binomial_coefficient_cuda_kernel", [&]() {
    jitted_gpu_kernel<binomial_coefficient_name, scalar_t, scalar_t, 2>(iterator, binomial_coefficient_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "binomial_coefficient_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator)

const auto bose_einstein_integral_g_string = jiterator_stringify(
  template<typename T1>
  T1 bose_einstein_integral_g(T1 a, T1 b) {
    return a;
  } // T1 bose_einstein_integral_g(T1 a, T1 b)
); // bose_einstein_integral_g_string

const char bose_einstein_integral_g_name[] = "bose_einstein_integral_g";

void special_bose_einstein_integral_g_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bose_einstein_integral_g_cuda_kernel", [&]() {
    jitted_gpu_kernel<bose_einstein_integral_g_name, scalar_t, scalar_t, 2>(iterator, bose_einstein_integral_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bose_einstein_integral_g_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bose_einstein_integral_g_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_el1_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el1(T1 a, T1 b) {
    return a;
  } // T1 bulirsch_elliptic_integral_el1(T1 a, T1 b)
); // bulirsch_elliptic_integral_el1_string

const char bulirsch_elliptic_integral_el1_name[] = "bulirsch_elliptic_integral_el1";

void special_bulirsch_elliptic_integral_el1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cuda_kernel", [&]() {
    jitted_gpu_kernel<bulirsch_elliptic_integral_el1_name, scalar_t, scalar_t, 2>(iterator, bulirsch_elliptic_integral_el1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bulirsch_elliptic_integral_el1_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_c_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_c(T1 a, T1 b) {
    return a;
  } // T1 carlson_elliptic_r_c(T1 a, T1 b)
); // carlson_elliptic_r_c_string

const char carlson_elliptic_r_c_name[] = "carlson_elliptic_r_c";

void special_carlson_elliptic_r_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cuda_kernel", [&]() {
    jitted_gpu_kernel<carlson_elliptic_r_c_name, scalar_t, scalar_t, 2>(iterator, carlson_elliptic_r_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_carlson_elliptic_r_c_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_t_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_t(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_t(T1 a, T1 b)
); // chebyshev_polynomial_t_string

const char chebyshev_polynomial_t_name[] = "chebyshev_polynomial_t";

void special_chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_cuda_kernel", [&]() {
    jitted_gpu_kernel<chebyshev_polynomial_t_name, scalar_t, scalar_t, 2>(iterator, chebyshev_polynomial_t_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_u_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_u(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_u(T1 a, T1 b)
); // chebyshev_polynomial_u_string

const char chebyshev_polynomial_u_name[] = "chebyshev_polynomial_u";

void special_chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_u_cuda_kernel", [&]() {
    jitted_gpu_kernel<chebyshev_polynomial_u_name, scalar_t, scalar_t, 2>(iterator, chebyshev_polynomial_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_u_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_v_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_v(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_v(T1 a, T1 b)
); // chebyshev_polynomial_v_string

const char chebyshev_polynomial_v_name[] = "chebyshev_polynomial_v";

void special_chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_v_cuda_kernel", [&]() {
    jitted_gpu_kernel<chebyshev_polynomial_v_name, scalar_t, scalar_t, 2>(iterator, chebyshev_polynomial_v_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_v_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator)

const auto chebyshev_polynomial_w_string = jiterator_stringify(
  template<typename T1>
  T1 chebyshev_polynomial_w(T1 a, T1 b) {
    return a;
  } // T1 chebyshev_polynomial_w(T1 a, T1 b)
); // chebyshev_polynomial_w_string

const char chebyshev_polynomial_w_name[] = "chebyshev_polynomial_w";

void special_chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cuda_kernel", [&]() {
    jitted_gpu_kernel<chebyshev_polynomial_w_name, scalar_t, scalar_t, 2>(iterator, chebyshev_polynomial_w_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator)

const auto clausen_cl_string = jiterator_stringify(
  template<typename T1>
  T1 clausen_cl(T1 a, T1 b) {
    return a;
  } // T1 clausen_cl(T1 a, T1 b)
); // clausen_cl_string

const char clausen_cl_name[] = "clausen_cl";

void special_clausen_cl_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_cl_cuda_kernel", [&]() {
    jitted_gpu_kernel<clausen_cl_name, scalar_t, scalar_t, 2>(iterator, clausen_cl_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_cl_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_clausen_cl_cuda_kernel(TensorIteratorBase &iterator)

const auto clausen_sl_string = jiterator_stringify(
  template<typename T1>
  T1 clausen_sl(T1 a, T1 b) {
    return a;
  } // T1 clausen_sl(T1 a, T1 b)
); // clausen_sl_string

const char clausen_sl_name[] = "clausen_sl";

void special_clausen_sl_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_sl_cuda_kernel", [&]() {
    jitted_gpu_kernel<clausen_sl_name, scalar_t, scalar_t, 2>(iterator, clausen_sl_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_sl_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_clausen_sl_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_carlson_elliptic_r_f_string = jiterator_stringify(
  template<typename T1>
  T1 complete_carlson_elliptic_r_f(T1 a, T1 b) {
    return a;
  } // T1 complete_carlson_elliptic_r_f(T1 a, T1 b)
); // complete_carlson_elliptic_r_f_string

const char complete_carlson_elliptic_r_f_name[] = "complete_carlson_elliptic_r_f";

void special_complete_carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_f_cuda_kernel", [&]() {
    jitted_gpu_kernel<complete_carlson_elliptic_r_f_name, scalar_t, scalar_t, 2>(iterator, complete_carlson_elliptic_r_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_complete_carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_carlson_elliptic_r_g_string = jiterator_stringify(
  template<typename T1>
  T1 complete_carlson_elliptic_r_g(T1 a, T1 b) {
    return a;
  } // T1 complete_carlson_elliptic_r_g(T1 a, T1 b)
); // complete_carlson_elliptic_r_g_string

const char complete_carlson_elliptic_r_g_name[] = "complete_carlson_elliptic_r_g";

void special_complete_carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cuda_kernel", [&]() {
    jitted_gpu_kernel<complete_carlson_elliptic_r_g_name, scalar_t, scalar_t, 2>(iterator, complete_carlson_elliptic_r_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_complete_carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator)

const auto complete_elliptic_integral_pi_string = jiterator_stringify(
  template<typename T1>
  T1 complete_elliptic_integral_pi(T1 a, T1 b) {
    return a;
  } // T1 complete_elliptic_integral_pi(T1 a, T1 b)
); // complete_elliptic_integral_pi_string

const char complete_elliptic_integral_pi_name[] = "complete_elliptic_integral_pi";

void special_complete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<complete_elliptic_integral_pi_name, scalar_t, scalar_t, 2>(iterator, complete_elliptic_integral_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_complete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto confluent_hypergeometric_0_f_1_string = jiterator_stringify(
  template<typename T1>
  T1 confluent_hypergeometric_0_f_1(T1 a, T1 b) {
    return a;
  } // T1 confluent_hypergeometric_0_f_1(T1 a, T1 b)
); // confluent_hypergeometric_0_f_1_string

const char confluent_hypergeometric_0_f_1_name[] = "confluent_hypergeometric_0_f_1";

void special_confluent_hypergeometric_0_f_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "confluent_hypergeometric_0_f_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<confluent_hypergeometric_0_f_1_name, scalar_t, scalar_t, 2>(iterator, confluent_hypergeometric_0_f_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "confluent_hypergeometric_0_f_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_confluent_hypergeometric_0_f_1_cuda_kernel(TensorIteratorBase &iterator)

const auto debye_d_string = jiterator_stringify(
  template<typename T1>
  T1 debye_d(T1 a, T1 b) {
    return a;
  } // T1 debye_d(T1 a, T1 b)
); // debye_d_string

const char debye_d_name[] = "debye_d";

void special_debye_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "debye_d_cuda_kernel", [&]() {
    jitted_gpu_kernel<debye_d_name, scalar_t, scalar_t, 2>(iterator, debye_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "debye_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_debye_d_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_i_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_i(T1 a, T1 b) {
    return a;
  } // T1 exp_modified_bessel_i(T1 a, T1 b)
); // exp_modified_bessel_i_string

const char exp_modified_bessel_i_name[] = "exp_modified_bessel_i";

void special_exp_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_i_cuda_kernel", [&]() {
    jitted_gpu_kernel<exp_modified_bessel_i_name, scalar_t, scalar_t, 2>(iterator, exp_modified_bessel_i_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_i_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exp_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator)

const auto exp_modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 exp_modified_bessel_k(T1 a, T1 b) {
    return a;
  } // T1 exp_modified_bessel_k(T1 a, T1 b)
); // exp_modified_bessel_k_string

const char exp_modified_bessel_k_name[] = "exp_modified_bessel_k";

void special_exp_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_cuda_kernel", [&]() {
    jitted_gpu_kernel<exp_modified_bessel_k_name, scalar_t, scalar_t, 2>(iterator, exp_modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exp_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator)

const auto exponential_integral_e_string = jiterator_stringify(
  template<typename T1>
  T1 exponential_integral_e(T1 a, T1 b) {
    return a;
  } // T1 exponential_integral_e(T1 a, T1 b)
); // exponential_integral_e_string

const char exponential_integral_e_name[] = "exponential_integral_e";

void special_exponential_integral_e_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_e_cuda_kernel", [&]() {
    jitted_gpu_kernel<exponential_integral_e_name, scalar_t, scalar_t, 2>(iterator, exponential_integral_e_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_e_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_exponential_integral_e_cuda_kernel(TensorIteratorBase &iterator)

const auto falling_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 falling_factorial(T1 a, T1 b) {
    return a;
  } // T1 falling_factorial(T1 a, T1 b)
); // falling_factorial_string

const char falling_factorial_name[] = "falling_factorial";

void special_falling_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "falling_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<falling_factorial_name, scalar_t, scalar_t, 2>(iterator, falling_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "falling_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_falling_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto fermi_dirac_integral_f_string = jiterator_stringify(
  template<typename T1>
  T1 fermi_dirac_integral_f(T1 a, T1 b) {
    return a;
  } // T1 fermi_dirac_integral_f(T1 a, T1 b)
); // fermi_dirac_integral_f_string

const char fermi_dirac_integral_f_name[] = "fermi_dirac_integral_f";

void special_fermi_dirac_integral_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fermi_dirac_integral_f_cuda_kernel", [&]() {
    jitted_gpu_kernel<fermi_dirac_integral_f_name, scalar_t, scalar_t, 2>(iterator, fermi_dirac_integral_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fermi_dirac_integral_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_fermi_dirac_integral_f_cuda_kernel(TensorIteratorBase &iterator)

const auto hankel_h_1_string = jiterator_stringify(
  template<typename T1>
  T1 hankel_h_1(T1 a, T1 b) {
    return a;
  } // T1 hankel_h_1(T1 a, T1 b)
); // hankel_h_1_string

const char hankel_h_1_name[] = "hankel_h_1";

void special_hankel_h_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<hankel_h_1_name, scalar_t, scalar_t, 2>(iterator, hankel_h_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hankel_h_1_cuda_kernel(TensorIteratorBase &iterator)

const auto hankel_h_2_string = jiterator_stringify(
  template<typename T1>
  T1 hankel_h_2(T1 a, T1 b) {
    return a;
  } // T1 hankel_h_2(T1 a, T1 b)
); // hankel_h_2_string

const char hankel_h_2_name[] = "hankel_h_2";

void special_hankel_h_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cuda_kernel", [&]() {
    jitted_gpu_kernel<hankel_h_2_name, scalar_t, scalar_t, 2>(iterator, hankel_h_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hankel_h_2_cuda_kernel(TensorIteratorBase &iterator)

const auto hermite_polynomial_h_string = jiterator_stringify(
  template<typename T1>
  T1 hermite_polynomial_h(T1 a, T1 b) {
    return a;
  } // T1 hermite_polynomial_h(T1 a, T1 b)
); // hermite_polynomial_h_string

const char hermite_polynomial_h_name[] = "hermite_polynomial_h";

void special_hermite_polynomial_h_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cuda_kernel", [&]() {
    jitted_gpu_kernel<hermite_polynomial_h_name, scalar_t, scalar_t, 2>(iterator, hermite_polynomial_h_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hermite_polynomial_h_cuda_kernel(TensorIteratorBase &iterator)

const auto hermite_polynomial_he_string = jiterator_stringify(
  template<typename T1>
  T1 hermite_polynomial_he(T1 a, T1 b) {
    return a;
  } // T1 hermite_polynomial_he(T1 a, T1 b)
); // hermite_polynomial_he_string

const char hermite_polynomial_he_name[] = "hermite_polynomial_he";

void special_hermite_polynomial_he_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cuda_kernel", [&]() {
    jitted_gpu_kernel<hermite_polynomial_he_name, scalar_t, scalar_t, 2>(iterator, hermite_polynomial_he_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hermite_polynomial_he_cuda_kernel(TensorIteratorBase &iterator)

const auto heuman_lambda_string = jiterator_stringify(
  template<typename T1>
  T1 heuman_lambda(T1 a, T1 b) {
    return a;
  } // T1 heuman_lambda(T1 a, T1 b)
); // heuman_lambda_string

const char heuman_lambda_name[] = "heuman_lambda";

void special_heuman_lambda_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "heuman_lambda_cuda_kernel", [&]() {
    jitted_gpu_kernel<heuman_lambda_name, scalar_t, scalar_t, 2>(iterator, heuman_lambda_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "heuman_lambda_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_heuman_lambda_cuda_kernel(TensorIteratorBase &iterator)

const auto hurwitz_zeta_string = jiterator_stringify(
  template<typename T1>
  T1 hurwitz_zeta(T1 a, T1 b) {
    return a;
  } // T1 hurwitz_zeta(T1 a, T1 b)
); // hurwitz_zeta_string

const char hurwitz_zeta_name[] = "hurwitz_zeta";

void special_hurwitz_zeta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hurwitz_zeta_cuda_kernel", [&]() {
    jitted_gpu_kernel<hurwitz_zeta_name, scalar_t, scalar_t, 2>(iterator, hurwitz_zeta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hurwitz_zeta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_hurwitz_zeta_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_elliptic_integral_e_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_elliptic_integral_e(T1 a, T1 b) {
    return a;
  } // T1 incomplete_elliptic_integral_e(T1 a, T1 b)
); // incomplete_elliptic_integral_e_string

const char incomplete_elliptic_integral_e_name[] = "incomplete_elliptic_integral_e";

void special_incomplete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_e_cuda_kernel", [&]() {
    jitted_gpu_kernel<incomplete_elliptic_integral_e_name, scalar_t, scalar_t, 2>(iterator, incomplete_elliptic_integral_e_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_e_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_incomplete_elliptic_integral_e_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_elliptic_integral_f_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_elliptic_integral_f(T1 a, T1 b) {
    return a;
  } // T1 incomplete_elliptic_integral_f(T1 a, T1 b)
); // incomplete_elliptic_integral_f_string

const char incomplete_elliptic_integral_f_name[] = "incomplete_elliptic_integral_f";

void special_incomplete_elliptic_integral_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_f_cuda_kernel", [&]() {
    jitted_gpu_kernel<incomplete_elliptic_integral_f_name, scalar_t, scalar_t, 2>(iterator, incomplete_elliptic_integral_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_incomplete_elliptic_integral_f_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_legendre_elliptic_integral_d_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_legendre_elliptic_integral_d(T1 a, T1 b) {
    return a;
  } // T1 incomplete_legendre_elliptic_integral_d(T1 a, T1 b)
); // incomplete_legendre_elliptic_integral_d_string

const char incomplete_legendre_elliptic_integral_d_name[] = "incomplete_legendre_elliptic_integral_d";

void special_incomplete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    jitted_gpu_kernel<incomplete_legendre_elliptic_integral_d_name, scalar_t, scalar_t, 2>(iterator, incomplete_legendre_elliptic_integral_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_legendre_elliptic_integral_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_incomplete_legendre_elliptic_integral_d_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_1_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_1(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_1(T1 a, T1 b)
); // jacobi_theta_1_string

const char jacobi_theta_1_name[] = "jacobi_theta_1";

void special_jacobi_theta_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<jacobi_theta_1_name, scalar_t, scalar_t, 2>(iterator, jacobi_theta_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_jacobi_theta_1_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_2_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_2(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_2(T1 a, T1 b)
); // jacobi_theta_2_string

const char jacobi_theta_2_name[] = "jacobi_theta_2";

void special_jacobi_theta_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_2_cuda_kernel", [&]() {
    jitted_gpu_kernel<jacobi_theta_2_name, scalar_t, scalar_t, 2>(iterator, jacobi_theta_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_jacobi_theta_2_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_3_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_3(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_3(T1 a, T1 b)
); // jacobi_theta_3_string

const char jacobi_theta_3_name[] = "jacobi_theta_3";

void special_jacobi_theta_3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_3_cuda_kernel", [&]() {
    jitted_gpu_kernel<jacobi_theta_3_name, scalar_t, scalar_t, 2>(iterator, jacobi_theta_3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_jacobi_theta_3_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_theta_4_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_theta_4(T1 a, T1 b) {
    return a;
  } // T1 jacobi_theta_4(T1 a, T1 b)
); // jacobi_theta_4_string

const char jacobi_theta_4_name[] = "jacobi_theta_4";

void special_jacobi_theta_4_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_4_cuda_kernel", [&]() {
    jitted_gpu_kernel<jacobi_theta_4_name, scalar_t, scalar_t, 2>(iterator, jacobi_theta_4_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_4_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_jacobi_theta_4_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_zeta_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_zeta(T1 a, T1 b) {
    return a;
  } // T1 jacobi_zeta(T1 a, T1 b)
); // jacobi_zeta_string

const char jacobi_zeta_name[] = "jacobi_zeta";

void special_jacobi_zeta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_zeta_cuda_kernel", [&]() {
    jitted_gpu_kernel<jacobi_zeta_name, scalar_t, scalar_t, 2>(iterator, jacobi_zeta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_zeta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_jacobi_zeta_cuda_kernel(TensorIteratorBase &iterator)

const auto laguerre_polynomial_l_string = jiterator_stringify(
  template<typename T1>
  T1 laguerre_polynomial_l(T1 a, T1 b) {
    return a;
  } // T1 laguerre_polynomial_l(T1 a, T1 b)
); // laguerre_polynomial_l_string

const char laguerre_polynomial_l_name[] = "laguerre_polynomial_l";

void special_laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda_kernel", [&]() {
    jitted_gpu_kernel<laguerre_polynomial_l_name, scalar_t, scalar_t, 2>(iterator, laguerre_polynomial_l_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator)

const auto lah_number_string = jiterator_stringify(
  template<typename T1>
  T1 lah_number(T1 a, T1 b) {
    return a;
  } // T1 lah_number(T1 a, T1 b)
); // lah_number_string

const char lah_number_name[] = "lah_number";

void special_lah_number_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lah_number_cuda_kernel", [&]() {
    jitted_gpu_kernel<lah_number_name, scalar_t, scalar_t, 2>(iterator, lah_number_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lah_number_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_lah_number_cuda_kernel(TensorIteratorBase &iterator)

const auto legendre_polynomial_p_string = jiterator_stringify(
  template<typename T1>
  T1 legendre_polynomial_p(T1 a, T1 b) {
    return a;
  } // T1 legendre_polynomial_p(T1 a, T1 b)
); // legendre_polynomial_p_string

const char legendre_polynomial_p_name[] = "legendre_polynomial_p";

void special_legendre_polynomial_p_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda_kernel", [&]() {
    jitted_gpu_kernel<legendre_polynomial_p_name, scalar_t, scalar_t, 2>(iterator, legendre_polynomial_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_legendre_polynomial_p_cuda_kernel(TensorIteratorBase &iterator)

const auto legendre_q_string = jiterator_stringify(
  template<typename T1>
  T1 legendre_q(T1 a, T1 b) {
    return a;
  } // T1 legendre_q(T1 a, T1 b)
); // legendre_q_string

const char legendre_q_name[] = "legendre_q";

void special_legendre_q_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_q_cuda_kernel", [&]() {
    jitted_gpu_kernel<legendre_q_name, scalar_t, scalar_t, 2>(iterator, legendre_q_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_q_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_legendre_q_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_binomial_coefficient_string = jiterator_stringify(
  template<typename T1>
  T1 ln_binomial_coefficient(T1 a, T1 b) {
    return a;
  } // T1 ln_binomial_coefficient(T1 a, T1 b)
); // ln_binomial_coefficient_string

const char ln_binomial_coefficient_name[] = "ln_binomial_coefficient";

void special_ln_binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_binomial_coefficient_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_binomial_coefficient_name, scalar_t, scalar_t, 2>(iterator, ln_binomial_coefficient_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_binomial_coefficient_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_binomial_coefficient_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_falling_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_falling_factorial(T1 a, T1 b) {
    return a;
  } // T1 ln_falling_factorial(T1 a, T1 b)
); // ln_falling_factorial_string

const char ln_falling_factorial_name[] = "ln_falling_factorial";

void special_ln_falling_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_falling_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_falling_factorial_name, scalar_t, scalar_t, 2>(iterator, ln_falling_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_falling_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_falling_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto ln_rising_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 ln_rising_factorial(T1 a, T1 b) {
    return a;
  } // T1 ln_rising_factorial(T1 a, T1 b)
); // ln_rising_factorial_string

const char ln_rising_factorial_name[] = "ln_rising_factorial";

void special_ln_rising_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_rising_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<ln_rising_factorial_name, scalar_t, scalar_t, 2>(iterator, ln_rising_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_rising_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_ln_rising_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto lower_incomplete_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 lower_incomplete_gamma(T1 a, T1 b) {
    return a;
  } // T1 lower_incomplete_gamma(T1 a, T1 b)
); // lower_incomplete_gamma_string

const char lower_incomplete_gamma_name[] = "lower_incomplete_gamma";

void special_lower_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lower_incomplete_gamma_cuda_kernel", [&]() {
    jitted_gpu_kernel<lower_incomplete_gamma_name, scalar_t, scalar_t, 2>(iterator, lower_incomplete_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lower_incomplete_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_lower_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_i_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_i(T1 a, T1 b) {
    return a;
  } // T1 modified_bessel_i(T1 a, T1 b)
); // modified_bessel_i_string

const char modified_bessel_i_name[] = "modified_bessel_i";

void special_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_cuda_kernel", [&]() {
    jitted_gpu_kernel<modified_bessel_i_name, scalar_t, scalar_t, 2>(iterator, modified_bessel_i_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator)

const auto modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 modified_bessel_k(T1 a, T1 b) {
    return a;
  } // T1 modified_bessel_k(T1 a, T1 b)
); // modified_bessel_k_string

const char modified_bessel_k_name[] = "modified_bessel_k";

void special_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cuda_kernel", [&]() {
    jitted_gpu_kernel<modified_bessel_k_name, scalar_t, scalar_t, 2>(iterator, modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_c_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_c(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_c(T1 a, T1 b)
); // neville_theta_c_string

const char neville_theta_c_name[] = "neville_theta_c";

void special_neville_theta_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_c_cuda_kernel", [&]() {
    jitted_gpu_kernel<neville_theta_c_name, scalar_t, scalar_t, 2>(iterator, neville_theta_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_neville_theta_c_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_d_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_d(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_d(T1 a, T1 b)
); // neville_theta_d_string

const char neville_theta_d_name[] = "neville_theta_d";

void special_neville_theta_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_d_cuda_kernel", [&]() {
    jitted_gpu_kernel<neville_theta_d_name, scalar_t, scalar_t, 2>(iterator, neville_theta_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_neville_theta_d_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_n_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_n(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_n(T1 a, T1 b)
); // neville_theta_n_string

const char neville_theta_n_name[] = "neville_theta_n";

void special_neville_theta_n_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_n_cuda_kernel", [&]() {
    jitted_gpu_kernel<neville_theta_n_name, scalar_t, scalar_t, 2>(iterator, neville_theta_n_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_n_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_neville_theta_n_cuda_kernel(TensorIteratorBase &iterator)

const auto neville_theta_s_string = jiterator_stringify(
  template<typename T1>
  T1 neville_theta_s(T1 a, T1 b) {
    return a;
  } // T1 neville_theta_s(T1 a, T1 b)
); // neville_theta_s_string

const char neville_theta_s_name[] = "neville_theta_s";

void special_neville_theta_s_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_s_cuda_kernel", [&]() {
    jitted_gpu_kernel<neville_theta_s_name, scalar_t, scalar_t, 2>(iterator, neville_theta_s_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_s_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_neville_theta_s_cuda_kernel(TensorIteratorBase &iterator)

const auto owens_t_string = jiterator_stringify(
  template<typename T1>
  T1 owens_t(T1 a, T1 b) {
    return a;
  } // T1 owens_t(T1 a, T1 b)
); // owens_t_string

const char owens_t_name[] = "owens_t";

void special_owens_t_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "owens_t_cuda_kernel", [&]() {
    jitted_gpu_kernel<owens_t_name, scalar_t, scalar_t, 2>(iterator, owens_t_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "owens_t_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_owens_t_cuda_kernel(TensorIteratorBase &iterator)

const auto polar_pi_string = jiterator_stringify(
  template<typename T1>
  T1 polar_pi(T1 a, T1 b) {
    return a;
  } // T1 polar_pi(T1 a, T1 b)
); // polar_pi_string

const char polar_pi_name[] = "polar_pi";

void special_polar_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polar_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<polar_pi_name, scalar_t, scalar_t, 2>(iterator, polar_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polar_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_polar_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto polylogarithm_li_string = jiterator_stringify(
  template<typename T1>
  T1 polylogarithm_li(T1 a, T1 b) {
    return a;
  } // T1 polylogarithm_li(T1 a, T1 b)
); // polylogarithm_li_string

const char polylogarithm_li_name[] = "polylogarithm_li";

void special_polylogarithm_li_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polylogarithm_li_cuda_kernel", [&]() {
    jitted_gpu_kernel<polylogarithm_li_name, scalar_t, scalar_t, 2>(iterator, polylogarithm_li_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polylogarithm_li_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_polylogarithm_li_cuda_kernel(TensorIteratorBase &iterator)

const auto rising_factorial_string = jiterator_stringify(
  template<typename T1>
  T1 rising_factorial(T1 a, T1 b) {
    return a;
  } // T1 rising_factorial(T1 a, T1 b)
); // rising_factorial_string

const char rising_factorial_name[] = "rising_factorial";

void special_rising_factorial_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "rising_factorial_cuda_kernel", [&]() {
    jitted_gpu_kernel<rising_factorial_name, scalar_t, scalar_t, 2>(iterator, rising_factorial_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "rising_factorial_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_rising_factorial_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_t_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_t(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_t(T1 a, T1 b)
); // shifted_chebyshev_polynomial_t_string

const char shifted_chebyshev_polynomial_t_name[] = "shifted_chebyshev_polynomial_t";

void special_shifted_chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cuda_kernel", [&]() {
    jitted_gpu_kernel<shifted_chebyshev_polynomial_t_name, scalar_t, scalar_t, 2>(iterator, shifted_chebyshev_polynomial_t_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_shifted_chebyshev_polynomial_t_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_u_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_u(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_u(T1 a, T1 b)
); // shifted_chebyshev_polynomial_u_string

const char shifted_chebyshev_polynomial_u_name[] = "shifted_chebyshev_polynomial_u";

void special_shifted_chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda_kernel", [&]() {
    jitted_gpu_kernel<shifted_chebyshev_polynomial_u_name, scalar_t, scalar_t, 2>(iterator, shifted_chebyshev_polynomial_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_shifted_chebyshev_polynomial_u_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_v_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_v(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_v(T1 a, T1 b)
); // shifted_chebyshev_polynomial_v_string

const char shifted_chebyshev_polynomial_v_name[] = "shifted_chebyshev_polynomial_v";

void special_shifted_chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cuda_kernel", [&]() {
    jitted_gpu_kernel<shifted_chebyshev_polynomial_v_name, scalar_t, scalar_t, 2>(iterator, shifted_chebyshev_polynomial_v_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_shifted_chebyshev_polynomial_v_cuda_kernel(TensorIteratorBase &iterator)

const auto shifted_chebyshev_polynomial_w_string = jiterator_stringify(
  template<typename T1>
  T1 shifted_chebyshev_polynomial_w(T1 a, T1 b) {
    return a;
  } // T1 shifted_chebyshev_polynomial_w(T1 a, T1 b)
); // shifted_chebyshev_polynomial_w_string

const char shifted_chebyshev_polynomial_w_name[] = "shifted_chebyshev_polynomial_w";

void special_shifted_chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cuda_kernel", [&]() {
    jitted_gpu_kernel<shifted_chebyshev_polynomial_w_name, scalar_t, scalar_t, 2>(iterator, shifted_chebyshev_polynomial_w_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_shifted_chebyshev_polynomial_w_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_bessel_j_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_j(T1 a, T1 b) {
    return a;
  } // T1 spherical_bessel_j(T1 a, T1 b)
); // spherical_bessel_j_string

const char spherical_bessel_j_name[] = "spherical_bessel_j";

void special_spherical_bessel_j_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_bessel_j_name, scalar_t, scalar_t, 2>(iterator, spherical_bessel_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_bessel_j_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_bessel_y_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_bessel_y(T1 a, T1 b) {
    return a;
  } // T1 spherical_bessel_y(T1 a, T1 b)
); // spherical_bessel_y_string

const char spherical_bessel_y_name[] = "spherical_bessel_y";

void special_spherical_bessel_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_y_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_bessel_y_name, scalar_t, scalar_t, 2>(iterator, spherical_bessel_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_bessel_y_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_hankel_h_1_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_hankel_h_1(T1 a, T1 b) {
    return a;
  } // T1 spherical_hankel_h_1(T1 a, T1 b)
); // spherical_hankel_h_1_string

const char spherical_hankel_h_1_name[] = "spherical_hankel_h_1";

void special_spherical_hankel_h_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_hankel_h_1_name, scalar_t, scalar_t, 2>(iterator, spherical_hankel_h_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_hankel_h_1_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_hankel_h_2_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_hankel_h_2(T1 a, T1 b) {
    return a;
  } // T1 spherical_hankel_h_2(T1 a, T1 b)
); // spherical_hankel_h_2_string

const char spherical_hankel_h_2_name[] = "spherical_hankel_h_2";

void special_spherical_hankel_h_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_2_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_hankel_h_2_name, scalar_t, scalar_t, 2>(iterator, spherical_hankel_h_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_hankel_h_2_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_modified_bessel_i_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_modified_bessel_i(T1 a, T1 b) {
    return a;
  } // T1 spherical_modified_bessel_i(T1 a, T1 b)
); // spherical_modified_bessel_i_string

const char spherical_modified_bessel_i_name[] = "spherical_modified_bessel_i";

void special_spherical_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_i_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_modified_bessel_i_name, scalar_t, scalar_t, 2>(iterator, spherical_modified_bessel_i_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_i_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_modified_bessel_i_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_modified_bessel_k_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_modified_bessel_k(T1 a, T1 b) {
    return a;
  } // T1 spherical_modified_bessel_k(T1 a, T1 b)
); // spherical_modified_bessel_k_string

const char spherical_modified_bessel_k_name[] = "spherical_modified_bessel_k";

void special_spherical_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_k_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_modified_bessel_k_name, scalar_t, scalar_t, 2>(iterator, spherical_modified_bessel_k_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_k_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_modified_bessel_k_cuda_kernel(TensorIteratorBase &iterator)

const auto stirling_number_1_string = jiterator_stringify(
  template<typename T1>
  T1 stirling_number_1(T1 a, T1 b) {
    return a;
  } // T1 stirling_number_1(T1 a, T1 b)
); // stirling_number_1_string

const char stirling_number_1_name[] = "stirling_number_1";

void special_stirling_number_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<stirling_number_1_name, scalar_t, scalar_t, 2>(iterator, stirling_number_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_stirling_number_1_cuda_kernel(TensorIteratorBase &iterator)

const auto stirling_number_2_string = jiterator_stringify(
  template<typename T1>
  T1 stirling_number_2(T1 a, T1 b) {
    return a;
  } // T1 stirling_number_2(T1 a, T1 b)
); // stirling_number_2_string

const char stirling_number_2_name[] = "stirling_number_2";

void special_stirling_number_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_2_cuda_kernel", [&]() {
    jitted_gpu_kernel<stirling_number_2_name, scalar_t, scalar_t, 2>(iterator, stirling_number_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_stirling_number_2_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_1_string = jiterator_stringify(
  template<typename T1>
  T1 theta_1(T1 a, T1 b) {
    return a;
  } // T1 theta_1(T1 a, T1 b)
); // theta_1_string

const char theta_1_name[] = "theta_1";

void special_theta_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<theta_1_name, scalar_t, scalar_t, 2>(iterator, theta_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_theta_1_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_2_string = jiterator_stringify(
  template<typename T1>
  T1 theta_2(T1 a, T1 b) {
    return a;
  } // T1 theta_2(T1 a, T1 b)
); // theta_2_string

const char theta_2_name[] = "theta_2";

void special_theta_2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_2_cuda_kernel", [&]() {
    jitted_gpu_kernel<theta_2_name, scalar_t, scalar_t, 2>(iterator, theta_2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_theta_2_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_3_string = jiterator_stringify(
  template<typename T1>
  T1 theta_3(T1 a, T1 b) {
    return a;
  } // T1 theta_3(T1 a, T1 b)
); // theta_3_string

const char theta_3_name[] = "theta_3";

void special_theta_3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_3_cuda_kernel", [&]() {
    jitted_gpu_kernel<theta_3_name, scalar_t, scalar_t, 2>(iterator, theta_3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_theta_3_cuda_kernel(TensorIteratorBase &iterator)

const auto theta_4_string = jiterator_stringify(
  template<typename T1>
  T1 theta_4(T1 a, T1 b) {
    return a;
  } // T1 theta_4(T1 a, T1 b)
); // theta_4_string

const char theta_4_name[] = "theta_4";

void special_theta_4_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_4_cuda_kernel", [&]() {
    jitted_gpu_kernel<theta_4_name, scalar_t, scalar_t, 2>(iterator, theta_4_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_4_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_theta_4_cuda_kernel(TensorIteratorBase &iterator)

const auto upper_incomplete_gamma_string = jiterator_stringify(
  template<typename T1>
  T1 upper_incomplete_gamma(T1 a, T1 b) {
    return a;
  } // T1 upper_incomplete_gamma(T1 a, T1 b)
); // upper_incomplete_gamma_string

const char upper_incomplete_gamma_name[] = "upper_incomplete_gamma";

void special_upper_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "upper_incomplete_gamma_cuda_kernel", [&]() {
    jitted_gpu_kernel<upper_incomplete_gamma_name, scalar_t, scalar_t, 2>(iterator, upper_incomplete_gamma_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "upper_incomplete_gamma_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_upper_incomplete_gamma_cuda_kernel(TensorIteratorBase &iterator)

const auto associated_laguerre_polynomial_l_string = jiterator_stringify(
  template<typename T1>
  T1 associated_laguerre_polynomial_l(T1 a, T1 b, T1 c) {
    return a;
  } // T1 associated_laguerre_polynomial_l(T1 a, T1 b, T1 c)
); // associated_laguerre_polynomial_l_string

const char associated_laguerre_polynomial_l_name[] = "associated_laguerre_polynomial_l";

void special_associated_laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_laguerre_polynomial_l_cuda_kernel", [&]() {
    jitted_gpu_kernel<associated_laguerre_polynomial_l_name, scalar_t, scalar_t, 3>(iterator, associated_laguerre_polynomial_l_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_laguerre_polynomial_l_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_associated_laguerre_polynomial_l_cuda_kernel(TensorIteratorBase &iterator)

const auto associated_legendre_p_string = jiterator_stringify(
  template<typename T1>
  T1 associated_legendre_p(T1 a, T1 b, T1 c) {
    return a;
  } // T1 associated_legendre_p(T1 a, T1 b, T1 c)
); // associated_legendre_p_string

const char associated_legendre_p_name[] = "associated_legendre_p";

void special_associated_legendre_p_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_p_cuda_kernel", [&]() {
    jitted_gpu_kernel<associated_legendre_p_name, scalar_t, scalar_t, 3>(iterator, associated_legendre_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_p_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_associated_legendre_p_cuda_kernel(TensorIteratorBase &iterator)

const auto associated_legendre_q_string = jiterator_stringify(
  template<typename T1>
  T1 associated_legendre_q(T1 a, T1 b, T1 c) {
    return a;
  } // T1 associated_legendre_q(T1 a, T1 b, T1 c)
); // associated_legendre_q_string

const char associated_legendre_q_name[] = "associated_legendre_q";

void special_associated_legendre_q_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_q_cuda_kernel", [&]() {
    jitted_gpu_kernel<associated_legendre_q_name, scalar_t, scalar_t, 3>(iterator, associated_legendre_q_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_q_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_associated_legendre_q_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_el3_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el3(T1 a, T1 b, T1 c) {
    return a;
  } // T1 bulirsch_elliptic_integral_el3(T1 a, T1 b, T1 c)
); // bulirsch_elliptic_integral_el3_string

const char bulirsch_elliptic_integral_el3_name[] = "bulirsch_elliptic_integral_el3";

void special_bulirsch_elliptic_integral_el3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el3_cuda_kernel", [&]() {
    jitted_gpu_kernel<bulirsch_elliptic_integral_el3_name, scalar_t, scalar_t, 3>(iterator, bulirsch_elliptic_integral_el3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bulirsch_elliptic_integral_el3_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_d_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_d(T1 a, T1 b, T1 c) {
    return a;
  } // T1 carlson_elliptic_r_d(T1 a, T1 b, T1 c)
); // carlson_elliptic_r_d_string

const char carlson_elliptic_r_d_name[] = "carlson_elliptic_r_d";

void special_carlson_elliptic_r_d_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_d_cuda_kernel", [&]() {
    jitted_gpu_kernel<carlson_elliptic_r_d_name, scalar_t, scalar_t, 3>(iterator, carlson_elliptic_r_d_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_d_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_carlson_elliptic_r_d_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_f_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_f(T1 a, T1 b, T1 c) {
    return a;
  } // T1 carlson_elliptic_r_f(T1 a, T1 b, T1 c)
); // carlson_elliptic_r_f_string

const char carlson_elliptic_r_f_name[] = "carlson_elliptic_r_f";

void special_carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_f_cuda_kernel", [&]() {
    jitted_gpu_kernel<carlson_elliptic_r_f_name, scalar_t, scalar_t, 3>(iterator, carlson_elliptic_r_f_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_f_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_carlson_elliptic_r_f_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_g_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_g(T1 a, T1 b, T1 c) {
    return a;
  } // T1 carlson_elliptic_r_g(T1 a, T1 b, T1 c)
); // carlson_elliptic_r_g_string

const char carlson_elliptic_r_g_name[] = "carlson_elliptic_r_g";

void special_carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_g_cuda_kernel", [&]() {
    jitted_gpu_kernel<carlson_elliptic_r_g_name, scalar_t, scalar_t, 3>(iterator, carlson_elliptic_r_g_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_g_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_carlson_elliptic_r_g_cuda_kernel(TensorIteratorBase &iterator)

const auto gegenbauer_polynomial_c_string = jiterator_stringify(
  template<typename T1>
  T1 gegenbauer_polynomial_c(T1 a, T1 b, T1 c) {
    return a;
  } // T1 gegenbauer_polynomial_c(T1 a, T1 b, T1 c)
); // gegenbauer_polynomial_c_string

const char gegenbauer_polynomial_c_name[] = "gegenbauer_polynomial_c";

void special_gegenbauer_polynomial_c_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gegenbauer_polynomial_c_cuda_kernel", [&]() {
    jitted_gpu_kernel<gegenbauer_polynomial_c_name, scalar_t, scalar_t, 3>(iterator, gegenbauer_polynomial_c_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gegenbauer_polynomial_c_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_gegenbauer_polynomial_c_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_beta_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_beta(T1 a, T1 b, T1 c) {
    return a;
  } // T1 incomplete_beta(T1 a, T1 b, T1 c)
); // incomplete_beta_string

const char incomplete_beta_name[] = "incomplete_beta";

void special_incomplete_beta_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_beta_cuda_kernel", [&]() {
    jitted_gpu_kernel<incomplete_beta_name, scalar_t, scalar_t, 3>(iterator, incomplete_beta_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_beta_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_incomplete_beta_cuda_kernel(TensorIteratorBase &iterator)

const auto incomplete_elliptic_integral_pi_string = jiterator_stringify(
  template<typename T1>
  T1 incomplete_elliptic_integral_pi(T1 a, T1 b, T1 c) {
    return a;
  } // T1 incomplete_elliptic_integral_pi(T1 a, T1 b, T1 c)
); // incomplete_elliptic_integral_pi_string

const char incomplete_elliptic_integral_pi_name[] = "incomplete_elliptic_integral_pi";

void special_incomplete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_pi_cuda_kernel", [&]() {
    jitted_gpu_kernel<incomplete_elliptic_integral_pi_name, scalar_t, scalar_t, 3>(iterator, incomplete_elliptic_integral_pi_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_pi_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_incomplete_elliptic_integral_pi_cuda_kernel(TensorIteratorBase &iterator)

const auto kummer_confluent_hypergeometric_1_f_1_string = jiterator_stringify(
  template<typename T1>
  T1 kummer_confluent_hypergeometric_1_f_1(T1 a, T1 b, T1 c) {
    return a;
  } // T1 kummer_confluent_hypergeometric_1_f_1(T1 a, T1 b, T1 c)
); // kummer_confluent_hypergeometric_1_f_1_string

const char kummer_confluent_hypergeometric_1_f_1_name[] = "kummer_confluent_hypergeometric_1_f_1";

void special_kummer_confluent_hypergeometric_1_f_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "kummer_confluent_hypergeometric_1_f_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<kummer_confluent_hypergeometric_1_f_1_name, scalar_t, scalar_t, 3>(iterator, kummer_confluent_hypergeometric_1_f_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "kummer_confluent_hypergeometric_1_f_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_kummer_confluent_hypergeometric_1_f_1_cuda_kernel(TensorIteratorBase &iterator)

const auto radial_polynomial_r_string = jiterator_stringify(
  template<typename T1>
  T1 radial_polynomial_r(T1 a, T1 b, T1 c) {
    return a;
  } // T1 radial_polynomial_r(T1 a, T1 b, T1 c)
); // radial_polynomial_r_string

const char radial_polynomial_r_name[] = "radial_polynomial_r";

void special_radial_polynomial_r_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "radial_polynomial_r_cuda_kernel", [&]() {
    jitted_gpu_kernel<radial_polynomial_r_name, scalar_t, scalar_t, 3>(iterator, radial_polynomial_r_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "radial_polynomial_r_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_radial_polynomial_r_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_legendre_y_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_legendre_y(T1 a, T1 b, T1 c) {
    return a;
  } // T1 spherical_legendre_y(T1 a, T1 b, T1 c)
); // spherical_legendre_y_string

const char spherical_legendre_y_name[] = "spherical_legendre_y";

void special_spherical_legendre_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_legendre_y_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_legendre_y_name, scalar_t, scalar_t, 3>(iterator, spherical_legendre_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_legendre_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_legendre_y_cuda_kernel(TensorIteratorBase &iterator)

const auto tricomi_confluent_hypergeometric_u_string = jiterator_stringify(
  template<typename T1>
  T1 tricomi_confluent_hypergeometric_u(T1 a, T1 b, T1 c) {
    return a;
  } // T1 tricomi_confluent_hypergeometric_u(T1 a, T1 b, T1 c)
); // tricomi_confluent_hypergeometric_u_string

const char tricomi_confluent_hypergeometric_u_name[] = "tricomi_confluent_hypergeometric_u";

void special_tricomi_confluent_hypergeometric_u_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tricomi_confluent_hypergeometric_u_cuda_kernel", [&]() {
    jitted_gpu_kernel<tricomi_confluent_hypergeometric_u_name, scalar_t, scalar_t, 3>(iterator, tricomi_confluent_hypergeometric_u_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tricomi_confluent_hypergeometric_u_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_tricomi_confluent_hypergeometric_u_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_cel_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_cel(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 bulirsch_elliptic_integral_cel(T1 a, T1 b, T1 c, T1 d)
); // bulirsch_elliptic_integral_cel_string

const char bulirsch_elliptic_integral_cel_name[] = "bulirsch_elliptic_integral_cel";

void special_bulirsch_elliptic_integral_cel_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_cel_cuda_kernel", [&]() {
    jitted_gpu_kernel<bulirsch_elliptic_integral_cel_name, scalar_t, scalar_t, 4>(iterator, bulirsch_elliptic_integral_cel_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_cel_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bulirsch_elliptic_integral_cel_cuda_kernel(TensorIteratorBase &iterator)

const auto bulirsch_elliptic_integral_el2_string = jiterator_stringify(
  template<typename T1>
  T1 bulirsch_elliptic_integral_el2(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 bulirsch_elliptic_integral_el2(T1 a, T1 b, T1 c, T1 d)
); // bulirsch_elliptic_integral_el2_string

const char bulirsch_elliptic_integral_el2_name[] = "bulirsch_elliptic_integral_el2";

void special_bulirsch_elliptic_integral_el2_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el2_cuda_kernel", [&]() {
    jitted_gpu_kernel<bulirsch_elliptic_integral_el2_name, scalar_t, scalar_t, 4>(iterator, bulirsch_elliptic_integral_el2_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el2_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_bulirsch_elliptic_integral_el2_cuda_kernel(TensorIteratorBase &iterator)

const auto carlson_elliptic_r_j_string = jiterator_stringify(
  template<typename T1>
  T1 carlson_elliptic_r_j(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 carlson_elliptic_r_j(T1 a, T1 b, T1 c, T1 d)
); // carlson_elliptic_r_j_string

const char carlson_elliptic_r_j_name[] = "carlson_elliptic_r_j";

void special_carlson_elliptic_r_j_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_j_cuda_kernel", [&]() {
    jitted_gpu_kernel<carlson_elliptic_r_j_name, scalar_t, scalar_t, 4>(iterator, carlson_elliptic_r_j_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_j_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_carlson_elliptic_r_j_cuda_kernel(TensorIteratorBase &iterator)

const auto gauss_hypergeometric_2_f_1_string = jiterator_stringify(
  template<typename T1>
  T1 gauss_hypergeometric_2_f_1(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 gauss_hypergeometric_2_f_1(T1 a, T1 b, T1 c, T1 d)
); // gauss_hypergeometric_2_f_1_string

const char gauss_hypergeometric_2_f_1_name[] = "gauss_hypergeometric_2_f_1";

void special_gauss_hypergeometric_2_f_1_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gauss_hypergeometric_2_f_1_cuda_kernel", [&]() {
    jitted_gpu_kernel<gauss_hypergeometric_2_f_1_name, scalar_t, scalar_t, 4>(iterator, gauss_hypergeometric_2_f_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gauss_hypergeometric_2_f_1_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_gauss_hypergeometric_2_f_1_cuda_kernel(TensorIteratorBase &iterator)

const auto jacobi_polynomial_p_string = jiterator_stringify(
  template<typename T1>
  T1 jacobi_polynomial_p(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 jacobi_polynomial_p(T1 a, T1 b, T1 c, T1 d)
); // jacobi_polynomial_p_string

const char jacobi_polynomial_p_name[] = "jacobi_polynomial_p";

void special_jacobi_polynomial_p_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_polynomial_p_cuda_kernel", [&]() {
    jitted_gpu_kernel<jacobi_polynomial_p_name, scalar_t, scalar_t, 4>(iterator, jacobi_polynomial_p_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_polynomial_p_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_jacobi_polynomial_p_cuda_kernel(TensorIteratorBase &iterator)

const auto spherical_harmonic_y_string = jiterator_stringify(
  template<typename T1>
  T1 spherical_harmonic_y(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 spherical_harmonic_y(T1 a, T1 b, T1 c, T1 d)
); // spherical_harmonic_y_string

const char spherical_harmonic_y_name[] = "spherical_harmonic_y";

void special_spherical_harmonic_y_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_harmonic_y_cuda_kernel", [&]() {
    jitted_gpu_kernel<spherical_harmonic_y_name, scalar_t, scalar_t, 4>(iterator, spherical_harmonic_y_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_harmonic_y_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_spherical_harmonic_y_cuda_kernel(TensorIteratorBase &iterator)

const auto zernike_polynomial_z_string = jiterator_stringify(
  template<typename T1>
  T1 zernike_polynomial_z(T1 a, T1 b, T1 c, T1 d) {
    return a;
  } // T1 zernike_polynomial_z(T1 a, T1 b, T1 c, T1 d)
); // zernike_polynomial_z_string

const char zernike_polynomial_z_name[] = "zernike_polynomial_z";

void special_zernike_polynomial_z_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "zernike_polynomial_z_cuda_kernel", [&]() {
    jitted_gpu_kernel<zernike_polynomial_z_name, scalar_t, scalar_t, 4>(iterator, zernike_polynomial_z_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "zernike_polynomial_z_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
#endif
} // void special_zernike_polynomial_z_cuda_kernel(TensorIteratorBase &iterator)
} // namespace (anonymous)

REGISTER_DISPATCH(special_airy_ai_stub, &special_airy_ai_cuda_kernel);
REGISTER_DISPATCH(special_airy_bi_stub, &special_airy_bi_cuda_kernel);
REGISTER_DISPATCH(special_associated_laguerre_polynomial_l_stub, &special_associated_laguerre_polynomial_l_cuda_kernel);
REGISTER_DISPATCH(special_associated_legendre_p_stub, &special_associated_legendre_p_cuda_kernel);
REGISTER_DISPATCH(special_associated_legendre_q_stub, &special_associated_legendre_q_cuda_kernel);
REGISTER_DISPATCH(special_bell_polynomial_b_stub, &special_bell_polynomial_b_cuda_kernel);
REGISTER_DISPATCH(special_bernoulli_number_stub, &special_bernoulli_number_cuda_kernel);
REGISTER_DISPATCH(special_bernoulli_polynomial_b_stub, &special_bernoulli_polynomial_b_cuda_kernel);
REGISTER_DISPATCH(special_bessel_j_0_stub, &special_bessel_j_0_cuda_kernel);
REGISTER_DISPATCH(special_bessel_j_1_stub, &special_bessel_j_1_cuda_kernel);
REGISTER_DISPATCH(special_bessel_j_stub, &special_bessel_j_cuda_kernel);
REGISTER_DISPATCH(special_bessel_y_0_stub, &special_bessel_y_0_cuda_kernel);
REGISTER_DISPATCH(special_bessel_y_1_stub, &special_bessel_y_1_cuda_kernel);
REGISTER_DISPATCH(special_bessel_y_stub, &special_bessel_y_cuda_kernel);
REGISTER_DISPATCH(special_beta_stub, &special_beta_cuda_kernel);
REGISTER_DISPATCH(special_binomial_coefficient_stub, &special_binomial_coefficient_cuda_kernel);
REGISTER_DISPATCH(special_bose_einstein_integral_g_stub, &special_bose_einstein_integral_g_cuda_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_cel_stub, &special_bulirsch_elliptic_integral_cel_cuda_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_el1_stub, &special_bulirsch_elliptic_integral_el1_cuda_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_el2_stub, &special_bulirsch_elliptic_integral_el2_cuda_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_el3_stub, &special_bulirsch_elliptic_integral_el3_cuda_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_c_stub, &special_carlson_elliptic_r_c_cuda_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_d_stub, &special_carlson_elliptic_r_d_cuda_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_f_stub, &special_carlson_elliptic_r_f_cuda_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_g_stub, &special_carlson_elliptic_r_g_cuda_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_j_stub, &special_carlson_elliptic_r_j_cuda_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_t_stub, &special_chebyshev_polynomial_t_cuda_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_u_stub, &special_chebyshev_polynomial_u_cuda_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_v_stub, &special_chebyshev_polynomial_v_cuda_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_w_stub, &special_chebyshev_polynomial_w_cuda_kernel);
REGISTER_DISPATCH(special_clausen_cl_stub, &special_clausen_cl_cuda_kernel);
REGISTER_DISPATCH(special_clausen_sl_stub, &special_clausen_sl_cuda_kernel);
REGISTER_DISPATCH(special_complete_carlson_elliptic_r_f_stub, &special_complete_carlson_elliptic_r_f_cuda_kernel);
REGISTER_DISPATCH(special_complete_carlson_elliptic_r_g_stub, &special_complete_carlson_elliptic_r_g_cuda_kernel);
REGISTER_DISPATCH(special_complete_elliptic_integral_e_stub, &special_complete_elliptic_integral_e_cuda_kernel);
REGISTER_DISPATCH(special_complete_elliptic_integral_k_stub, &special_complete_elliptic_integral_k_cuda_kernel);
REGISTER_DISPATCH(special_complete_elliptic_integral_pi_stub, &special_complete_elliptic_integral_pi_cuda_kernel);
REGISTER_DISPATCH(special_complete_legendre_elliptic_integral_d_stub, &special_complete_legendre_elliptic_integral_d_cuda_kernel);
REGISTER_DISPATCH(special_confluent_hypergeometric_0_f_1_stub, &special_confluent_hypergeometric_0_f_1_cuda_kernel);
REGISTER_DISPATCH(special_cos_pi_stub, &special_cos_pi_cuda_kernel);
REGISTER_DISPATCH(special_cosh_pi_stub, &special_cosh_pi_cuda_kernel);
REGISTER_DISPATCH(special_cosine_integral_ci_stub, &special_cosine_integral_ci_cuda_kernel);
REGISTER_DISPATCH(special_debye_d_stub, &special_debye_d_cuda_kernel);
REGISTER_DISPATCH(special_dilogarithm_li_2_stub, &special_dilogarithm_li_2_cuda_kernel);
REGISTER_DISPATCH(special_dirichlet_beta_stub, &special_dirichlet_beta_cuda_kernel);
REGISTER_DISPATCH(special_dirichlet_eta_stub, &special_dirichlet_eta_cuda_kernel);
REGISTER_DISPATCH(special_dirichlet_lambda_stub, &special_dirichlet_lambda_cuda_kernel);
REGISTER_DISPATCH(special_double_factorial_stub, &special_double_factorial_cuda_kernel);
REGISTER_DISPATCH(special_exp_airy_ai_stub, &special_exp_airy_ai_cuda_kernel);
REGISTER_DISPATCH(special_exp_airy_bi_stub, &special_exp_airy_bi_cuda_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_i_stub, &special_exp_modified_bessel_i_cuda_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_k_0_stub, &special_exp_modified_bessel_k_0_cuda_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_k_1_stub, &special_exp_modified_bessel_k_1_cuda_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_k_stub, &special_exp_modified_bessel_k_cuda_kernel);
REGISTER_DISPATCH(special_exponential_integral_e_stub, &special_exponential_integral_e_cuda_kernel);
REGISTER_DISPATCH(special_exponential_integral_ei_stub, &special_exponential_integral_ei_cuda_kernel);
REGISTER_DISPATCH(special_factorial_stub, &special_factorial_cuda_kernel);
REGISTER_DISPATCH(special_falling_factorial_stub, &special_falling_factorial_cuda_kernel);
REGISTER_DISPATCH(special_fermi_dirac_integral_f_stub, &special_fermi_dirac_integral_f_cuda_kernel);
REGISTER_DISPATCH(special_fresnel_integral_c_stub, &special_fresnel_integral_c_cuda_kernel);
REGISTER_DISPATCH(special_fresnel_integral_s_stub, &special_fresnel_integral_s_cuda_kernel);
REGISTER_DISPATCH(special_gauss_hypergeometric_2_f_1_stub, &special_gauss_hypergeometric_2_f_1_cuda_kernel);
REGISTER_DISPATCH(special_gegenbauer_polynomial_c_stub, &special_gegenbauer_polynomial_c_cuda_kernel);
REGISTER_DISPATCH(special_hankel_h_1_stub, &special_hankel_h_1_cuda_kernel);
REGISTER_DISPATCH(special_hankel_h_2_stub, &special_hankel_h_2_cuda_kernel);
REGISTER_DISPATCH(special_harmonic_number_stub, &special_harmonic_number_cuda_kernel);
REGISTER_DISPATCH(special_hermite_polynomial_h_stub, &special_hermite_polynomial_h_cuda_kernel);
REGISTER_DISPATCH(special_hermite_polynomial_he_stub, &special_hermite_polynomial_he_cuda_kernel);
REGISTER_DISPATCH(special_heuman_lambda_stub, &special_heuman_lambda_cuda_kernel);
REGISTER_DISPATCH(special_hurwitz_zeta_stub, &special_hurwitz_zeta_cuda_kernel);
REGISTER_DISPATCH(special_hyperbolic_cosine_integral_chi_stub, &special_hyperbolic_cosine_integral_chi_cuda_kernel);
REGISTER_DISPATCH(special_hyperbolic_sine_integral_shi_stub, &special_hyperbolic_sine_integral_shi_cuda_kernel);
REGISTER_DISPATCH(special_incomplete_beta_stub, &special_incomplete_beta_cuda_kernel);
REGISTER_DISPATCH(special_incomplete_elliptic_integral_e_stub, &special_incomplete_elliptic_integral_e_cuda_kernel);
REGISTER_DISPATCH(special_incomplete_elliptic_integral_f_stub, &special_incomplete_elliptic_integral_f_cuda_kernel);
REGISTER_DISPATCH(special_incomplete_elliptic_integral_pi_stub, &special_incomplete_elliptic_integral_pi_cuda_kernel);
REGISTER_DISPATCH(special_incomplete_legendre_elliptic_integral_d_stub, &special_incomplete_legendre_elliptic_integral_d_cuda_kernel);
REGISTER_DISPATCH(special_jacobi_polynomial_p_stub, &special_jacobi_polynomial_p_cuda_kernel);
REGISTER_DISPATCH(special_jacobi_theta_1_stub, &special_jacobi_theta_1_cuda_kernel);
REGISTER_DISPATCH(special_jacobi_theta_2_stub, &special_jacobi_theta_2_cuda_kernel);
REGISTER_DISPATCH(special_jacobi_theta_3_stub, &special_jacobi_theta_3_cuda_kernel);
REGISTER_DISPATCH(special_jacobi_theta_4_stub, &special_jacobi_theta_4_cuda_kernel);
REGISTER_DISPATCH(special_jacobi_zeta_stub, &special_jacobi_zeta_cuda_kernel);
REGISTER_DISPATCH(special_kummer_confluent_hypergeometric_1_f_1_stub, &special_kummer_confluent_hypergeometric_1_f_1_cuda_kernel);
REGISTER_DISPATCH(special_laguerre_polynomial_l_stub, &special_laguerre_polynomial_l_cuda_kernel);
REGISTER_DISPATCH(special_lah_number_stub, &special_lah_number_cuda_kernel);
REGISTER_DISPATCH(special_legendre_polynomial_p_stub, &special_legendre_polynomial_p_cuda_kernel);
REGISTER_DISPATCH(special_legendre_q_stub, &special_legendre_q_cuda_kernel);
REGISTER_DISPATCH(special_ln_binomial_coefficient_stub, &special_ln_binomial_coefficient_cuda_kernel);
REGISTER_DISPATCH(special_ln_double_factorial_stub, &special_ln_double_factorial_cuda_kernel);
REGISTER_DISPATCH(special_ln_factorial_stub, &special_ln_factorial_cuda_kernel);
REGISTER_DISPATCH(special_ln_falling_factorial_stub, &special_ln_falling_factorial_cuda_kernel);
REGISTER_DISPATCH(special_ln_gamma_sign_stub, &special_ln_gamma_sign_cuda_kernel);
REGISTER_DISPATCH(special_ln_gamma_stub, &special_ln_gamma_cuda_kernel);
REGISTER_DISPATCH(special_ln_rising_factorial_stub, &special_ln_rising_factorial_cuda_kernel);
REGISTER_DISPATCH(special_logarithmic_integral_li_stub, &special_logarithmic_integral_li_cuda_kernel);
REGISTER_DISPATCH(special_lower_incomplete_gamma_stub, &special_lower_incomplete_gamma_cuda_kernel);
REGISTER_DISPATCH(special_modified_bessel_i_0_stub, &special_modified_bessel_i_0_cuda_kernel);
REGISTER_DISPATCH(special_modified_bessel_i_1_stub, &special_modified_bessel_i_1_cuda_kernel);
REGISTER_DISPATCH(special_modified_bessel_i_stub, &special_modified_bessel_i_cuda_kernel);
REGISTER_DISPATCH(special_modified_bessel_k_0_stub, &special_modified_bessel_k_0_cuda_kernel);
REGISTER_DISPATCH(special_modified_bessel_k_1_stub, &special_modified_bessel_k_1_cuda_kernel);
REGISTER_DISPATCH(special_modified_bessel_k_stub, &special_modified_bessel_k_cuda_kernel);
REGISTER_DISPATCH(special_neville_theta_c_stub, &special_neville_theta_c_cuda_kernel);
REGISTER_DISPATCH(special_neville_theta_d_stub, &special_neville_theta_d_cuda_kernel);
REGISTER_DISPATCH(special_neville_theta_n_stub, &special_neville_theta_n_cuda_kernel);
REGISTER_DISPATCH(special_neville_theta_s_stub, &special_neville_theta_s_cuda_kernel);
REGISTER_DISPATCH(special_nome_q_stub, &special_nome_q_cuda_kernel);
REGISTER_DISPATCH(special_owens_t_stub, &special_owens_t_cuda_kernel);
REGISTER_DISPATCH(special_polar_pi_stub, &special_polar_pi_cuda_kernel);
REGISTER_DISPATCH(special_polylogarithm_li_stub, &special_polylogarithm_li_cuda_kernel);
REGISTER_DISPATCH(special_prime_number_stub, &special_prime_number_cuda_kernel);
REGISTER_DISPATCH(special_radial_polynomial_r_stub, &special_radial_polynomial_r_cuda_kernel);
REGISTER_DISPATCH(special_reciprocal_gamma_stub, &special_reciprocal_gamma_cuda_kernel);
REGISTER_DISPATCH(special_riemann_zeta_stub, &special_riemann_zeta_cuda_kernel);
REGISTER_DISPATCH(special_rising_factorial_stub, &special_rising_factorial_cuda_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_t_stub, &special_shifted_chebyshev_polynomial_t_cuda_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_u_stub, &special_shifted_chebyshev_polynomial_u_cuda_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_v_stub, &special_shifted_chebyshev_polynomial_v_cuda_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_w_stub, &special_shifted_chebyshev_polynomial_w_cuda_kernel);
REGISTER_DISPATCH(special_sin_pi_stub, &special_sin_pi_cuda_kernel);
REGISTER_DISPATCH(special_sinc_pi_stub, &special_sinc_pi_cuda_kernel);
REGISTER_DISPATCH(special_sinh_pi_stub, &special_sinh_pi_cuda_kernel);
REGISTER_DISPATCH(special_sinhc_pi_stub, &special_sinhc_pi_cuda_kernel);
REGISTER_DISPATCH(special_sinhc_stub, &special_sinhc_cuda_kernel);
REGISTER_DISPATCH(special_spherical_bessel_j_0_stub, &special_spherical_bessel_j_0_cuda_kernel);
REGISTER_DISPATCH(special_spherical_bessel_j_stub, &special_spherical_bessel_j_cuda_kernel);
REGISTER_DISPATCH(special_spherical_bessel_y_stub, &special_spherical_bessel_y_cuda_kernel);
REGISTER_DISPATCH(special_spherical_hankel_h_1_stub, &special_spherical_hankel_h_1_cuda_kernel);
REGISTER_DISPATCH(special_spherical_hankel_h_2_stub, &special_spherical_hankel_h_2_cuda_kernel);
REGISTER_DISPATCH(special_spherical_harmonic_y_stub, &special_spherical_harmonic_y_cuda_kernel);
REGISTER_DISPATCH(special_spherical_legendre_y_stub, &special_spherical_legendre_y_cuda_kernel);
REGISTER_DISPATCH(special_spherical_modified_bessel_i_stub, &special_spherical_modified_bessel_i_cuda_kernel);
REGISTER_DISPATCH(special_spherical_modified_bessel_k_stub, &special_spherical_modified_bessel_k_cuda_kernel);
REGISTER_DISPATCH(special_stirling_number_1_stub, &special_stirling_number_1_cuda_kernel);
REGISTER_DISPATCH(special_stirling_number_2_stub, &special_stirling_number_2_cuda_kernel);
REGISTER_DISPATCH(special_tan_pi_stub, &special_tan_pi_cuda_kernel);
REGISTER_DISPATCH(special_tanh_pi_stub, &special_tanh_pi_cuda_kernel);
REGISTER_DISPATCH(special_theta_1_stub, &special_theta_1_cuda_kernel);
REGISTER_DISPATCH(special_theta_2_stub, &special_theta_2_cuda_kernel);
REGISTER_DISPATCH(special_theta_3_stub, &special_theta_3_cuda_kernel);
REGISTER_DISPATCH(special_theta_4_stub, &special_theta_4_cuda_kernel);
REGISTER_DISPATCH(special_tricomi_confluent_hypergeometric_u_stub, &special_tricomi_confluent_hypergeometric_u_cuda_kernel);
REGISTER_DISPATCH(special_upper_incomplete_gamma_stub, &special_upper_incomplete_gamma_cuda_kernel);
REGISTER_DISPATCH(special_zernike_polynomial_z_stub, &special_zernike_polynomial_z_cuda_kernel);
} // namespace native
} // namespace at
