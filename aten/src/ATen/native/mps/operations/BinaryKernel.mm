#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/complex_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/nextafter_native.h>
#include <ATen/ops/polar_native.h>
#include <ATen/ops/view_as_real.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/BinaryKernel_metallib.h>
#endif

namespace mps {

void binary_op_kernel(const std::string func_name,
                      const Tensor& input,
                      const Tensor& other,
                      const Tensor& output,
                      const std::optional<Scalar> alpha) {
  auto new_size = at::infer_size(input.sizes(), other.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }

  auto iter = TensorIteratorConfig()
                  .allow_cpu_scalars(true)
                  .add_output(output)
                  .add_input(input)
                  .add_input(other)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(true)
                  .build();

  lib.exec_binary_kernel(iter, func_name, alpha);
}

} // namespace mps

static void fmax_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    lib.exec_binary_kernel(iter, "fmax");
  } else {
    at::maximum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}

static void fmin_mps_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    lib.exec_binary_kernel(iter, "fmin");
  } else {
    at::minimum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
  }
}

static void copysign_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "copysign");
}

static void nextafter_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "nextafter_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "nextafter");
}

static void zeta_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "zeta_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "zeta");
}

static void logaddexp_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "logaddexp");
}

static void logaddexp2_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "logaddexp2");
}

static void xlog1py_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()), "xlog1py_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "xlog1py");
}

static void chebyshev_polynomial_t_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_t_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_t");
}

static void chebyshev_polynomial_u_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_u_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_u");
}

static void chebyshev_polynomial_v_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_v_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_v");
}

static void chebyshev_polynomial_w_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "chebyshev_polynomial_w_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "chebyshev_polynomial_w");
}

static void shifted_chebyshev_polynomial_t_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "shifted_chebyshev_polynomial_t_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "shifted_chebyshev_polynomial_t");
}

static void shifted_chebyshev_polynomial_u_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "shifted_chebyshev_polynomial_u_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "shifted_chebyshev_polynomial_u");
}

static void shifted_chebyshev_polynomial_v_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "shifted_chebyshev_polynomial_v_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "shifted_chebyshev_polynomial_v");
}

static void shifted_chebyshev_polynomial_w_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "shifted_chebyshev_polynomial_w_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "shifted_chebyshev_polynomial_w");
}

static void hermite_polynomial_h_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "hermite_polynomial_h_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "hermite_polynomial_h");
}

static void hermite_polynomial_he_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "hermite_polynomial_he_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "hermite_polynomial_he");
}

static void polar_mps_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "polar");
}

static void complex_mps_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "make_complex");
}

static void lerp_scalar_mps_kernel(at::TensorIteratorBase& iter, const Scalar& weight) {
  lib.exec_binary_kernel(iter, "lerp_alpha", weight);
}

static void native_dropout_mask_and_scale_mps_kernel(at::TensorIteratorBase& iter, const Scalar& scale) {
  lib.exec_binary_kernel(iter, "native_dropout_mask_and_scale", scale);
}

static void mul_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "mul");
}

static void div_true_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "div_true");
}

static void div_floor_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "div_floor");
}

static void div_trunc_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "div_trunc");
}

static void remainder_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "remainder");
}

static void fmod_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "fmod");
}

static void igamma_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "igamma");
}

static void igammac_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "igammac");
}

static void hypot_mps_kernel(TensorIteratorBase& iter) {
  lib.exec_binary_kernel(iter, "hypot");
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel)
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel)
REGISTER_DISPATCH(copysign_stub, &copysign_mps_kernel)
REGISTER_DISPATCH(nextafter_stub, &nextafter_mps_kernel)
REGISTER_DISPATCH(zeta_stub, &zeta_mps_kernel)
REGISTER_DISPATCH(logaddexp_stub, &logaddexp_mps_kernel);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_mps_kernel);
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_t_stub, &chebyshev_polynomial_t_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_u_stub, &chebyshev_polynomial_u_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_v_stub, &chebyshev_polynomial_v_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_mps_kernel)
REGISTER_DISPATCH(shifted_chebyshev_polynomial_t_stub, &shifted_chebyshev_polynomial_t_mps_kernel)
REGISTER_DISPATCH(shifted_chebyshev_polynomial_u_stub, &shifted_chebyshev_polynomial_u_mps_kernel)
REGISTER_DISPATCH(shifted_chebyshev_polynomial_v_stub, &shifted_chebyshev_polynomial_v_mps_kernel)
REGISTER_DISPATCH(shifted_chebyshev_polynomial_w_stub, &shifted_chebyshev_polynomial_w_mps_kernel)
REGISTER_DISPATCH(hermite_polynomial_h_stub, &hermite_polynomial_h_mps_kernel)
REGISTER_DISPATCH(hermite_polynomial_he_stub, &hermite_polynomial_he_mps_kernel)
REGISTER_DISPATCH(polar_stub, &polar_mps_kernel);
REGISTER_DISPATCH(complex_stub, &complex_mps_kernel);
REGISTER_DISPATCH(lerp_kernel_scalar_weight, &lerp_scalar_mps_kernel)
REGISTER_DISPATCH(mul_stub, &mul_mps_kernel)
REGISTER_DISPATCH(div_true_stub, &div_true_mps_kernel)
REGISTER_DISPATCH(div_floor_stub, &div_floor_mps_kernel)
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_mps_kernel)
REGISTER_DISPATCH(fmod_stub, &fmod_mps_kernel)
REGISTER_DISPATCH(remainder_stub, &remainder_mps_kernel)
REGISTER_DISPATCH(igamma_stub, &igamma_mps_kernel)
REGISTER_DISPATCH(igammac_stub, &igammac_mps_kernel)
REGISTER_DISPATCH(hypot_stub, &hypot_mps_kernel)
} // namespace at::native
