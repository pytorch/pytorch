#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIndexing.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/operations/BinaryKernel.h>
// For MTLLanguageVersion_3_1
#include <ATen/native/mps/MPSGraphSonomaOps.h>
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

void complex_mul_out(const Tensor& input, const Tensor& other, const Tensor& output) {
  TORCH_INTERNAL_ASSERT(c10::isComplexType(input.scalar_type()) || c10::isComplexType(other.scalar_type()));
  auto new_size = at::infer_size(input.sizes(), other.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }
  auto common_dtype = output.scalar_type();
  auto input_cast = input.to(kMPS, common_dtype);
  auto other_cast = other.to(kMPS, common_dtype);
  auto iter = TensorIteratorConfig().add_output(output).add_input(input_cast).add_input(other_cast).build();

  lib.exec_binary_kernel(iter, "complex_mul");
}

// When alpha = 1, binary alpha ops can call into a binary kernel for reduced overhead, see BinaryOps.mm
void binary_op_kernel(const std::string func_name, const Tensor& input, const Tensor& other, const Tensor& output) {

  auto new_size = at::infer_size(input.sizes(), other.sizes());
  if (!output.sizes().equals(new_size)) {
    output.resize_(new_size);
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }
  auto common_dtype = output.scalar_type();
  auto input_cast = input.to(kMPS, common_dtype);
  auto other_cast = other.to(kMPS, common_dtype);
  if (input.scalar_type() != common_dtype || other.scalar_type() != common_dtype) {
    input_cast = input.to(kMPS, common_dtype);

  }
  auto iter =
      TensorIteratorConfig().add_output(output).add_input(input_cast).add_input(other_cast).build();

  lib.exec_binary_kernel(iter, func_name);
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

static void hermite_polynomial_h_mps_kernel(TensorIteratorBase& iter) {
  TORCH_CHECK_TYPE(isFloatingType(iter.common_dtype()),
                   "hermite_polynomial_h_mps not implemented for non-floating types");
  lib.exec_binary_kernel(iter, "hermite_polynomial_h");
}

static void polar_mps_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "polar");
}

static void complex_mps_kernel(TensorIterator& iter) {
  lib.exec_binary_kernel(iter, "make_complex");
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel)
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel)
REGISTER_DISPATCH(copysign_stub, &copysign_mps_kernel)
REGISTER_DISPATCH(nextafter_stub, &nextafter_mps_kernel)
REGISTER_DISPATCH(zeta_stub, &zeta_mps_kernel)
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_t_stub, &chebyshev_polynomial_t_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_u_stub, &chebyshev_polynomial_u_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_v_stub, &chebyshev_polynomial_v_mps_kernel)
REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_mps_kernel)
REGISTER_DISPATCH(hermite_polynomial_h_stub, &hermite_polynomial_h_mps_kernel)
REGISTER_DISPATCH(polar_stub, &polar_mps_kernel);
REGISTER_DISPATCH(complex_stub, &complex_mps_kernel);
} // namespace at::native
