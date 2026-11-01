//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/TensorCompare.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/isin_native.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/where_native.h>
#endif

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/TensorCompare_metallib.h>
#endif

namespace mps {

static void isin_Tensor_Tensor_out_mps(const Tensor& elements,
                                       const Tensor& test_elements,
                                       bool invert,
                                       const Tensor& out) {
  TORCH_CHECK(elements.is_mps() && test_elements.is_mps());
  if (elements.numel() == 0 || test_elements.numel() == 0) {
    out.fill_(invert);
    return;
  }
  // Above this many test elements, sorting them once and binary searching beats a linear scan per element
  constexpr int64_t max_linear_scan_test_elements = 8;
  const auto common_type = at::result_type(elements, test_elements);
  const auto elements_c = elements.to(common_type).contiguous();
  const auto sorted = test_elements.numel() > max_linear_scan_test_elements;
  auto test_c = test_elements.to(common_type).contiguous();
  const auto test_s = sorted ? std::get<0>(test_c.view(-1).sort()) : std::move(test_c);
  const auto num_test = c10::checked_convert<uint32_t>(test_s.numel(), "uint32_t");
  const auto kernel = (sorted ? "isin_sorted_" : "isin_") + scalarToMetalTypeString(common_type);
  const auto out_c = out.contiguous();
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto enc = stream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(kernel);
      [enc setComputePipelineState:pso];
      mtl_setArgs(enc, elements_c, test_s, out_c, num_test, invert);
      mtl_dispatch1DJob(enc, pso, elements_c.numel());
    }
  });
  if (!out_c.is_same(out)) {
    out.copy_(out_c);
  }
}

} // namespace mps

// APIs exposed to at::native scope

TORCH_IMPL_FUNC(isin_Tensor_Tensor_out_mps)
(const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out) {
  mps::isin_Tensor_Tensor_out_mps(elements, test_elements, invert, out);
}
TORCH_IMPL_FUNC(isin_Scalar_Tensor_out_mps)
(const Scalar& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out) {
  at::native::resize_output(out, {});
  mps::isin_Tensor_Tensor_out_mps(mps::wrapped_scalar_tensor_mps(elements, kMPS), test_elements, invert, out);
}

static void where_kernel_mps(TensorIterator& iter) {
  lib.exec_ternary_kernel(iter, "where");
}

static void isneginf_kernel_mps(TensorIteratorBase& iter) {
  // bool output is non-floating, so ILP is opt-in (logical_not precedent);
  // the check is trivial ALU and bandwidth-bound at large sizes.
  lib.exec_unary_kernel(iter, "isneginf", std::nullopt, std::nullopt, /*ilp_threshold=*/1u << 18);
}

static void isposinf_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_unary_kernel(iter, "isposinf", std::nullopt, std::nullopt, /*ilp_threshold=*/1u << 18);
}

static void clamp_kernel_mps(TensorIteratorBase& iter) {
  lib.exec_ternary_kernel(iter, "clamp");
}

static void clamp_scalar_kernel_mps(TensorIteratorBase& iter, const Scalar& min_, const Scalar& max_) {
  auto scalar_type = iter.common_dtype();
  AT_DISPATCH_ALL_TYPES_AND3(c10::kBool, c10::kBFloat16, c10::kHalf, scalar_type, "clamp_scalar_mps", [&]() {
    ClampScalarParams<scalar_t> params{min_.to<scalar_t>(), max_.to<scalar_t>()};
    lib.exec_unary_kernel_with_params(
        iter, "clamp_scalar", params, fmt::format("ClampScalarParams_{}", mps::scalarToMetalTypeString(scalar_type)));
  });
}

static void clamp_min_scalar_kernel_mps(TensorIteratorBase& iter, Scalar min_) {
  lib.exec_unary_kernel(iter, "clamp_min_scalar", min_, iter.common_dtype());
}

static void clamp_max_scalar_kernel_mps(TensorIteratorBase& iter, Scalar max_) {
  lib.exec_unary_kernel(iter, "clamp_max_scalar", max_, iter.common_dtype());
}

REGISTER_DISPATCH(where_kernel, &where_kernel_mps)
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_mps)
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_mps)
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_mps)
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_mps)
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_mps)
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_mps)

} // namespace at::native
