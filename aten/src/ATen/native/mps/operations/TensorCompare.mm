//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/TensorCompare.h>
#include <c10/util/safe_conv.h>
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
  // Same for elements when one thread per test element searches them
  constexpr int64_t max_linear_scan_elements = 4;
  // Parallelize over test elements (sorting elements instead) once they outnumber elements by more than this
  constexpr int64_t test_parallel_min_ratio = 2;
  const auto common_type = at::result_type(elements, test_elements);
  const auto elements_c = elements.to(common_type).contiguous();
  const auto test_c = test_elements.to(common_type).contiguous();
  const auto test_parallel =
      test_c.numel() > max_linear_scan_test_elements && test_c.numel() > test_parallel_min_ratio * elements_c.numel();
  const auto& lookup_c = test_parallel ? elements_c : test_c;
  const auto sorted = lookup_c.numel() > (test_parallel ? max_linear_scan_elements : max_linear_scan_test_elements);
  const auto lookup = sorted ? std::get<0>(lookup_c.view(-1).sort()) : lookup_c;
  const auto lookup_size = c10::safe_conv<uint32_t>(lookup.numel(), "uint32_t");
  const auto found = test_parallel ? elements_c.new_zeros({elements_c.numel()}, kInt) : Tensor();
  const auto suffix = (sorted ? "sorted_" : "") + scalarToMetalTypeString(common_type);
  const auto out_c = out.contiguous();
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto enc = stream->commandEncoder();
      if (test_parallel) {
        auto mark = lib.getPipelineStateForFunc("isin_mark_" + suffix);
        [enc setComputePipelineState:mark];
        mtl_setArgs(enc, lookup, test_c, found, lookup_size);
        mtl_dispatch1DJob(enc, mark, test_c.numel());
      }
      auto pso = lib.getPipelineStateForFunc((test_parallel ? "isin_gather_" : "isin_") + suffix);
      [enc setComputePipelineState:pso];
      if (test_parallel) {
        mtl_setArgs(enc, elements_c, lookup, out_c, lookup_size, invert, found);
      } else {
        mtl_setArgs(enc, elements_c, lookup, out_c, lookup_size, invert);
      }
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
