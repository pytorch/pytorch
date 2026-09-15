#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ScaledMM.h>
#include <c10/util/TypeCast.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {
namespace mps {
namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ScaledMM_metallib.h>
#endif

} // namespace
} // namespace mps

Tensor& _scaled_mm_out_mps(const Tensor& self,
                           const Tensor& mat2,
                           const Tensor& scale_a,
                           const Tensor& scale_b,
                           const std::optional<Tensor>& bias,
                           const std::optional<Tensor>& scale_result,
                           std::optional<ScalarType> out_dtype,
                           bool use_fast_accum,
                           Tensor& out) {
  using namespace mps;
  check_mm_shapes(self, mat2, "_scaled_mm");
  const auto m = self.size(0);
  const auto n = mat2.size(1);
  const auto k = self.size(1);
  const bool tensorwise = scale_a.numel() == 1 && scale_b.numel() == 1;
  const bool rowwise = scale_a.sizes() == IntArrayRef({m, 1}) && scale_b.sizes() == IntArrayRef({1, n}) &&
      scale_a.is_contiguous() && scale_b.is_contiguous();
  if (!(scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat && (tensorwise || rowwise))) {
    if (scale_a.numel() == 1) {
      TORCH_CHECK_VALUE(scale_a.scalar_type() == kFloat, "scale_a must have 1 Float element");
      TORCH_CHECK_VALUE(scale_b.numel() == 1 && scale_b.scalar_type() == kFloat, "scale_b must have 1 Float element");
    } else {
      TORCH_CHECK_VALUE(scale_a.numel() == m && scale_a.scalar_type() == kFloat,
                        "scale_a must have ",
                        m,
                        " Float elements, got ",
                        scale_a.numel());
      TORCH_CHECK_VALUE(scale_b.numel() == n && scale_b.scalar_type() == kFloat,
                        "scale_b must have ",
                        n,
                        " Float elements, got ",
                        scale_b.numel());
      TORCH_CHECK_VALUE(scale_a.stride(1) == 1, "expected scale_a.stride(1) to be 1, but got ", scale_a.stride(1));
      TORCH_CHECK_VALUE(scale_b.stride(1) == 1, "expected scale_b.stride(1) to be 1, but got ", scale_b.stride(1));
    }
    TORCH_CHECK_VALUE(false,
                      "scale_a must be (",
                      m,
                      ", 1) and scale_b (1, ",
                      n,
                      "), got ",
                      scale_a.sizes(),
                      " and ",
                      scale_b.sizes());
  }
  TORCH_CHECK_VALUE(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
                    "scale_result must be a float scalar");
  TORCH_CHECK_VALUE(!bias || bias->numel() == n, "Bias must be size ", n, " but got ", bias->numel());
  TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK_TYPE(
      self.scalar_type() == kFloat8_e4m3fn, "Expected self to be Float8_e4m3fn matrix got ", self.scalar_type());
  TORCH_CHECK_TYPE(
      mat2.scalar_type() == kFloat8_e4m3fn, "Expected mat2 to be Float8_e4m3fn matrix got ", mat2.scalar_type());
  const auto dtype = out.scalar_type();
  TORCH_CHECK_TYPE(dtype == kFloat || dtype == kHalf || dtype == kBFloat16 || dtype == kFloat8_e4m3fn,
                   "MPS _scaled_mm supports Float, Half, BFloat16 and Float8_e4m3fn outputs, got ",
                   dtype);
  if (bias) {
    TORCH_CHECK_VALUE(dtype != kFloat, "Bias is not supported when out_dtype is set to Float32");
    TORCH_CHECK_VALUE(bias->scalar_type() == kBFloat16 || bias->scalar_type() == kHalf,
                      "Bias must be BFloat16 or Half, but got ",
                      bias->scalar_type());
    TORCH_CHECK_VALUE(dtype != kBFloat16 || bias->scalar_type() == kBFloat16,
                      "Bias must be BFloat16 to compute ",
                      dtype,
                      " output, but got ",
                      bias->scalar_type());
    TORCH_CHECK_VALUE(dtype != kHalf || bias->scalar_type() == kHalf,
                      "Bias must be Float16 to compute ",
                      dtype,
                      " output, but got ",
                      bias->scalar_type());
  }
  // Kernel index math is 64-bit; the uint32 params and the uint row/col/k arithmetic need each dim to fit int32.
  TORCH_CHECK_NOT_IMPLEMENTED(m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX,
                              "MPS _scaled_mm requires m, n and k to fit in int32");
  resize_output(out, {m, n});
  assert_no_internal_overlap(out);
  if (out.numel() == 0) {
    return out;
  }
  if (k == 0) {
    return out.zero_();
  }
  const auto bias_vec = bias ? std::make_optional(bias->contiguous().view({n})) : std::nullopt;
  const ScaledMMParams<> params{
      .m = c10::checked_convert<uint32_t>(m, "m"),
      .n = c10::checked_convert<uint32_t>(n, "n"),
      .k = c10::checked_convert<uint32_t>(k, "k"),
      .a_row_stride = self.stride(0),
      .a_col_stride = self.stride(1),
      .b_row_stride = mat2.stride(0),
      .b_col_stride = mat2.stride(1),
      .out_row_stride = out.stride(0),
      .out_col_stride = out.stride(1),
      .rowwise = !tensorwise,
      .has_bias = bias.has_value(),
      .bias_bfloat16 = bias && bias->scalar_type() == kBFloat16,
      .has_scale_result = scale_result.has_value(),
  };
  using namespace std::string_view_literals;
  const auto type_str = scalarToMetalTypeString(dtype);
  const auto kernel_name = fmt::format("scaled_mm_{}{}", has_mpp() ? "mpp_"sv : ""sv, type_str);
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, kernel_name, {self, mat2}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, self, mat2, out, scale_a, scale_b, bias_vec, scale_result, params);
      [encoder dispatchThreadgroups:MTLSizeMake(c10::metal::ceil_div<int64_t>(n, scaled_mm_tile),
                                                c10::metal::ceil_div<int64_t>(m, scaled_mm_tile),
                                                1)
              threadsPerThreadgroup:MTLSizeMake(scaled_mm_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  return out;
}

Tensor _scaled_mm_mps(const Tensor& self,
                      const Tensor& mat2,
                      const Tensor& scale_a,
                      const Tensor& scale_b,
                      const std::optional<Tensor>& bias,
                      const std::optional<Tensor>& scale_result,
                      std::optional<ScalarType> out_dtype,
                      bool use_fast_accum) {
  auto out = at::empty({0}, self.options().dtype(out_dtype.value_or(self.scalar_type())));
  return _scaled_mm_out_mps(self, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

} // namespace at::native
