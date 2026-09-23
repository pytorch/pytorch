#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ScaledMM.h>
#include <c10/util/TypeCast.h>
#include <fmt/format.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_mm_v2_native.h>
#endif

namespace at::native {
namespace mps {
namespace {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ScaledMM_metallib.h>
#endif

const std::array<scaled::ScaleKernelDispatchEntry, 2> scale_kernel_dispatch = {{
    {"tensorwise_tensorwise", scaled::check_tensorwise_recipe, scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE},
    {"rowwise_rowwise", scaled::check_rowwise_recipe, scaled::ScaledGemmImplementation::ROWWISE_ROWWISE},
}};

} // namespace
} // namespace mps

TORCH_IMPL_FUNC(_scaled_mm_mps_v2_out)
(const Tensor& mat_a,
 const Tensor& mat_b,
 const at::ITensorListRef& scale_a_list,
 IntArrayRef scale_recipe_a,
 IntArrayRef swizzle_a,
 const at::ITensorListRef& scale_b_list,
 IntArrayRef scale_recipe_b,
 IntArrayRef swizzle_b,
 at::OptionalTensorRef bias,
 std::optional<c10::ScalarType> out_dtype,
 IntArrayRef contraction_dim,
 bool use_fast_accum,
 const Tensor& out) {
  using namespace mps;
  check_mm_shapes(mat_a, mat_b, "_scaled_mm_v2");
  const auto m = mat_a.size(0);
  const auto n = mat_b.size(1);
  const auto k = mat_a.size(1);
  TORCH_CHECK_TYPE(
      mat_a.scalar_type() == kFloat8_e4m3fn, "Expected mat_a to be Float8_e4m3fn matrix got ", mat_a.scalar_type());
  TORCH_CHECK_TYPE(
      mat_b.scalar_type() == kFloat8_e4m3fn, "Expected mat_b to be Float8_e4m3fn matrix got ", mat_b.scalar_type());
  const auto dtype = out.scalar_type();
  TORCH_CHECK_TYPE(dtype == kFloat || dtype == kHalf || dtype == kBFloat16 || dtype == kFloat8_e4m3fn,
                   "MPS _scaled_mm_v2 supports Float, Half, BFloat16 and Float8_e4m3fn outputs, got ",
                   dtype);
  std::vector<Tensor> scale_a_vec(scale_a_list.begin(), scale_a_list.end());
  std::vector<Tensor> scale_b_vec(scale_b_list.begin(), scale_b_list.end());
  ArrayRef<Tensor> scale_a_ref(scale_a_vec);
  ArrayRef<Tensor> scale_b_ref(scale_b_vec);
  auto recipe_a = scaled::convert_int_to_enum<ScalingType>(scale_recipe_a);
  auto recipe_b = scaled::convert_int_to_enum<ScalingType>(scale_recipe_b);
  const auto swizzle_a_enum = scaled::convert_int_to_enum<SwizzleType>(swizzle_a);
  const auto swizzle_b_enum = scaled::convert_int_to_enum<SwizzleType>(swizzle_b);
  if (!swizzle_a_enum.empty() && !swizzle_b_enum.empty()) {
    TORCH_CHECK_VALUE(swizzle_a_enum[0] == SwizzleType::NO_SWIZZLE && swizzle_b_enum[0] == SwizzleType::NO_SWIZZLE,
                      "MPS does not support swizzle.");
  }
  const auto gemm_impl = scaled::find_scaled_gemm_impl(
      scale_kernel_dispatch, mat_a.scalar_type(), recipe_a, scale_a_ref, mat_b.scalar_type(), recipe_b, scale_b_ref);
  TORCH_CHECK_VALUE(
      gemm_impl != scaled::ScaledGemmImplementation::NONE,
      "MPS _scaled_mm_v2 only supports TensorWise or RowWise scaling with one Float scale per operand, got recipe_a=",
      scale_recipe_a,
      " and recipe_b=",
      scale_recipe_b);
  const auto& scale_a = scale_a_vec[0];
  const auto& scale_b = scale_b_vec[0];
  const bool rowwise = gemm_impl == scaled::ScaledGemmImplementation::ROWWISE_ROWWISE;
  if (rowwise) {
    TORCH_CHECK_VALUE(scale_a.stride(1) == 1, "expected scale_a.stride(1) to be 1, but got ", scale_a.stride(1));
    TORCH_CHECK_VALUE(scale_b.stride(1) == 1, "expected scale_b.stride(1) to be 1, but got ", scale_b.stride(1));
    TORCH_CHECK_VALUE(scale_a.sizes() == IntArrayRef({m, 1}) && scale_b.sizes() == IntArrayRef({1, n}) &&
                          scale_a.is_contiguous() && scale_b.is_contiguous(),
                      "scale_a must be (",
                      m,
                      ", 1) and scale_b (1, ",
                      n,
                      "), got ",
                      scale_a.sizes(),
                      " and ",
                      scale_b.sizes());
  }
  if (bias.has_value()) {
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
                              "MPS _scaled_mm_v2 requires m, n and k to fit in int32");
  assert_no_internal_overlap(out);
  if (k == 0) {
    out.zero_();
    return;
  }
  if (out.numel() == 0) {
    return;
  }
  const auto bias_vec = bias.has_value() ? std::make_optional(bias->contiguous().view({n})) : std::nullopt;
  const ScaledMMParams<> params{
      .m = c10::checked_convert<uint32_t>(m, "m"),
      .n = c10::checked_convert<uint32_t>(n, "n"),
      .k = c10::checked_convert<uint32_t>(k, "k"),
      .a_row_stride = mat_a.stride(0),
      .a_col_stride = mat_a.stride(1),
      .b_row_stride = mat_b.stride(0),
      .b_col_stride = mat_b.stride(1),
      .out_row_stride = out.stride(0),
      .out_col_stride = out.stride(1),
      .rowwise = rowwise,
      .has_bias = bias.has_value(),
      .bias_bfloat16 = bias.has_value() && bias->scalar_type() == kBFloat16,
  };
  using namespace std::string_view_literals;
  const auto type_str = scalarToMetalTypeString(dtype);
  const auto kernel_name = fmt::format("scaled_mm_{}{}", has_mpp() ? "mpp_"sv : ""sv, type_str);
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, kernel_name, {mat_a, mat_b}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, mat_a, mat_b, out, scale_a, scale_b, bias_vec, params);
      [encoder dispatchThreadgroups:MTLSizeMake(c10::metal::ceil_div<int64_t>(n, scaled_mm_tile),
                                                c10::metal::ceil_div<int64_t>(m, scaled_mm_tile),
                                                1)
              threadsPerThreadgroup:MTLSizeMake(scaled_mm_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

} // namespace at::native
