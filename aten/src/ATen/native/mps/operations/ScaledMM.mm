#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/ScaledMM.h>
#include <c10/util/safe_conv.h>
#include <fmt/format.h>
#include <utility>

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

const std::array<scaled::ScaleKernelDispatchEntry, 5> scale_kernel_dispatch = {{
    {"tensorwise_tensorwise", scaled::check_tensorwise_recipe, scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE},
    {"rowwise_rowwise", scaled::check_rowwise_recipe, scaled::ScaledGemmImplementation::ROWWISE_ROWWISE},
    {"mxfp8_mxfp8", scaled::check_mxfp8_recipe, scaled::ScaledGemmImplementation::MXFP8_MXFP8},
    {"nvfp4_nvfp4", scaled::check_nvfp4_recipe, scaled::ScaledGemmImplementation::NVFP4_NVFP4},
    {"nvfp4_nvfp4_single_scale",
     scaled::check_nvfp4_recipe_single_scale,
     scaled::ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE},
}};

// Block-scaled recipes read A row-major and B column-major (both K-contiguous, like cuBLASLt) with unswizzled
// [rows, K / block] scales; the validator has checked the scale shapes.
void scaled_mm_blockwise_out(const Tensor& mat_a,
                             const Tensor& mat_b,
                             ArrayRef<Tensor> scale_a,
                             ArrayRef<Tensor> scale_b,
                             scaled::ScaledGemmImplementation gemm_impl,
                             const std::optional<Tensor>& bias,
                             int64_t k,
                             const Tensor& out) {
  TORCH_CHECK_VALUE(mat_a.stride(1) == 1 && mat_b.stride(0) == 1,
                    "MPS block-scaled _scaled_mm_v2 expects a row-major mat_a and a column-major mat_b, got strides ",
                    mat_a.strides(),
                    " and ",
                    mat_b.strides());
  TORCH_CHECK_TYPE(out.scalar_type() != kFloat8_e4m3fn, "MPS block-scaled _scaled_mm_v2 does not support fp8 output");
  const bool nvfp4 = gemm_impl != scaled::ScaledGemmImplementation::MXFP8_MXFP8;
  const bool two_level = gemm_impl == scaled::ScaledGemmImplementation::NVFP4_NVFP4;
  const auto tensor_scale = [&](ArrayRef<Tensor> scales) {
    return two_level ? std::make_optional(scales[1]) : std::nullopt;
  };
  const ScaledMMParams<> params{
      .m = c10::safe_conv<uint32_t>(mat_a.size(0), "mat_a.size(0)"),
      .n = c10::safe_conv<uint32_t>(mat_b.size(1), "mat_b.size(1)"),
      .k = c10::safe_conv<uint32_t>(k, "K"),
      .a_row_stride = mat_a.stride(0),
      .a_col_stride = mat_a.stride(1),
      .b_row_stride = mat_b.stride(0),
      .b_col_stride = mat_b.stride(1),
      .out_row_stride = out.stride(0),
      .out_col_stride = out.stride(1),
      .rowwise = false,
      .has_tensor_scales = two_level,
      .has_bias = bias.has_value(),
      .bias_bfloat16 = bias.has_value() && bias->scalar_type() == kBFloat16,
  };
  // Metal 4.1 MPP applies MX block scales natively. It needs whole blocks along K and dense operands, since it derives
  // the scale plane's row pitch from the data's, and indexes each tile in int32.
  const bool native = !nvfp4 && has_metal_4_1() && k % 32 == 0 && mat_a.stride(0) == k && mat_b.stride(1) == k &&
      std::in_range<int32_t>(scaled_mm_native_tile * k);
  // The few rows kernel reads whole chunks of K with 16-byte vector loads
  const auto aligned16 = [](const Tensor& mat, int64_t leading_stride) {
    return mat.storage_offset() % 16 == 0 && leading_stride % 16 == 0;
  };
  const bool few_rows = params.m <= scaled_mm_few_rows_max && k % scaled_mm_blockwise_chunk == 0 &&
      aligned16(mat_a, mat_a.stride(0)) && aligned16(mat_b, mat_b.stride(1));
  const auto type_str = scalarToMetalTypeString(out.scalar_type());
  const auto format = nvfp4 ? "nvfp4" : "mxfp8";
  const auto kernel_name = few_rows ? fmt::format("scaled_mm_few_rows_{}_{}_{}", format, params.m, type_str)
      : native                      ? fmt::format("scaled_mm_native_mxfp8_{}", type_str)
                                    : fmt::format("scaled_mm_{}{}_{}", has_mpp() ? "mpp_" : "", format, type_str);
  const auto tile = native ? scaled_mm_native_tile : scaled_mm_tile;
  const auto groups = few_rows
      ? MTLSizeMake(c10::metal::ceil_div<int64_t>(params.n, scaled_mm_simdgroups), 1, 1)
      : MTLSizeMake(c10::metal::ceil_div<int64_t>(params.n, tile), c10::metal::ceil_div<int64_t>(params.m, tile), 1);
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, kernel_name, {mat_a, mat_b}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder,
                  mat_a,
                  mat_b,
                  out,
                  scale_a[0],
                  scale_b[0],
                  tensor_scale(scale_a),
                  tensor_scale(scale_b),
                  bias,
                  params);
      [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(scaled_mm_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

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
  TORCH_CHECK_TYPE(mat_a.scalar_type() == kFloat8_e4m3fn || mat_a.scalar_type() == kFloat4_e2m1fn_x2,
                   "Expected mat_a to be Float8_e4m3fn or Float4_e2m1fn_x2 matrix got ",
                   mat_a.scalar_type());
  TORCH_CHECK_TYPE(mat_b.scalar_type() == kFloat8_e4m3fn || mat_b.scalar_type() == kFloat4_e2m1fn_x2,
                   "Expected mat_b to be Float8_e4m3fn or Float4_e2m1fn_x2 matrix got ",
                   mat_b.scalar_type());
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
  const auto gemm_impl = scaled::find_scaled_gemm_impl(scale_kernel_dispatch,
                                                       mat_a.scalar_type(),
                                                       recipe_a,
                                                       scale_a_ref,
                                                       swizzle_a_enum,
                                                       mat_b.scalar_type(),
                                                       recipe_b,
                                                       scale_b_ref,
                                                       swizzle_b_enum);
  TORCH_CHECK_VALUE(
      gemm_impl != scaled::ScaledGemmImplementation::NONE,
      "MPS _scaled_mm_v2 supports TensorWise or RowWise scaling with one Float scale per operand, MXFP8 and NVFP4, "
      "got recipe_a=",
      scale_recipe_a,
      " and recipe_b=",
      scale_recipe_b);
  const bool blockwise = gemm_impl == scaled::ScaledGemmImplementation::MXFP8_MXFP8 ||
      gemm_impl == scaled::ScaledGemmImplementation::NVFP4_NVFP4 ||
      gemm_impl == scaled::ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE;
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
  TORCH_CHECK_NOT_IMPLEMENTED(std::in_range<int32_t>(m) && std::in_range<int32_t>(n) && std::in_range<int32_t>(k),
                              "MPS _scaled_mm_v2 requires m, n and k to fit in int32");
  assert_no_internal_overlap(out);
  if (k == 0) {
    out.zero_();
    return;
  }
  if (out.numel() == 0) {
    return;
  }
  if (blockwise) {
    const auto bias_vec = bias.has_value() ? std::make_optional(bias->contiguous().view({n})) : std::nullopt;
    // Float4_e2m1fn_x2 packs two values along K
    const auto k_unpacked = mat_a.scalar_type() == kFloat4_e2m1fn_x2 ? 2 * k : k;
    scaled_mm_blockwise_out(mat_a, mat_b, scale_a_ref, scale_b_ref, gemm_impl, bias_vec, k_unpacked, out);
    return;
  }
  TORCH_CHECK_VALUE(k % scaled_mm_alignment == 0 && n % scaled_mm_alignment == 0,
                    "Expected K and N to be divisible by ",
                    scaled_mm_alignment,
                    " but got mat_a shape ",
                    mat_a.sizes(),
                    " and mat_b shape ",
                    mat_b.sizes());
  const auto check_alignment = [](const Tensor& mat, std::string_view name) {
    if (mat.stride(0) != 1 && mat.stride(1) != 1) {
      return;
    }
    const auto leading_stride = mat.stride(1) == 1 ? mat.stride(0) : mat.stride(1);
    TORCH_CHECK_VALUE(mat.storage_offset() % scaled_mm_alignment == 0 && leading_stride % scaled_mm_alignment == 0,
                      "Expected ",
                      name,
                      " storage offset and leading stride to be divisible by ",
                      scaled_mm_alignment,
                      " but got storage offset ",
                      mat.storage_offset(),
                      " and strides ",
                      mat.strides());
  };
  check_alignment(mat_a, "mat_a");
  check_alignment(mat_b, "mat_b");
  const auto bias_vec = bias.has_value() ? std::make_optional(bias->contiguous().view({n})) : std::nullopt;
  const ScaledMMParams<> params{
      .m = c10::safe_conv<uint32_t>(m, "mat_a.size(0)"),
      .n = c10::safe_conv<uint32_t>(n, "mat_b.size(1)"),
      .k = c10::safe_conv<uint32_t>(k, "mat_a.size(1)"),
      .a_row_stride = mat_a.stride(0),
      .a_col_stride = mat_a.stride(1),
      .b_row_stride = mat_b.stride(0),
      .b_col_stride = mat_b.stride(1),
      .out_row_stride = out.stride(0),
      .out_col_stride = out.stride(1),
      .rowwise = rowwise,
      .has_tensor_scales = true,
      .has_bias = bias.has_value(),
      .bias_bfloat16 = bias.has_value() && bias->scalar_type() == kBFloat16,
  };
  using namespace std::string_view_literals;
  const auto type_str = scalarToMetalTypeString(dtype);
  const bool k_contiguous = mat_a.stride(1) == 1 && mat_b.stride(0) == 1;
  const bool few_rows = m <= scaled_mm_few_rows_max && k_contiguous;
  // MPP tensors take int32 strides
  const bool native = !few_rows && k_contiguous && has_metal_4_1() && std::in_range<int32_t>(mat_a.stride(0)) &&
      std::in_range<int32_t>(mat_b.stride(1));
  const auto kernel_name = few_rows ? fmt::format("scaled_mm_few_rows_{}_{}", m, type_str)
      : native                      ? fmt::format("scaled_mm_native_{}", type_str)
                                    : fmt::format("scaled_mm_{}{}", has_mpp() ? "mpp_"sv : ""sv, type_str);
  const auto tile = native ? scaled_mm_native_tile : scaled_mm_tile;
  const auto cols_per_group = few_rows ? scaled_mm_simdgroups : tile;
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, kernel_name, {mat_a, mat_b}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, mat_a, mat_b, out, scale_a, scale_b, bias_vec, params);
      [encoder dispatchThreadgroups:MTLSizeMake(c10::metal::ceil_div<int64_t>(n, cols_per_group),
                                                few_rows ? 1 : c10::metal::ceil_div<int64_t>(m, tile),
                                                1)
              threadsPerThreadgroup:MTLSizeMake(scaled_mm_threads, 1, 1)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

} // namespace at::native
