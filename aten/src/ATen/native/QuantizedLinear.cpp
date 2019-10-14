#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtilsMulti.h"
#include "ATen/cpp_custom_type_hack.h"

#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmFP16.h"
#include "fbgemm/QuantUtils.h"
#endif // USE_FBGEMM

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include <chrono>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(fbgemm::PackBMatrix<int8_t>);
CAFFE_KNOWN_TYPE(fbgemm::PackedGemmMatrixFP16);
#endif // USE_FBGEMM
} // namespace caffe2

namespace at {
namespace native {

#ifdef USE_FBGEMM

Tensor fbgemm_linear_int8_weight_fp32_activation(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& packed,
    const Tensor& col_offsets,
    Scalar weight_scale,
    Scalar weight_zero_point,
    const Tensor& bias) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  auto input_contig = input.contiguous();
  auto* input_ptr = input_contig.data_ptr<float>();

  TORCH_CHECK(input.dim() >= 2);
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(K == weight.size(1));
  auto N = weight.size(0);
  TORCH_CHECK(bias.dim() == 1);
  TORCH_CHECK(bias.size(0) == N);
  TORCH_CHECK(weight_scale.isFloatingPoint());
  TORCH_CHECK(weight_zero_point.isIntegral(false));

  // Calculate statistics for quantization of the input Tensor
  float x_min, x_max;
  fbgemm::FindMinMax(
      /*m=*/input_ptr,
      /*min=*/&x_min,
      /*max=*/&x_max,
      /*len=*/input.numel());

  // Input tensor is quantized as 8-bit unsigned values
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;
  static constexpr int bound = (1 << (precision - 1));

  // Calculate scale and zero point for quantization of input tensor
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/is_signed ? -bound : 0,
      /*qmax=*/is_signed ? (bound - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false);

  q_params.precision = precision;

  // This operation does the following:
  // 1) Quantizes the input matrix given the statistics we've calculated above
  // 2) Creates a "row buffer" vector with offset values that must be added
  //    to the integer matrix multiplication operation to ensure correctness
  // 3) Packs the resulting quantized matrix into vector-register and cache
  //    friendly tiles.
  //
  //  Note this is not executed eagerly, but rather within the fbgemmPacked call
  //  below.
  fbgemm::PackAWithQuantRowOffset<uint8_t> packA(
      /*trans=*/fbgemm::matrix_op_t::NoTranspose,
      /*nRow=*/M,
      /*nCol=*/K,
      /*smat=*/input_ptr,
      /*ld=*/K,
      /*pmat=*/nullptr, // packA manages ownership of `pmat`
      /*scale=*/q_params.scale,
      /*zero_pt=*/q_params.zero_point);

  // ReQuantizeForFloat requires pointers to the scale and zero point values,
  // since in the case of rowwise quantization these will be arrays rather than
  // scalars. But in this case, we're doing whole-tensor quantization so we just
  // pass a pointer to the scale values (and internally ReQuantizeFor Float
  // won't index past 0
  float weight_scale_float = static_cast<float>(weight_scale.to<double>());
  int32_t weight_zero_point_int32 =
      static_cast<int32_t>(weight_zero_point.to<int64_t>());

  // This is the end of the pipeline, pass the resulting matrix through
  fbgemm::DoNothing<float, float> doNothingObj{};

  auto bias_contig = bias.contiguous();

  // After the uint8 * int8 matrix multiplication is performed, this operation
  // does:
  //  1) Add in row and column offsets to the rows and columns, respectively
  //  2) Dequantize the results into floating point
  //  3) Add in the bias term
  fbgemm::ReQuantizeForFloat</*FUSE_RELU*/false> outputProcObj(
      /*nextop=*/doNothingObj,
      /*Aq_scale=*/q_params.scale,
      /*Bq_scale=*/&weight_scale_float,
      /*Aq_zero_point=*/q_params.zero_point,
      /*Bq_zero_point=*/&weight_zero_point_int32,
      /*row_offsets=*/packA.getRowOffsetBuffer(),
      /*col_offsets=*/col_offsets.data_ptr<int32_t>(),
      /*bias=*/bias_contig.data_ptr<float>(),
      /*nCol=*/N);

  // Allocate output Tensor and a buffer for fbgemmPacked to use
  auto output = at::zeros(
      {M, N}, bias.options().dtype(at::kFloat));
  auto buffer = at::zeros_like(output, output.options().dtype(at::kInt));

  // Pull out the PackBMatrix instance from the owning tensor
  auto& packB = cpp_custom_type_hack::cast<fbgemm::PackBMatrix<int8_t>>(packed);

  // Do the GEMM
  fbgemm::fbgemmPacked(
      /*packA=*/packA,
      /*packB=*/packB,
      /*C=*/output.data_ptr<float>(),
      /*C_buffer=*/buffer.data_ptr<int32_t>(),
      /*ldc=*/N,
      /*outProcess=*/outputProcObj,
      /*thread_id=*/0,
      /*num_threads=*/1);

  // The resulting matrix here is 2-D, let's view it with the original
  // left hand dimensions of the input.
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  return output.view(out_sizes);
}

Tensor fbgemm_linear_int8_weight(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& packed,
    const Tensor& col_offsets,
    Scalar weight_scale,
    Scalar weight_zero_point,
    const Tensor& bias) {
  // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
  // TORCH_WARN(
  //     "fbgemm_linear_int8_weight will be deprecated soon."
  //     "Please use fbgemm_linear_int8_weight_fp32_activation instead.");

  return at::native::fbgemm_linear_int8_weight_fp32_activation(
      input,
      weight,
      packed,
      col_offsets,
      weight_scale,
      weight_zero_point,
      bias);
}

namespace {
// Calculate the column offsets
// Note this includes the sum of the columns as well as the scalar term
// B_zero_point * K, whereas the row_offsets created by
// PackAWithQuantRowOffset is only the sum of the A rows.
void calc_col_offsets_transpose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t B_zero_point,
    int32_t* col_offsets) {
  for (size_t i = 0; i < N; ++i) {
    int32_t sum = 0;
    for (size_t j = 0; j < K; ++j) {
      sum += Bint8[i * K + j];
    }
    col_offsets[i] = sum - B_zero_point * K;
  }
}
} // namespace

std::tuple<Tensor, Tensor, double, int64_t> fbgemm_linear_quantize_weight(
    const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
  auto weight_contig = weight.contiguous();

  // Calculate weight statistics
  float w_min, w_max;
  fbgemm::FindMinMax(
      /*m=*/weight_contig.data_ptr<float>(),
      /*min=*/&w_min,
      /*max=*/&w_max,
      /*len=*/weight_contig.numel());

  // Choose parameters for quantizing the weight as 8-bit signed integer
  static constexpr bool is_signed = true;
  static constexpr int precision = 8;
  static constexpr int bound = (1 << (precision - 1));
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/w_min,
      /*max=*/w_max,
      /*qmin=*/is_signed ? -bound : 0,
      /*qmax=*/is_signed ? (bound - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false);

  q_params.precision = precision;

  auto quantized = at::zeros_like(weight_contig).to(at::kChar).contiguous();
  fbgemm::Quantize<int8_t>(
      /*src=*/weight_contig.data_ptr<float>(),
      /*dst=*/quantized.data_ptr<int8_t>(),
      /*len=*/weight_contig.numel(),
      /*qparams=*/q_params);

  // Calculate column offsets of the weight and store them away in a tensor.
  // Similarly to quantization, this can be done once and cached.
  auto col_offsets =
      at::zeros_like(quantized).sum({1}).to(at::kInt).contiguous();
  calc_col_offsets_transpose(
      /*K=*/quantized.size(1),
      /*N=*/quantized.size(0),
      /*Bint8=*/quantized.data_ptr<int8_t>(),
      /*B_zero_point=*/q_params.zero_point,
      /*col_offsets=*/col_offsets.data_ptr<int32_t>());

  return std::make_tuple(
      quantized, col_offsets, q_params.scale, q_params.zero_point);
}

Tensor fbgemm_pack_quantized_matrix(const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");
  int64_t K = weight.size(1);
  int64_t N = weight.size(0);
  auto weight_contig = weight.contiguous();
  auto contiguous_ptr = weight_contig.data_ptr<int8_t>();
  auto ptr = guts::make_unique<fbgemm::PackBMatrix<int8_t>>(
      /*trans=*/fbgemm::matrix_op_t::Transpose,
      /*nRow=*/K,
      /*nCol=*/N,
      /*smat=*/contiguous_ptr,
      /*ld=*/K,
      /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
      /*groups=*/1);
  return cpp_custom_type_hack::create(std::move(ptr), weight.options());
}

Tensor fbgemm_pack_quantized_matrix(
    const Tensor& weight,
    int64_t K,
    int64_t N) {
  // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
  // TORCH_WARN(
  //     "fbgemm_pack_quantized_matrix(weight, K, N) will be deprecated soon."
  //     "Please use fbgemm_pack_quantized_matrix(weight) instead.");

  return at::native::fbgemm_pack_quantized_matrix(weight);
}

float raw_uint16_to_fp16(unsigned short value) {
  // Convert raw 16 bits half precision floating point number
  // to single precision floating point number.
  unsigned short sign_bits = value >> 15;
  unsigned short exponent_bits = value >> 10 & 0x1f;
  unsigned short significand_bits = value & 0x3ff;

  float sign = sign_bits ? -1 : 1;
  float significand = 1 + significand_bits * 0x1p-10;
  float exponent = exponent_bits - 0xf;

  return sign * std::ldexp(significand, exponent);
}

template <typename T>
bool check_and_saturate(T* element, T MAX) {
  if (*element > MAX) {
    *element = MAX;
    return true;
  }
  if (*element < -MAX) {
    *element = -MAX;
    return true;
  }
  return false;
}

// The range for using FP16 quantization of weights requires that the elements
// should be in the range of [5.96e-8, 65504]. If it is out of range, then the
// number will be saturated to max or min representable values by FP16.
void handle_weights_saturation(float* weight, int64_t length) {
  float FP16_MAX = raw_uint16_to_fp16(0x7BFF);
  bool found_out_of_range = false;

  for (int i = 0; i < length; ++i) {
    if (check_and_saturate<float>(&weight[i], FP16_MAX)) {
      found_out_of_range = true;
    }
  }

  if (found_out_of_range) {
    TORCH_WARN("FOUND weight out of range ");
  }
}

Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  int64_t K = weight.size(1);
  int64_t N = weight.size(0);
  Tensor weight_contig = weight.contiguous();
  auto weight_contig_ptr = weight_contig.data_ptr<float>();

  handle_weights_saturation(weight_contig_ptr, K * N);

  // TODO(mingzhe09088):
  // Consider using a functor here in PackedGemmMatrixFP16
  // Comments from (XQ): Not entirely sure this make_unique is safe. make_unique
  // is created with regular "new", and freed through TypeMetaData::deleteFn in
  // this function. This is perfectly fine if the tensors are created and freed
  // within this translation unit. It might be very problematic if that tensor
  // flows across dll boundaries.
  auto ptr = guts::make_unique<fbgemm::PackedGemmMatrixFP16>(
      fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr);
  return cpp_custom_type_hack::create(std::move(ptr), weight.options());
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  auto input_contig = input.contiguous();
  auto* input_ptr = input_contig.data_ptr<float>();

  // Pull out the PackedGemmMatrixFP16 instance from the owning tensor
  const fbgemm::PackedGemmMatrixFP16& packed_weight_fp16 =
      cpp_custom_type_hack::cast<fbgemm::PackedGemmMatrixFP16>(packed_weight);

  TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
  TORCH_CHECK(input.dim() >= 2);
  TORCH_CHECK(bias.dim() == 1);

  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  int64_t N = packed_weight_fp16.numCols();

  auto output = at::empty({M, N}, bias.options().dtype(at::kFloat));

  // Call the fp16 gemm interface
  fbgemm::cblas_gemm_compute(
      fbgemm::matrix_op_t::NoTranspose,
      M,
      input_ptr,
      packed_weight_fp16,
      0.0f,
      output.data_ptr<float>());

  // Add bias term
  output.add_(bias);

  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  return output.view(out_sizes);
}

Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
  // TORCH_WARN(
  //     "fbgemm_linear_fp16_weight will be deprecated soon."
  //     "Please use fbgemm_linear_fp16_weight_fp32_activation instead.");

  return at::native::fbgemm_linear_fp16_weight_fp32_activation(
      input, packed_weight, bias);
}

#else // USE_FBGEMM

Tensor fbgemm_linear_int8_weight_fp32_activation(
    const Tensor& /*input*/,
    const Tensor& /*weight*/,
    const Tensor& /*packed*/,
    const Tensor& /*col_offsets*/,
    Scalar /*weight_scale*/,
    Scalar /*weight_zero_point*/,
    const Tensor& /*bias*/) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_int8_weight(
    const Tensor& /*input*/,
    const Tensor& /*weight*/,
    const Tensor& /*packed*/,
    const Tensor& /*col_offsets*/,
    Scalar /*weight_scale*/,
    Scalar /*weight_zero_point*/,
    const Tensor& /*bias*/) {
  // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
  // TORCH_WARN(
  //     "fbgemm_linear_int8_weight will be deprecated soon."
  //     "Please use fbgemm_linear_int8_weight_fp32_activation instead.");

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

std::tuple<Tensor, Tensor, double, int64_t> fbgemm_linear_quantize_weight(
    const Tensor& /*weight*/) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_pack_quantized_matrix(const Tensor& /*input*/) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_pack_quantized_matrix(
    const Tensor& /*input*/,
    int64_t /*K*/,
    int64_t /*N*/) {
  // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
  // TORCH_WARN(
  //     "fbgemm_pack_quantized_matrix(weight, K, N) will be deprecated soon."
  //     "Please use fbgemm_pack_quantized_matrix(weight) instead.");

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight_fp32_activation(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_linear_fp16_weight(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& bias) {
  // Replace after https://github.com/pytorch/pytorch/issues/24354 is fixed
  // TORCH_WARN(
  //     "fbgemm_linear_fp16_weight will be deprecated soon."
  //     "Please use fbgemm_linear_fp16_weight_fp32_activation instead.");

  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

bool fbgemm_is_cpu_supported() {
  return false;
}

#endif // USE_FBGEMM
} // namespace native
} // namespace at
