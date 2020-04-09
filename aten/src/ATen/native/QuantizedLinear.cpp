#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Parallel.h"
#include "ATen/WrapDimUtilsMulti.h"
#include "ATen/cpp_custom_type_hack.h"

#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmFP16.h"
#include "fbgemm/QuantUtils.h"
#endif // USE_FBGEMM

#ifdef USE_FBGEMM
namespace caffe2 {
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(fbgemm::PackBMatrix<int8_t>);
CAFFE_KNOWN_TYPE(fbgemm::PackedGemmMatrixFP16);
} // namespace caffe2
#endif // USE_FBGEMM

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

  const Tensor input_contig = input.contiguous();
  const float* input_ptr = input_contig.data_ptr<float>();

  TORCH_CHECK(input.dim() >= 2);
  const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  const int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(K == weight.size(1));
  const int64_t N = weight.size(0);
  TORCH_CHECK(bias.dim() == 1);
  TORCH_CHECK(bias.size(0) == N);
  TORCH_CHECK(weight_scale.isFloatingPoint());
  TORCH_CHECK(weight_zero_point.isIntegral(false));

  // Calculate statistics for quantization of the input Tensor
  float x_min;
  float x_max;
  fbgemm::FindMinMax(
      /*m=*/input_ptr,
      /*min=*/&x_min,
      /*max=*/&x_max,
      /*len=*/input.numel());

  // Input tensor is quantized as 8-bit unsigned values
  constexpr int kPrecision = 8;
  constexpr bool kIsSigned = false;
  constexpr int kBound = (1 << (kPrecision - 1));

  // Calculate scale and zero point for quantization of input tensor
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/kIsSigned ? -kBound : 0,
      /*qmax=*/kIsSigned ? (kBound - 1) : (1 << kPrecision) - 1,
      /*preserve_sparsity=*/false);
  q_params.precision = kPrecision;

  // ReQuantizeForFloat requires pointers to the scale and zero point values,
  // since in the case of rowwise quantization these will be arrays rather than
  // scalars. But in this case, we're doing whole-tensor quantization so we just
  // pass a pointer to the scale values (and internally ReQuantizeFor Float
  // won't index past 0
  const float weight_scale_float =
      static_cast<float>(weight_scale.to<double>());
  const int32_t weight_zero_point_int32 =
      static_cast<int32_t>(weight_zero_point.to<int64_t>());

  const Tensor bias_contig = bias.contiguous();

  // Allocate output Tensor and a buffer for fbgemmPacked to use
  std::vector<int64_t> output_size = input.sizes().vec();
  output_size.back() = N;
  Tensor output = at::empty(output_size, input.options().dtype(at::kFloat), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor buffer = at::empty(output_size, input.options().dtype(at::kInt), LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // Pull out the PackBMatrix instance from the owning tensor
  auto& pack_b =
      cpp_custom_type_hack::cast<fbgemm::PackBMatrix<int8_t>>(packed);

  const int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    // This operation does the following:
    // 1) Quantizes the input matrix given the statistics we've calculated
    //    above.
    // 2) Creates a "row buffer" vector with offset values that must be added
    //    to the integer matrix multiplication operation to ensure correctness.
    // 3) Packs the resulting quantized matrix into vector-register and cache
    //    friendly tiles.
    //
    //  Note this is not executed eagerly, but rather within the fbgemmPacked
    //  call below.
    fbgemm::PackAWithQuantRowOffset<uint8_t> pack_a(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr, // pack_a manages ownership of `pmat`
        /*scale=*/q_params.scale,
        /*zero_pt=*/q_params.zero_point);

    // This is the end of the pipeline, pass the resulting matrix through
    fbgemm::DoNothing<float, float> kDoNothingObj{};
    for (int task_id = begin; task_id < end; ++task_id) {
      // After the uint8 * int8 matrix multiplication is performed, this
      // operation does:
      //  1) Add in row and column offsets to the rows and columns, respectively
      //  2) Dequantize the results into floating point
      //  3) Add in the bias term
      fbgemm::ReQuantizeForFloat</* FUSE_RELU */ false> output_proc_obj(
          /*nextop=*/kDoNothingObj,
          /*Aq_scale=*/q_params.scale,
          /*Bq_scale=*/&weight_scale_float,
          /*Aq_zero_point=*/q_params.zero_point,
          /*Bq_zero_point=*/&weight_zero_point_int32,
          /*row_offsets=*/pack_a.getRowOffsetBuffer(),
          /*col_offsets=*/col_offsets.data_ptr<int32_t>(),
          /*bias=*/bias_contig.data_ptr<float>(),
          /*nCol=*/N);
      // Do the GEMM
      fbgemm::fbgemmPacked(
          /*packA=*/pack_a,
          /*packB=*/pack_b,
          /*C=*/output.data_ptr<float>(),
          /*C_buffer=*/buffer.data_ptr<int32_t>(),
          /*ldc=*/N,
          /*outProcess=*/output_proc_obj,
          /*thread_id=*/task_id,
          /*num_threads=*/num_tasks);
    }
  });

  return output;
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
void CalcColOffsetsTranspose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t B_zero_point,
    int32_t* col_offsets) {
  for (int i = 0; i < N; ++i) {
    int32_t sum = 0;
    for (int j = 0; j < K; ++j) {
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
  const Tensor weight_contig = weight.contiguous();

  // Calculate weight statistics
  float w_min;
  float w_max;
  fbgemm::FindMinMax(
      /*m=*/weight_contig.data_ptr<float>(),
      /*min=*/&w_min,
      /*max=*/&w_max,
      /*len=*/weight_contig.numel());

  // Choose parameters for quantizing the weight as 8-bit signed integer
  constexpr bool kIsSigned = true;
  constexpr int kPrecision = 8;
  constexpr int kBound = (1 << (kPrecision - 1));
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/w_min,
      /*max=*/w_max,
      /*qmin=*/kIsSigned ? -kBound : 0,
      /*qmax=*/kIsSigned ? (kBound - 1) : (1 << kPrecision) - 1,
      /*preserve_sparsity=*/false);
  q_params.precision = kPrecision;

  Tensor quantized = at::native::empty_like(
      weight_contig, weight_contig.options().dtype(at::kChar), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // Tensor quantized = at::native::empty_cpu(
  //     weight_contig.sizes(), weight_contig.options().dtype(at::kChar));
  fbgemm::Quantize<int8_t>(
      /*src=*/weight_contig.data_ptr<float>(),
      /*dst=*/quantized.data_ptr<int8_t>(),
      /*len=*/weight_contig.numel(),
      /*qparams=*/q_params);

  // Calculate column offsets of the weight and store them away in a tensor.
  // Similarly to quantization, this can be done once and cached.
  Tensor col_offsets = at::empty(
      {weight_contig.size(0)}, weight_contig.options().dtype(at::kInt), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  CalcColOffsetsTranspose(
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
  const int64_t K = weight.size(1);
  const int64_t N = weight.size(0);
  const Tensor weight_contig = weight.contiguous();
  const int8_t* weight_ptr = weight_contig.data_ptr<int8_t>();
  auto ptr = std::make_unique<fbgemm::PackBMatrix<int8_t>>(
      /*trans=*/fbgemm::matrix_op_t::Transpose,
      /*nRow=*/K,
      /*nCol=*/N,
      /*smat=*/weight_ptr,
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

namespace {

float RawUint16ToFp16(unsigned short value) {
  // Convert raw 16 bits half precision floating point number
  // to single precision floating point number.
  const unsigned short sign_bits = value >> 15;
  const unsigned short exponent_bits = value >> 10 & 0x1f;
  const unsigned short significand_bits = value & 0x3ff;

  const float sign = sign_bits ? -1 : 1;
  const float significand =
      1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10
  const float exponent = exponent_bits - 0xf;

  return sign * std::ldexp(significand, exponent);
}

template <typename T>
bool CheckAndSaturate(T max_val, T* element) {
  if (*element > max_val) {
    *element = max_val;
    return true;
  }
  if (*element < -max_val) {
    *element = -max_val;
    return true;
  }
  return false;
}

// The range for using FP16 quantization of weights requires that the elements
// should be in the range of [5.96e-8, 65504]. If it is out of range, then the
// number will be saturated to max or min representable values by FP16.
void HandleWeightsSaturation(int64_t N, float* weight) {
  const float kFp16Max = RawUint16ToFp16(0x7BFF);
  bool found_out_of_range = false;
  for (int64_t i = 0; i < N; ++i) {
    if (CheckAndSaturate<float>(kFp16Max, weight + i)) {
      found_out_of_range = true;
    }
  }
  if (found_out_of_range) {
    TORCH_WARN("FOUND weight out of range ");
  }
}

} // namespace

Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor& weight) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

  const int64_t K = weight.size(1);
  const int64_t N = weight.size(0);
  Tensor weight_contig = weight.contiguous();
  float* weight_contig_ptr = weight_contig.data_ptr<float>();
  HandleWeightsSaturation(K * N, weight_contig_ptr);

  // TODO(mingzhe09088):
  // Consider using a functor here in PackedGemmMatrixFP16
  // Comments from (XQ): Not entirely sure this make_unique is safe. make_unique
  // is created with regular "new", and freed through TypeMetaData::deleteFn in
  // this function. This is perfectly fine if the tensors are created and freed
  // within this translation unit. It might be very problematic if that tensor
  // flows across dll boundaries.
  auto ptr = std::make_unique<fbgemm::PackedGemmMatrixFP16>(
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

  const Tensor input_contig = input.contiguous();
  const float* input_ptr = input_contig.data_ptr<float>();

  // Pull out the PackedGemmMatrixFP16 instance from the owning tensor
  const fbgemm::PackedGemmMatrixFP16& packed_weight_fp16 =
      cpp_custom_type_hack::cast<fbgemm::PackedGemmMatrixFP16>(packed_weight);

  TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
  TORCH_CHECK(input.dim() >= 2);
  TORCH_CHECK(bias.dim() == 1);

  const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  const int64_t N = packed_weight_fp16.numCols();
  std::vector<int64_t> output_size = input.sizes().vec();
  output_size.back() = N;
  Tensor output = at::empty(output_size, input.options().dtype(at::kFloat));

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

  return output;
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
