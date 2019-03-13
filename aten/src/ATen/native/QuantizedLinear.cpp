#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtilsMulti.h"

#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
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
namespace at {
namespace native {

#ifdef USE_FBGEMM

Tensor fbgemm_linear_int8_weight(
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
  AT_ASSERTM(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  auto input_contig = input.contiguous();
  auto* input_ptr = input_contig.data<float>();

  AT_ASSERT(input.dim() >= 2);
  int64_t M = 1;
  for (size_t i = 0; i < input.dim() - 1; ++i) {
    M *= input.size(i);
  }
  int64_t K = input.size(input.dim() - 1);
  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(K == weight.size(1));
  auto N = weight.size(0);
  AT_ASSERT(bias.dim() == 1);
  AT_ASSERT(bias.size(0) == N);
  AT_ASSERT(weight_scale.isFloatingPoint());
  AT_ASSERT(weight_zero_point.isIntegral());

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

  // Calculate scale and zero point for quantization of input tensor
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
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
  fbgemm::ReQuantizeForFloat<false /* FUSE_RELU*/> outputProcObj(
      /*nextop=*/doNothingObj,
      /*Aq_scale=*/q_params.scale,
      /*Bq_scale=*/&weight_scale_float,
      /*Aq_zero_point=*/q_params.zero_point,
      /*Bq_zero_point=*/&weight_zero_point_int32,
      /*row_offsets=*/packA.getRowOffsetBuffer(),
      /*col_offsets=*/col_offsets.data<int32_t>(),
      /*bias=*/bias_contig.data<float>(),
      /*ncol=*/N);

  // Allocate output Tensor and a buffer for fbgemmPacked to use
  auto output = at::zeros({M, N}, bias.options().dtype(at::kFloat));
  auto buffer = at::zeros_like(output, output.options().dtype(at::kInt));

  // Pull out the PackBMatrix instance from the owning tensor
  auto* packB = reinterpret_cast<fbgemm::PackBMatrix<int8_t>*>(
      packed.storage().data_ptr().get());

  // Do the GEMM
  fbgemm::fbgemmPacked(
      /*packA=*/packA,
      /*packB=*/*packB,
      /*C=*/output.data<float>(),
      /*C_buffer=*/buffer.data<int32_t>(),
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

namespace {
// Calculate the column offsets
// Note this includes the sum of the columns as well as the scalar term
// B_zero_point * K, whereas the row_offsets created by PackAWithQuantRowOffset
// is only the sum of the A rows.
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
  AT_ASSERTM(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  auto weight_contig = weight.contiguous();

  // Calculate weight statistics
  float w_min, w_max;
  fbgemm::FindMinMax(
      /*m=*/weight_contig.data<float>(),
      /*min=*/&w_min,
      /*max=*/&w_max,
      /*len=*/weight_contig.numel());

  // Choose parameters for quantizing the weight as 8-bit signed integer
  static constexpr bool is_signed = true;
  static constexpr int precision = 8;
  auto q_params = fbgemm::ChooseQuantizationParams(
      /*min=*/w_min,
      /*max=*/w_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false);

  q_params.precision = precision;

  auto quantized = at::zeros_like(weight_contig).to(at::kChar).contiguous();
  fbgemm::Quantize<int8_t>(
      /*src=*/weight_contig.data<float>(),
      /*dst=*/quantized.data<int8_t>(),
      /*len=*/weight_contig.numel(),
      /*qparams=*/q_params);

  // Calculate column offsets of the weight and store them away in a tensor.
  // Similarly to quantization, this can be done once and cached.
  auto col_offsets =
      at::zeros_like(quantized).sum({1}).to(at::kInt).contiguous();
  calc_col_offsets_transpose(
      /*K=*/quantized.size(1),
      /*N=*/quantized.size(0),
      /*Bint8=*/quantized.data<int8_t>(),
      /*B_zero_point=*/q_params.zero_point,
      /*col_offsets=*/col_offsets.data<int32_t>());

  return std::make_tuple(
      quantized, col_offsets, q_params.scale, q_params.zero_point);
}

bool fbgemm_is_cpu_supported() {
  return fbgemm::fbgemmSupportedCPU();
}

Tensor fbgemm_pack_quantized_matrix(
    const Tensor& weight,
    int64_t K,
    int64_t N) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  AT_ASSERTM(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  auto weight_contig = weight.contiguous();
  auto contiguous_ptr = weight_contig.data<int8_t>();
  auto* ptr = new fbgemm::PackBMatrix<int8_t>(
      /*trans=*/fbgemm::matrix_op_t::Transpose,
      /*nRow=*/K,
      /*nCol=*/N,
      /*smat=*/contiguous_ptr,
      /*ld=*/K,
      /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
      /*groups=*/1);

  // We store this instance away in a Tensor and register a deleter function
  // so that we do not leak memory. On the other side, we pull out the storage's
  // data_ptr and get the PackBMatrix's pointer.
  at::DataPtr at_ptr(
      ptr,
      ptr,
      [](void* ptr) {
        fbgemm::PackBMatrix<int8_t>* typed_ptr =
            reinterpret_cast<fbgemm::PackBMatrix<int8_t>*>(ptr);
        delete typed_ptr;
      },
      at::kCPU);

  auto retval = at::empty(
      {sizeof(fbgemm::PackBMatrix<int8_t>)}, weight.options().dtype(at::kByte));

  retval.storage().set_data_ptr(std::move(at_ptr));

  return retval;
}

#else // USE_FBGEMM

Tensor fbgemm_linear_int8_weight(
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
  AT_ASSERTM(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

std::tuple<Tensor, Tensor, double, int64_t> fbgemm_linear_quantize_weight(
    const Tensor& /*weight*/) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  AT_ASSERTM(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

Tensor fbgemm_pack_quantized_matrix(
    const Tensor& /*input*/,
    int64_t /*K*/,
    int64_t /*N*/) {
  // We make a strong guarantee that models using these operators will have the
  // same numerics across different machines. Therefore, we do not provide a
  // fallback path and rather fail loudly if we cannot run FBGEMM.
  AT_ASSERTM(
      false, "This PyTorch installation was not built with FBGEMM operators");
}

bool fbgemm_is_cpu_supported() {
  return false;
}

#endif // USE_FBGEMM
}
} // namespace at
