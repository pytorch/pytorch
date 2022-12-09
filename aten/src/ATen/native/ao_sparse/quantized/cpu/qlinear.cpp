#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#endif

namespace ao {
namespace sparse {

int register_linear_params();

#ifdef USE_FBGEMM

template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_impl(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  // uint8 * int8 -> uint8 (no quantization/dequantization)

  // We make a strong guarantee that models using these operators will have
  // the same numerics across different machines. Therefore, we do not provide
  // a fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  // TODO: contiguous is called for further jit optimizations.
  auto input_contig = input.contiguous();
  const auto* input_ptr =
      reinterpret_cast<uint8_t*>(input_contig.data_ptr<c10::quint8>());

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  int64_t batch_size = size_to_dim_(input.dim() - 1, input.sizes());

  auto packW = w.get();

  int64_t out_channels = static_cast<int64_t>(packW->R);
  int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(
      K == static_cast<int64_t>(packW->C),
      "The number of columns in the packW should be equal to K: " +
          std::to_string(K));

  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  float input_scale_float = input.q_scale();
  int32_t input_zero_point_int32 = input.q_zero_point();

  std::vector<float> output_multiplier_float(1, 0.0);
  std::vector<float> act_times_w_scale(1, 0.0);
  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");
  if (q_scheme == c10::kPerTensorAffine) {
    // Process the per tensor quantization.
    act_times_w_scale[0] = (input_scale_float * w_scale[0]);
    output_multiplier_float[0] =
        act_times_w_scale[0] / static_cast<float>(output_scale);
  } else if (q_scheme == c10::kPerChannelAffine) {
    // Process the per channel quantization.
    output_multiplier_float.resize(out_channels, 0.0);
    act_times_w_scale.resize(out_channels, 1.0f);
    for (const auto i : c10::irange(out_channels)) {
      act_times_w_scale[i] = (input_scale_float * w_scale[i]);
      output_multiplier_float[i] =
          act_times_w_scale[i] / static_cast<float>(output_scale);
    }
  }
  int32_t output_zero_point_int32 = static_cast<int32_t>(output_zero_point);

  const float* bias_ptr = nullptr;
  at::Tensor bias;
  if (this->bias_.has_value()) {
    bias = this->bias_.value();
    bias = bias.contiguous();
    TORCH_CHECK(bias.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.size(0) == out_channels,
        "bias should have out_channels elements: " +
            std::to_string(out_channels));
    bias_ptr = reinterpret_cast<float*>(bias.data_ptr<float>());
  }

  // The resulting matrix here is 2-D, let's view it with the original
  // left hand dimensions of the input. Here are two examples:
  // 1. If the input tensor is {batch_size, K}, the output tensor is
  // {batch_size, out_channels}.
  // 2. If the input tensor is {x, batch_size, K}, the output tensor is {x,
  // batch_size, out_channels}.
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = out_channels; // NOLINT
  // Allocate output Tensor and a buffer for fbgemmPacked to use
  auto output_tr = at::_empty_affine_quantized(
      out_sizes,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  auto output = at::_empty_affine_quantized(
      out_sizes,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);

  auto buffer = at::empty(out_sizes, output.options().dtype(at::kInt));

  // fbgemm kernel computes the following:
  // C(output) = A(weight) x B(input), where C, A, B are out_channels x
  // batch_size, out_channels x K, K x batch_size matrices, respectively.
  // Therefore we need to transpose input
  auto input_tr = at::_empty_affine_quantized(
      input.sizes(),
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      input_scale_float,
      input_zero_point_int32);

  auto* input_tr_ptr =
      reinterpret_cast<uint8_t*>(input_tr.data_ptr<c10::quint8>());
  // TODO: Activation transpose before and after the kernel can be removed if we
  // keep activation tensor always tranposed.
  fbgemm::transpose_simd<uint8_t>(
      batch_size, K, input_ptr, K, input_tr_ptr, batch_size);

  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    for (const auto task_id : c10::irange(begin, end)) {
      fbgemm::trRequantizationParams_t reqParams = {
          input_zero_point_int32,
          w_zp.data(),
          output_zero_point_int32,
          static_cast<float>(output_scale),
          col_offsets.data(),
          /*activation offsets*/ nullptr,
          bias_ptr,
          act_times_w_scale.data()};

      if (q_scheme == c10::kPerTensorAffine) {
        // Process the per tensor quantization.
        //
        // After the uint8 * int8 matrix multiplication is performed, this
        // operation does:
        //  1) Add in row and column offsets to the rows and columns,
        //  respectively.
        //  2) Add in the bias term.

        // Do the GEMM
        fbgemm::fbgemmSparseDenseInt8MM<
            ReluFused,
            fbgemm::QuantizationGranularity::TENSOR>(
            batch_size,
            w,
            input_tr_ptr,
            /*ldb=*/batch_size,
            /*C_i32=*/buffer.data_ptr<int32_t>(),
            /*C_u8=*/
            reinterpret_cast<uint8_t*>(output_tr.data_ptr<c10::quint8>()),
            /*ldc=*/batch_size,
            /*rParams=*/reqParams,
            /*accum=*/false,
            /*thread_id=*/task_id,
            /*num_threads=*/num_tasks);
      } else if (q_scheme == c10::kPerChannelAffine) {
        // Process the per channel quantization.
        //
        // After the uint8 * int8 matrix multiplication is performed, this
        // operation does:
        //  1) Add in row and column offsets to the rows and columns,
        //  respectively.
        //  2) Add in the bias term.

        // Do the GEMM
        fbgemm::fbgemmSparseDenseInt8MM<
            ReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL>(
            batch_size,
            w,
            input_tr_ptr,
            /*ldb=*/batch_size,
            /*C_i32=*/buffer.data_ptr<int32_t>(),
            /*C_u8=*/
            reinterpret_cast<uint8_t*>(output_tr.data_ptr<c10::quint8>()),
            /*ldc=*/batch_size,
            /*rParams=*/reqParams,
            /*accum*/ false,
            /*thread_id=*/task_id,
            /*num_threads=*/num_tasks);
      }
    }
  });

  // transpose output_tr back to batch_size x out_channels
  fbgemm::transpose_simd<uint8_t>(
      out_channels,
      batch_size,
      reinterpret_cast<uint8_t*>(output_tr.data_ptr<c10::quint8>()),
      batch_size,
      reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
      out_channels);

  return output;
}

at::Tensor PackedLinearWeight::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point);
}

at::Tensor PackedLinearWeight::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point);
}

#endif // USE_FBGEMM

namespace {

template <bool ReluFused>
class QLinearInt8 final {
 public:
  static at::Tensor run(
      const at::Tensor& input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    if (ReluFused) {
      return packed_weight->apply_relu(input, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(input, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear"),
      TORCH_FN(QLinearInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_relu"),
      TORCH_FN(QLinearInt8<true>::run));
}

} // namespace
}} // namespace ao::sparse
