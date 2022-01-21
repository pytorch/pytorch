#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>         // for _empty_affine_q...
#include <ATen/ops/empty.h>                           // for empty
#include <ATen/ops/quantize_per_channel_native.h>     // for quantize_per_ch...
#include <ATen/ops/quantize_per_tensor_native.h>      // for quantize_per_te...
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <string>

int register_linear_params();

#ifdef USE_FBGEMM
template <bool ReluFused>
at::Tensor& PackedLinearWeight::apply_impl(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    at::Tensor& output) {
  // uint8 * int8 -> uint8 (no quantization/dequantization)

  // We make a strong guarantee that models using these operators will have
  // the same numerics across different machines. Therefore, we do not provide
  // a fallback path and rather fail loudly if we cannot run FBGEMM.
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
                "Expected input data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(input.scalar_type()));

  // TODO: contiguous is called for further jit optimizations.
  auto input_contig = input.expect_contiguous();
  const auto* input_ptr =
      reinterpret_cast<uint8_t*>(input_contig->data_ptr<c10::quint8>());

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
  // matrices, respectively.
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  auto packB = w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = input.sizes()[input.dim() - 1];
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
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
    output_multiplier_float.resize(N, 0.0);
    act_times_w_scale.resize(N, 1.0f);
    for (const auto i : c10::irange(N)) {
      act_times_w_scale[i] = (input_scale_float * w_scale[i]);
      output_multiplier_float[i] =
          act_times_w_scale[i] / static_cast<float>(output_scale);
    }
  }
  int32_t output_zero_point_int32 = static_cast<int32_t>(output_zero_point);

  const float* bias_ptr = nullptr;
  c10::MaybeOwned<at::Tensor> bias_contig;
  if (this->bias_.has_value()) {
    auto& bias = this->bias_.value();
    bias_contig = bias.expect_contiguous();
    TORCH_CHECK(bias_contig->dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_contig->sizes()[0] == N, "bias should have N elements: " + std::to_string(N));
    bias_ptr = reinterpret_cast<float*>(bias_contig->data_ptr<float>());
  }

  // The resulting matrix here is 2-D, let's view it with the original
  // left hand dimensions of the input. Here are two examples:
  // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
  // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
  at::DimVector out_sizes(input.sizes());
  out_sizes.back() = N;
  // Resize output Tensor
  output.resize_(out_sizes);

  // Allocate a buffer for fbgemmPacked to use
  auto buffer = at::empty(out_sizes, output.options().dtype(at::kInt));

  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    for (const auto task_id : c10::irange(begin, end)) {
      // This operation does the following:
      // 1) Creates a "row buffer" vector with offset values that must be
      //    added to the integer matrix multiplication operation to ensure
      //    correctness. This "row buffer" is also called the row offset, and
      //    it is needed when we use affine quantization for weights.
      // 2) Packs the resulting quantized matrix into vector-register and
      //    cache friendly tiles.
      //
      //  Note this is not executed eagerly, but rather within the
      //  fbgemmPacked call below.
      fbgemm::PackAWithRowOffset<uint8_t> packA(
          /*trans=*/fbgemm::matrix_op_t::NoTranspose,
          /*nRow=*/M,
          /*nCol=*/K,
          /*smat=*/input_ptr,
          /*ld=*/K,
          /*pmat=*/nullptr); // Currently, packA manages ownership of `pmat`.
                             // TODO: Consider a way to pre-allocate and reuse
                             // pmat buffer.

      // ReQuantizeOutput requires pointers to the zero point values,
      // since in the case of rowwise quantization these will be arrays rather
      // than scalars. But in this case, we're doing whole-tensor quantization
      // so we just pass a pointer to the scale values (and internally
      // ReQuantizeOutput won't index past 0.

      // This is the end of the pipeline, pass the resulting matrix through.
      fbgemm::DoNothing<> doNothingObj{};

      if (q_scheme == c10::kPerTensorAffine) {
        // Process the per tensor quantization.
        //
        // After the uint8 * int8 matrix multiplication is performed, this
        // operation does:
        //  1) Add in row and column offsets to the rows and columns,
        //  respectively.
        //  2) Add in the bias term.
        fbgemm::ReQuantizeOutput<
            ReluFused,
            fbgemm::QuantizationGranularity::TENSOR,
            float>
            outputProcObj(
                doNothingObj,
                output_multiplier_float.data(),
                output_zero_point_int32,
                input_zero_point_int32,
                w_zp.data(),
                packA.getRowOffsetBuffer(),
                col_offsets.data(),
                bias_ptr,
                N, /* nCol */
                1 /* groups */,
                act_times_w_scale.data());

        // Do the GEMM
        fbgemm::fbgemmPacked(
            /*packA=*/packA,
            /*packB=*/*packB,
            /*C=*/reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
            /*C_buffer=*/buffer.data_ptr<int32_t>(),
            /*ldc=*/N,
            /*outProcess=*/outputProcObj,
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
        fbgemm::ReQuantizeOutput<
            ReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL,
            float>
            outputProcObj(
                doNothingObj,
                output_multiplier_float.data(),
                output_zero_point_int32,
                input_zero_point_int32,
                w_zp.data(),
                packA.getRowOffsetBuffer(),
                col_offsets.data(),
                bias_ptr,
                // NOLINTNEXTLINE(bugprone-argument-comment)
                N, /*nCol=*/
                1, /* groups*/
                act_times_w_scale.data());

        // Do the GEMM
        fbgemm::fbgemmPacked(
            /*packA=*/packA,
            /*packB=*/*packB,
            /*C=*/reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
            /*C_buffer=*/buffer.data_ptr<int32_t>(),
            /*ldc=*/N,
            /*outProcess=*/outputProcObj,
            /*thread_id=*/task_id,
            /*num_threads=*/num_tasks);
      }
    }
  });

  return output;
}

at::Tensor PackedLinearWeight::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  // Allocate output Tensor
  auto output = at::_empty_affine_quantized(
      {0},
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  apply_impl<false>(input, output_scale, output_zero_point, output);
  return output;
}

at::Tensor PackedLinearWeight::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  auto output = at::_empty_affine_quantized(
      {0},
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  apply_impl<true>(input, output_scale, output_zero_point, output);
  return output;
}

at::Tensor& PackedLinearWeight::apply_out(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    at::Tensor& output) {
  TORCH_CHECK(
      (output.device() == c10::kCPU) && (output.dtype() == c10::kQUInt8) &&
      (output.q_scale() == output_scale) &&
      (output.q_zero_point() == output_zero_point));
  return apply_impl<false>(input, output_scale, output_zero_point, output);
}

at::Tensor& PackedLinearWeight::apply_relu_out(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point,
    at::Tensor& output) {
  TORCH_CHECK(
      (output.device() == c10::kCPU) && (output.dtype() == c10::kQUInt8) &&
      (output.q_scale() == output_scale) &&
      (output.q_zero_point() == output_zero_point));
  return apply_impl<true>(input, output_scale, output_zero_point, output);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <bool ReluFused>
at::Tensor PackedLinearWeightsQnnp::apply_impl(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  TORCH_CHECK(
      input.dim() >= 2,
      "quantized::linear(): Input tensor rank should be >= 2");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
                "quantized::linear (qnnpack): Expected input data type ",
                toString(c10::kQUInt8),
                " but got ",
                toString(input.scalar_type()));

  auto input_contig = input.contiguous();

  // Weight packing is not thread safe
  std::lock_guard<std::mutex> lock(qnnp_mutex_);
  auto packB = w.get();
  size_t rows_w = bias_.size(0);
  size_t cols_w = input_contig.size(input_contig.dim() - 1);
  auto input_scale = input_contig.q_scale();

  if (!this->input_scale.has_value() ||
      this->input_scale.value() != input_scale) {
    // Get the original weight and adjust it to uint8 from int8
    auto weight_contig = orig_weight;
    auto bias_fp32 = bias_;
    int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();

    float* weight_scales_data = w_scales.data_ptr<float>();
    // We calculate requant scale here as the vector holding the requant scale
    // is owned by this module. The pointer is then passed to qnnpack backend.
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales, input_scale, output_scale, requantization_scales);

    at::Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8),
        weight_scales_data[0],
        w_zero_points[0]);
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    auto wt_numel = weight_contig.numel();
    for (const auto i : c10::irange(wt_numel)) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }
    // Original bias was float, so we requantize it here.
    const bool is_per_channel = orig_weight.qscheme() == at::kPerChannelAffine;
    at::Tensor qbias;
    // Original bias was float, so we requantize it here.
    if (is_per_channel) {
      at::Tensor bias_quant_scales =
          weight_contig.q_per_channel_scales() * input_scale;
      at::Tensor bias_zp = at::zeros(bias_quant_scales.sizes(), c10::kInt);
      qbias = at::native::quantize_per_channel(
          bias_fp32, bias_quant_scales, bias_zp, 0, c10::kQInt32);
    } else {
      qbias = at::native::quantize_per_tensor(
          bias_fp32, weight_contig.q_scale() * input_scale, 0, c10::kQInt32);
    }

    // Update the input scale to not pack again.
    this->input_scale = input_scale;
    w.reset();
    w = std::make_unique<qnnpack::PackBMatrix>(
        cols_w /* input_channels */,
        rows_w /* output_channels */,
        w_zero_points.data(),
        requantization_scales.data(),
        reinterpret_cast<uint8_t*>(qnnp_w_data),
        reinterpret_cast<int32_t*>(qbias.data_ptr<c10::qint32>()));
    packB = w.get();
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
      // On mobile, we release the original weight by resetting the intrusive_ptr.
      // Calling unpack after this will throw an assertion.
      orig_weight.reset();
    }
  }

  size_t rows_input = 1;
  size_t cols_input = input_contig.size(input_contig.dim() - 1);
  for (const auto i : c10::irange(input_contig.dim() -1)) {
    rows_input *= input_contig.size(i);
  }

  TORCH_CHECK(
      cols_input == cols_w,
      "quantized::linear(): input size does not match weight dimension 1 size: \
         got ",
      cols_input,
      " but expected ",
      cols_w);

  // Allocate output Tensor and a buffer for QNNPACK to use
  // The resulting matrix here is 2-D, let's view it with the original
  // left hand dimensions of the input. Here are two examples:
  // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
  // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = static_cast<long>(rows_w);
  at::Tensor output = at::_empty_affine_quantized(
      out_sizes,
      input.options(),
      output_scale,
      output_zero_point);

  auto output_min = ReluFused
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU)
            .first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = ReluFused
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      ? activationLimits<uint8_t>(output_scale, output_zero_point, Activation::RELU)
            .second
      : std::numeric_limits<uint8_t>::max();
  TORCH_INTERNAL_ASSERT(packB != nullptr, "Packed Weights are NULL");
  const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinear(
      rows_input /* batch_size */,
      cols_input /* input_channels */,
      rows_w /* output_channels */,
      input_contig.q_zero_point(),
      w_zero_points.data(),
      requantization_scales.data(),
      output_zero_point,
      output_min,
      output_max,
      (uint8_t*)input_contig.data_ptr<c10::quint8>(),
      cols_input /* input_stride */,
      packB->getPackedWeights(),
      (uint8_t*)output.data_ptr<c10::quint8>(),
      rows_w /* output_stride */,
      // TODO (Ashkan): Disabling temporarily.
      // Throws a floating point exception with OSS pthreadpool.
      caffe2::pthreadpool_() /* threadpool */);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Linear operator");

  return output;
}

at::Tensor PackedLinearWeightsQnnp::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(std::move(input), output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightsQnnp::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(std::move(input), output_scale, output_zero_point);
}

#endif // USE_PYTORCH_QNNPACK

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    if (ReluFused) {
      return packed_weight->apply_relu(
          std::move(input), output_scale, output_zero_point);
    } else {
      return packed_weight->apply(
          std::move(input), output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear"), TORCH_FN(QLinearInt8<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu"), TORCH_FN(QLinearInt8<true>::run));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear"), TORCH_FN(QLinearInt8<false>::run));
}

} // namespace
} // namespace native
} // namespace at
