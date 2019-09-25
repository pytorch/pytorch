#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include <algorithm>
#include <string>

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearInt8 final : public torch::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor fbgemm_linear(
      at::Tensor input,
      at::Tensor packed_weight,
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
    // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
    // matrices, respectively.
    int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

    // Pull out the PackBMatrix and col_offsets instance from the owning tensor.
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeight>(packed_weight);
    auto packB = pack_ptr.w.get();
    // packB->printPackedMatrix("packedB inside fbgemm_linear (QLinearInt8): ");
    auto& col_offsets = pack_ptr.col_offsets;

    int64_t N = static_cast<int64_t>(packB->numCols());
    int64_t K = input.size(input.dim() - 1);
    TORCH_CHECK(
        K == static_cast<int64_t>(packB->numRows()),
        "The number of rows in the packB should be equal to K: " +
            std::to_string(K));

    float input_scale_float = input.q_scale();
    int32_t input_zero_point_int32 = input.q_zero_point();

    std::vector<float> output_multiplier_float(1, 0.0);
    std::vector<float> act_times_w_scale(1, 0.0);
    TORCH_CHECK(
        pack_ptr.w_scale.size() == pack_ptr.w_zp.size(),
        "Weight scales and zero points vectors should have the same size.");
    if (pack_ptr.q_scheme == kPerTensorAffine) {
      // Process the per tensor quantization.
      act_times_w_scale[0] = (input_scale_float * pack_ptr.w_scale[0]);
      output_multiplier_float[0] =
          act_times_w_scale[0] / static_cast<float>(output_scale);
    } else if (pack_ptr.q_scheme == kPerChannelAffine) {
      // Process the per channel quantization.
      output_multiplier_float.resize(N, 0.0);
      act_times_w_scale.resize(N, 1.0f);
      for (int i = 0; i < N; ++i) {
        act_times_w_scale[i] = (input_scale_float * pack_ptr.w_scale[i]);
        output_multiplier_float[i] =
            act_times_w_scale[i] / static_cast<float>(output_scale);
      }
    }
    int32_t output_zero_point_int32 = static_cast<int32_t>(output_zero_point);

    const float* bias_ptr = nullptr;
    at::Tensor bias;
    if (pack_ptr.bias.has_value()) {
      bias = pack_ptr.bias.value();
      bias = bias.contiguous();
      TORCH_CHECK(bias.dim() == 1, "bias should be a vector (1D Tensor)");
      TORCH_CHECK(
          bias.size(0) == N,
          "bias should have N elements: " + std::to_string(N));
      bias_ptr = reinterpret_cast<float*>(bias.data_ptr<float>());
    }

    // The resulting matrix here is 2-D, let's view it with the original
    // left hand dimensions of the input. Here are two examples:
    // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
    // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
    std::vector<int64_t> out_sizes = input.sizes().vec();
    out_sizes.back() = N;
    // Allocate output Tensor and a buffer for fbgemmPacked to use
    auto output = _empty_affine_quantized(
        out_sizes,
        at::device(kCPU).dtype(kQUInt8),
        output_scale,
        output_zero_point);

    auto buffer = at::zeros_like(output, output.options().dtype(at::kInt));

    int num_tasks = at::get_num_threads();
    at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
      for (int task_id = begin; task_id < end; ++task_id) {
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

        if (pack_ptr.q_scheme == kPerTensorAffine) {
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
                  pack_ptr.w_zp.data(),
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
        } else if (pack_ptr.q_scheme == kPerChannelAffine) {
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
                  pack_ptr.w_zp.data(),
                  packA.getRowOffsetBuffer(),
                  col_offsets.data(),
                  bias_ptr,
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
#endif
#ifdef USE_PYTORCH_QNNPACK
  at::Tensor qnnpack_linear(
      at::Tensor input,
      at::Tensor packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(
        input.dim() >= 2,
        "quantized::linear(): Input tensor rank should be >= 2");
    auto input_contig = input.contiguous();

    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeightsQnnp>(packed_weight);
    auto packB = pack_ptr.w.get();
    auto kernel_zp = pack_ptr.w_zp;
    auto kernel_scale = pack_ptr.w_scale;
    size_t rows_w = pack_ptr.bias.size(0);
    size_t cols_w = input_contig.size(input_contig.dim() - 1);
    auto input_scale = input_contig.q_scale();

    if (!pack_ptr.input_scale.has_value() ||
        pack_ptr.input_scale.value() != input_scale) {
      // Get the original weight and adjust it to uint8 from int8
      auto weight_contig = pack_ptr.orig_weight;
      auto bias_fp32 = pack_ptr.bias;
      int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
      Tensor qnnp_weight = at::_empty_affine_quantized(
          weight_contig.sizes(),
          at::device(kCPU).dtype(kQUInt8),
          kernel_scale,
          kernel_zp);
      auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
      for (int i = 0; i < weight_contig.numel(); ++i) {
        qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
      }
      // Original bias was float, so we requantize it here.
      auto bias = at::quantize_per_tensor(
          bias_fp32, kernel_scale * input_scale, 0, kQInt32);
      // Update the input scale to not pack again.
      pack_ptr.input_scale = input_scale;
      pack_ptr.w.reset();
      pack_ptr.w = guts::make_unique<qnnpack::PackBMatrix>(
          cols_w /* input_channels */,
          rows_w /* output_channels */,
          kernel_zp,
          kernel_scale,
          (uint8_t*)qnnp_w_data,
          (int32_t*)bias.data_ptr<c10::qint32>());
      packB = pack_ptr.w.get();
    }

    size_t rows_input = 1;
    size_t cols_input = input_contig.size(input_contig.dim() - 1);
    for (size_t i = 0; i < input_contig.dim() - 1; ++i) {
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
    Tensor output = at::_empty_affine_quantized(
        {static_cast<long>(rows_input), static_cast<long>(rows_w)},
        input.options(),
        output_scale,
        output_zero_point);

    auto output_min = ReluFused
        ? activationLimits(output_scale, output_zero_point, Activation::RELU)
              .first
        : std::numeric_limits<uint8_t>::min();
    auto output_max = ReluFused
        ? activationLimits(output_scale, output_zero_point, Activation::RELU)
              .second
        : std::numeric_limits<uint8_t>::max();
    TORCH_INTERNAL_ASSERT(packB != nullptr, "Packed Weights are NULL");
    const pytorch_qnnp_status runStatus = qnnpack::qnnpackLinear(
        rows_input /* batch_size */,
        cols_input /* input_channels */,
        rows_w /* output_channels */,
        input_contig.q_zero_point(),
        input_contig.q_scale(),
        kernel_zp,
        kernel_scale,
        output_zero_point,
        output_scale,
        output_min,
        output_max,
        (uint8_t*)input_contig.data_ptr<c10::quint8>(),
        cols_input /* input_stride */,
        packB->getPackedWeights(),
        (uint8_t*)output.data_ptr<c10::quint8>(),
        rows_w /* output_stride */,
        caffe2::mobile_pthreadpool() /* threadpool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Linear operator");

    return output;
  }
#endif
  at::Tensor operator()(
      at::Tensor input,
      at::Tensor packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear(
          input, packed_weight, output_scale, output_zero_point);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear(
          input, packed_weight, output_scale, output_zero_point);
    }
#endif
    TORCH_INTERNAL_ASSERT(
        "Didn't find engine for operation quantized::linear ",
        toString(ctx.qEngine()));
    return at::Tensor();
  }
};

static auto registry =
    torch::RegisterOperators()
        .op("quantized::linear(Tensor X, Tensor W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y",
            torch::RegisterOperators::options().kernel<QLinearInt8<false>>(
                TensorTypeId::QuantizedCPUTensorId))
        .op("quantized::linear_relu(Tensor X, Tensor W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y",
            torch::RegisterOperators::options().kernel<QLinearInt8<true>>(
                TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
