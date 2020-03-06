#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include <algorithm>
#include <string>

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearDynamicInt8 final : public torch::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor fbgemm_linear(at::Tensor input, at::Tensor packed_weight) {
    // fp32 * int8 -> fp32 (with quantization on activation, and dequantization
    // on the result).

    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

    // TODO: contiguous is called for further jit optimizations.
    auto input_contig = input.contiguous();
    const auto* input_ptr = input_contig.data_ptr<float>();

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
    // packB->printPackedMatrix("packedB inside fbgemm_linear_dynamic
    // (QLinearDynamicInt8): ");
    auto& col_offsets = pack_ptr.col_offsets;

    int64_t N = static_cast<int64_t>(packB->numCols());
    int64_t K = input.size(input.dim() - 1);
    TORCH_CHECK(
        K == static_cast<int64_t>(packB->numRows()),
        "The number of rows in the packB should be equal to K: " +
            std::to_string(K));

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
    auto q_params = quant_utils::ChooseQuantizationParams(
        /*min=*/x_min,
        /*max=*/x_max,
        /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
        /*qmax=*/
        is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
        /*preserve_sparsity=*/false);

    q_params.precision = precision;

    // ReQuantizeForFloat requires pointers to the zero point values,
    // since in the case of rowwise quantization these will be arrays rather
    // than scalars. But in this case, we're doing whole-tensor quantization so
    // we just pass a pointer to the scale values (and internally
    // ReQuantizeForFloat won't index past 0.

    const float* bias_ptr = nullptr;
    at::Tensor bias_vec;
    if (pack_ptr.bias.has_value()) {
      bias_vec = pack_ptr.bias.value();
      TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
      TORCH_CHECK(
          bias_vec.size(0) == N,
          "bias should have N elements: " + std::to_string(N));
      // TODO: contiguous is called for further jit optimizations.
      auto bias_contig = bias_vec.contiguous();
      bias_ptr = bias_contig.data_ptr<float>();
    }
    // The resulting matrix here is 2-D, let's view it with the original
    // left hand dimensions of the input. Here are two examples:
    // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
    // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
    std::vector<int64_t> out_sizes = input.sizes().vec();
    out_sizes.back() = N;
    // Allocate output Tensor and a buffer for fbgemmPacked to use
    auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
    auto buffer = at::empty_like(
        output,
        output.options().dtype(at::kInt),
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    int num_tasks = at::get_num_threads();
    at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
      // This operation does the following:
      // 1) Quantizes the input matrix given the statistics we've calculated
      // above
      // 2) Creates a "row buffer" vector with offset values that must be
      // added
      //    to the integer matrix multiplication operation to ensure
      //    correctness. This "row buffer" is also called the row offset, and it
      //    is needed when we use affine quantization for weights.
      // 3) Packs the resulting quantized matrix into vector-register and cache
      //    friendly tiles.
      //
      //  Note this is not executed eagerly, but rather within the fbgemmPacked
      //  call below.

      fbgemm::PackAWithQuantRowOffset<uint8_t> packA(
          /*trans=*/fbgemm::matrix_op_t::NoTranspose,
          /*nRow=*/M,
          /*nCol=*/K,
          /*smat=*/input_ptr,
          /*ld=*/K,
          /*pmat=*/nullptr, // Currently, packA manages ownership of `pmat`.
          /*scale=*/q_params.scale,
          /*zero_pt=*/q_params.zero_point);
      // TODO: Consider a way to pre-allocate and reuse
      // pmat buffer.

      // This is the end of the pipeline, pass the resulting matrix through.
      fbgemm::DoNothing<float, float> doNothingObj{};

      for (int task_id = begin; task_id < end; ++task_id) {
        if (pack_ptr.q_scheme == kPerTensorAffine) {
          // Process the per tensor quantization.
          //
          // After the uint8 * int8 matrix multiplication is performed, this
          // operation does:
          //  1) Add in row and column offsets to the rows and columns,
          //  respectively.
          //  2) Dequantize the results into floating point.
          //  3) Add in the bias term.
          fbgemm::ReQuantizeForFloat<ReluFused> outputProcObj(
              /*nextop=*/doNothingObj,
              /*Aq_scale=*/q_params.scale,
              /*Bq_scale=*/pack_ptr.w_scale.data(),
              /*Aq_zero_point=*/q_params.zero_point,
              /*Bq_zero_point=*/pack_ptr.w_zp.data(),
              /*row_offsets=*/packA.getRowOffsetBuffer(),
              /*col_offsets=*/col_offsets.data(),
              /*bias=*/bias_ptr,
              /*nCol=*/N);

          // Do the GEMM
          fbgemm::fbgemmPacked(
              /*packA=*/packA,
              /*packB=*/*packB,
              /*C=*/output.data_ptr<float>(),
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
          //  2) Dequantize the results into floating point.
          //  3) Add in the bias term.
          fbgemm::ReQuantizeForFloat<
              ReluFused,
              fbgemm::QuantizationGranularity::OUT_CHANNEL>
              outputProcObj(
                  /*nextop=*/doNothingObj,
                  /*Aq_scale=*/q_params.scale,
                  /*Bq_scale=*/pack_ptr.w_scale.data(),
                  /*Aq_zero_point=*/q_params.zero_point,
                  /*Bq_zero_point=*/pack_ptr.w_zp.data(),
                  /*row_offsets=*/packA.getRowOffsetBuffer(),
                  /*col_offsets=*/col_offsets.data(),
                  /*bias=*/bias_ptr,
                  /*nCol=*/N);

          // Do the GEMM
          fbgemm::fbgemmPacked(
              /*packA=*/packA,
              /*packB=*/*packB,
              /*C=*/output.data_ptr<float>(),
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
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK

  at::Tensor qnnpack_linear(at::Tensor input, at::Tensor packed_weight) {
    TORCH_CHECK(
        input.dim() >= 2,
        "The dimension of input tensor should be larger than or equal to 2");
    auto input_contig = input.contiguous();
    // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
    // matrices, respectively.

    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeightsQnnp>(packed_weight);
    auto packB = pack_ptr.w.get();
    // Adjust weight zero point, similar to weight data.
    auto kernel_zp = pack_ptr.w_zp + 128;
    auto kernel_scale = pack_ptr.w_scale;
    size_t rows_w = pack_ptr.bias.size(0);
    size_t cols_w = input_contig.size(input_contig.dim() - 1);

    at::Tensor bias_vec = pack_ptr.bias;

    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

    auto bias_contig = bias_vec.contiguous();
    const float* bias_ptr = bias_contig.data_ptr<float>();

    // Calculate statistics for quantization of input Tensor
    // TODO: optimized kernel
    float x_min = input_contig.min().item<float>();
    float x_max = input_contig.max().item<float>();

    auto q_params = quant_utils::ChooseQuantizationParams(
        /*min=*/x_min,
        /*max=*/x_max,
        /*qmin=*/0,
        /*qmax=*/255);
    if (!pack_ptr.input_scale.has_value()) {
      // Get the original weight and adjust it to uint8 from int8
      auto weight_contig = pack_ptr.orig_weight;
      int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
      Tensor qnnp_weight = at::_empty_affine_quantized(
          weight_contig.sizes(),
          at::device(kCPU).dtype(kQUInt8),
          kernel_scale,
          kernel_zp);
      auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
      auto wt_numel = weight_contig.numel();
      for (int i = 0; i < wt_numel; ++i) {
        qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
      }

      // Update the input scale to not pack again.
      // Pass in nullptr for bias, as we pass FP32 bias to run function.
      pack_ptr.input_scale = q_params.scale;
      pack_ptr.w.reset();
      pack_ptr.w = std::make_unique<qnnpack::PackBMatrix>(
          cols_w /* input_channels */,
          rows_w /* output_channels */,
          kernel_zp,
          kernel_scale,
          (uint8_t*)qnnp_w_data,
          nullptr);
      packB = pack_ptr.w.get();
    }

    // Quantize input
    Tensor q_input = at::quantize_per_tensor(
        input_contig, q_params.scale, q_params.zero_point, kQUInt8);

    // The resulting matrix here is 2-D, let's view it with the original
    // left hand dimensions of the input. Here are two examples:
    // 1. If the input tensor is {M, K}, the output tensor is {M, N}.
    // 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
    std::vector<int64_t> out_sizes = input.sizes().vec();
    out_sizes.back() = rows_w;

    auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));

    size_t rows_input = 1;
    size_t cols_input = input_contig.size(input_contig.dim() - 1);
    for (size_t i = 0; i < input_contig.dim() - 1; ++i) {
      rows_input *= input_contig.size(i);
    }
    pytorch_qnnp_status runStatus = qnnpack::qnnpackLinearDynamic(
        rows_input /* batch_size */,
        cols_input /* input_channels */,
        rows_w /* output_channels */,
        q_input.q_zero_point(),
        q_input.q_scale(),
        kernel_zp,
        kernel_scale,
        (uint8_t*)q_input.data_ptr<c10::quint8>(),
        cols_input /* input_stride */,
        packB->getPackedWeights(),
        bias_ptr,
        output.data_ptr<float>(),
        rows_w /* output_stride */,
        caffe2::mobile_pthreadpool() /* threadpool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == pytorch_qnnp_status_success,
        "failed to run QNNPACK Linear operator");
    return output;
  }
#endif // USE_PYTORCH_QNNPACK
  at::Tensor operator()(at::Tensor input, at::Tensor packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear(input, packed_weight);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear(input, packed_weight);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear ",
        toString(ctx.qEngine()));
  }
};

template <bool ReluFused>
class QLinearDynamicFp16 final : public torch::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor operator()(at::Tensor input, at::Tensor packed_weight) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

    const Tensor input_contig = input.contiguous();
    const float* input_ptr = input_contig.data_ptr<float>();

    // Pull out the PackedGemmMatrixFP16 instance from the owning tensor
    auto& packed_param_struct =
        cpp_custom_type_hack::cast<PackedLinearWeightFp16>(packed_weight);
    auto& packed_weight_fp16 = *packed_param_struct.w;
    auto& bias = packed_param_struct.bias;

    TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
    TORCH_CHECK(input.dim() >= 2);

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
    if (bias.has_value()) {
      TORCH_CHECK(bias->dim() == 1);
      output.add_(*bias);
    }

    return output;
  }
#else // USE_FBGEMM
  at::Tensor operator()(
      at::Tensor /* input */,
      at::Tensor /* packed_weight */) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry =
    torch::RegisterOperators()
        .op("quantized::linear_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .kernel<QLinearDynamicInt8<false>>(DispatchKey::CPUTensorId))
        .op("_quantized::linear_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .kernel<QLinearDynamicInt8<false>>(DispatchKey::CPUTensorId))
        .op("quantized::linear_relu_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .kernel<QLinearDynamicInt8<true>>(DispatchKey::CPUTensorId))
        .op("quantized::linear_dynamic_fp16(Tensor X, Tensor W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .kernel<QLinearDynamicFp16<false>>(DispatchKey::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
