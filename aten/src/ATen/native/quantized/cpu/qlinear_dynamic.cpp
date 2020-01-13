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
    auto q_params = fbgemm::ChooseQuantizationParams(
        /*min=*/x_min,
        /*max=*/x_max,
        /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
        /*qmax=*/
        is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
        /*preserve_sparsity=*/false);

    q_params.precision = precision;
    std::cout << "Input scale is " << q_params.scale << " min " << x_min
              << " max " << x_max << std::endl;
    // ReQuantizeForFloat requires pointers to the zero point values,
    // since in the case of rowwise quantization these will be arrays rather
    // than scalars. But in this case, we're doing whole-tensor quantization so
    // we just pass a pointer to the scale values (and internally
    // ReQuantizeForFloat won't index past 0.

    float* bias_ptr = nullptr;
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

  struct qparams {
    float scale;
    std::int32_t zero_point;
  };

  qparams chooseQuantizationParams(
      float min,
      float max,
      int32_t qmin,
      int32_t qmax) {
    double scale =
        (std::max(max, 0.f) - std::min(min, 0.f)) / ((double)qmax - qmin);
    if (scale == 0) {
      scale = 0.1;
    }
    // If scale is 0, we arbitrary adjust the scale to 0.1
    assert(scale > 0);
    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);

    double zero_point_from_min = qmin - min / scale;
    double zero_point_from_max = qmax - max / scale;
    double zero_point_from_min_error = std::abs(qmin) + std::abs(min / scale);
    double zero_point_from_max_error = std::abs(qmax) + std::abs(max / scale);
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
        ? zero_point_from_min
        : zero_point_from_max;

    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = nearbyint(initial_zero_point);
    }
    qparams result;
    result.scale = scale;
    result.zero_point = nudged_zero_point;
    return result;
  }

  at::Tensor qnnpack_linear(at::Tensor input, at::Tensor packed_weight) {
    TORCH_CHECK(
        input.dim() >= 2,
        "The dimension of input tensor should be larger than or equal to 2");
    auto input_contig = input.contiguous();
    const auto* input_ptr = input_contig.data_ptr<float>();
    // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
    // matrices, respectively.

    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeightsQnnp>(packed_weight);
    auto packB = pack_ptr.w.get();
    auto kernel_zp = pack_ptr.w_zp;
    auto kernel_scale = pack_ptr.w_scale;
    size_t rows_w = pack_ptr.bias.size(0);
    size_t cols_w = input_contig.size(input_contig.dim() - 1);

    at::Tensor bias_vec = pack_ptr.bias;

    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

    auto bias_contig = bias_vec.contiguous();
    const float* bias_ptr = bias_contig.data_ptr<float>();

    // Calculate statistics for quantization of input Tensor
    // TODO: optimized kernel
    float x_min = std::numeric_limits<float>::infinity(),
          x_max = -std::numeric_limits<float>::infinity();
    auto input_numel = input_contig.numel();
    std::cout << "Input data ";
    for (int64_t i = 0; i < input_numel; ++i) {
      std::cout << " " << input_ptr[i];
      auto curr = input_ptr[i];
      if (curr < x_min) {
        x_min = curr;
      }
      if (curr > x_max) {
        x_max = curr;
      }
    }
    std::cout << "\n";
    auto q_params = chooseQuantizationParams(
        /*min=*/x_min,
        /*max=*/x_max,
        0,
        255);
    std::cout << "Input scale is " << q_params.scale << " min " << x_min
              << " max " << x_max << std::endl;
    std::cout << "kernel scale is " << kernel_scale <<" zp " <<kernel_zp << std::endl;
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
      std::cout << "weight data ";
      for (int i = 0; i < wt_numel; ++i) {
        qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
        std::cout << " " << w_data[i] + 128;
      }
      std::cout << std::endl;
      // Pass in dummy zero bias to pack function
      auto bias_fp32 = at::zeros(rows_w, weight_contig.options().dtype(at::kFloat));

      at::Tensor bias = at::quantize_per_tensor(bias_fp32, 1.0, 0, kQInt32);

      // Update the input scale to not pack again.
      pack_ptr.input_scale = q_params.scale;
      pack_ptr.w.reset();
      pack_ptr.w = std::make_unique<qnnpack::PackBMatrix>(
          cols_w /* input_channels */,
          rows_w /* output_channels */,
          kernel_zp,
          kernel_scale,
          (uint8_t*)qnnp_w_data,
          (int32_t*)bias.data_ptr<c10::qint32>());
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
    std::cout << "input sizes are " << input_contig.sizes()  <<" output sizes are " << out_sizes << std::endl;

    auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));

    size_t rows_input = 1;
    size_t cols_input = input_contig.size(input_contig.dim() - 1);
    for (size_t i = 0; i < input_contig.dim() - 1; ++i) {
      rows_input *= input_contig.size(i);
    }
    std::cout << " input stride " << cols_input << " output stride " << rows_w << std::endl;
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

    std::cout << "run status " << runStatus << std::endl;

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

static auto registry =
    torch::RegisterOperators()
        .op("quantized::linear_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .kernel<QLinearDynamicInt8<false>>(TensorTypeId::CPUTensorId))
        .op("quantized::linear_relu_dynamic(Tensor X, Tensor W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
                .kernel<QLinearDynamicInt8<true>>(TensorTypeId::CPUTensorId));
} // namespace
} // namespace native
} // namespace at
