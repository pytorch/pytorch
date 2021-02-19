#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#include <torch/custom_class.h>

#include <algorithm>
#include <string>

torch::class_<LinearPackedParamsBase> register_linear_params();

#ifdef USE_FBGEMM
template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_dynamic_impl(at::Tensor input, bool reduce_range) {
  using at::Tensor;
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

  auto packB = w.get();

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
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  q_params.precision = precision;

  // ReQuantizeForFloat requires pointers to the zero point values,
  // since in the case of rowwise quantization these will be arrays rather
  // than scalars. But in this case, we're doing whole-tensor quantization so
  // we just pass a pointer to the scale values (and internally
  // ReQuantizeForFloat won't index past 0.

  const float* bias_ptr = nullptr;
  at::Tensor bias_vec;
  if (bias_.has_value()) {
    bias_vec = bias_.value();
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
      if (q_scheme == c10::kPerTensorAffine) {
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
            /*Bq_scale=*/w_scale.data(),
            /*Aq_zero_point=*/q_params.zero_point,
            /*Bq_zero_point=*/w_zp.data(),
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

      } else if (q_scheme == c10::kPerChannelAffine) {
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
                /*Bq_scale=*/w_scale.data(),
                /*Aq_zero_point=*/q_params.zero_point,
                /*Bq_zero_point=*/w_zp.data(),
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

at::Tensor PackedLinearWeight::apply_dynamic(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(std::move(input), reduce_range);
}

at::Tensor PackedLinearWeight::apply_dynamic_relu(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input), reduce_range);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <bool ReluFused>
at::Tensor PackedLinearWeightsQnnp::apply_dynamic_impl(at::Tensor input) {
  using at::Tensor;
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  auto input_contig = input.contiguous();
  // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
  // matrices, respectively.

  auto packB = w.get();
  size_t rows_w = bias_.size(0);
  size_t cols_w = input_contig.size(input_contig.dim() - 1);

  at::Tensor bias_vec = bias_;

  TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

  auto bias_contig = bias_vec.contiguous();
  const float* bias_ptr = bias_contig.data_ptr<float>();

  // Calculate statistics for quantization of input Tensor
  // TODO: optimized kernel
  float x_min;
  float x_max;
  if (input.numel() > 0) {
    x_min = input_contig.min().item<float>();
    x_max = input_contig.max().item<float>();
  } else {
    // On empty input, no output data will be generated,
    // so use arbitrary qparams.
    x_min = 0;
    x_max = 0;
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255);
  float* weight_scales_data = w_scales.data_ptr<float>();
  if (!input_scale.has_value() || input_scale.value() != q_params.scale) {
    generate_requantization_scales(
        w_scales, q_params.scale, 1.f, requantization_scales);
  }

  if (!input_scale.has_value()) {
    // Get the original weight and adjust it to uint8 from int8
    auto weight_contig = orig_weight;

    // TODO(kimishpatel), we are allocating affine_quantized regardless of per channel or not.
    // This allocation is actually used only for packing weight and thus will be freed.
    // Still we should be consistent. Fix this.
    Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8),
        weight_scales_data[0],
        w_zero_points[0]);
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
    auto wt_numel = weight_contig.numel();
    for (int i = 0; i < wt_numel; ++i) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }

    // Pass in nullptr for bias, as we pass FP32 bias to run function.
    w.reset();
    w = std::make_unique<qnnpack::PackBMatrix>(
        cols_w /* input_channels */,
        rows_w /* output_channels */,
        w_zero_points.data(),
        requantization_scales.data(),
        (uint8_t*)qnnp_w_data,
        nullptr);
    packB = w.get();
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
      // On mobile, we release the original weight by resetting the intrusive_ptr.
      // Calling unpack after this will throw an assertion.
      orig_weight.reset();
    }
  }

  // Update the input scale to not pack weights again.
  // as well as to avoid repopulating requant scale if scale has not changed.
  input_scale = q_params.scale;

  // Quantize input
  Tensor q_input = at::quantize_per_tensor(
      input_contig, q_params.scale, q_params.zero_point, c10::kQUInt8);

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
      w_zero_points.data(),
      /* for dynamic should really be called dequant scale */
      requantization_scales.data(),
      (uint8_t*)q_input.data_ptr<c10::quint8>(),
      cols_input /* input_stride */,
      packB->getPackedWeights(),
      bias_ptr,
      output.data_ptr<float>(),
      rows_w /* output_stride */,
      caffe2::pthreadpool_() /* threadpool */);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Linear operator");
  return output;
}

at::Tensor PackedLinearWeightsQnnp::apply_dynamic(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(std::move(input));
}

at::Tensor PackedLinearWeightsQnnp::apply_dynamic_relu(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input));
}

#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM

template <bool ReluFused>
at::Tensor PackedLinearWeightFp16::apply_dynamic_impl(at::Tensor input) {
  const at::Tensor input_contig = input.contiguous();
  const float* input_ptr = input_contig.data_ptr<float>();

  auto& packed_weight_fp16 = *w;

  TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
  TORCH_CHECK(input.dim() >= 2);

  const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  const int64_t N = packed_weight_fp16.numCols();
  std::vector<int64_t> output_size = input.sizes().vec();
  output_size.back() = N;
  at::Tensor output = at::empty(output_size, input.options().dtype(at::kFloat));

  // Call the fp16 gemm interface
  fbgemm::cblas_gemm_compute(
      fbgemm::matrix_op_t::NoTranspose,
      M,
      input_ptr,
      packed_weight_fp16,
      0.0f,
      output.data_ptr<float>());

  // Add bias term
  if (bias_.has_value()) {
    TORCH_CHECK(bias_->dim() == 1);
    output.add_(*bias_);
  }

  return output;
}

at::Tensor PackedLinearWeightFp16::apply_dynamic(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(std::move(input));
}

at::Tensor PackedLinearWeightFp16::apply_dynamic_relu(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input));
}

void PackedLinearWeightFp16::set_bias(c10::optional<at::Tensor> bias) {
  bias_ = std::move(bias);
}

#endif // USE_FBGEMM

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearDynamicInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      bool reduce_range) {
    auto& ctx = at::globalContext();

    if (ReluFused) {
      return packed_weight->apply_dynamic_relu(std::move(input), reduce_range);
    } else {
      return packed_weight->apply_dynamic(std::move(input), reduce_range);
    }
  }
};

template <bool ReluFused>
class QLinearDynamicFp16 final {
 public:
#ifdef USE_FBGEMM
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

    TORCH_INTERNAL_ASSERT(!ReluFused);
    return packed_weight->apply_dynamic(std::move(input));
  }
#else // USE_FBGEMM
  static at::Tensor run(
      at::Tensor /* input */,
      const c10::intrusive_ptr<LinearPackedParamsBase>& /* packed_weight */) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic"), TORCH_FN(QLinearDynamicInt8<true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16"), TORCH_FN(QLinearDynamicFp16<false>::run));
}

TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
}

} // namespace
} // namespace native
} // namespace at
