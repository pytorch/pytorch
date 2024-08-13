#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/XnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/onednn/MKLDNNCommon.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>         // for _empty_affine_q...
#include <ATen/ops/_empty_affine_quantized_native.h>  // for empty_affine_qu...
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

  auto output_data = reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>());

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
            /*C=*/output_data,
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
            /*C=*/output_data,
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

at::Tensor PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32(
  at::Tensor input,
  double input_scale,
  int64_t input_zero_point) {
  TORCH_CHECK(!input.is_quantized(), "Input tensor for apply_with_input_q_dq_qweight_dq_output_fp32 is quantized; "
  "Expected input tensor in PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32 to be full precision.");

  return apply_with_input_q_dq_qweight_dq_output_fp32_impl<false>(input, input_scale, input_zero_point);
}

at::Tensor PackedLinearWeight::apply_with_input_q_dq_qweight_dq_relu_output_fp32(
  at::Tensor input,
  double input_scale,
  int64_t input_zero_point) {
  TORCH_CHECK(!input.is_quantized(), "Input tensor for apply_with_input_q_dq_qweight_dq_output_fp32 is quantized; "
  "Expected input tensor in PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32 to be full precision.");

  return apply_with_input_q_dq_qweight_dq_output_fp32_impl<true>(input, input_scale, input_zero_point);
}


template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_with_input_q_dq_qweight_dq_output_fp32_impl(
    const at::Tensor& input,
    double input_scale,
    int64_t input_zero_point) {
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  auto input_contig = input.expect_contiguous();
  const auto* input_ptr = input_contig->const_data_ptr<float>();

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  auto packB = w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = input.sizes()[input.dim() - 1];
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float input_scale_float = input_scale;
  int32_t input_zero_point_int32 = input_zero_point;

  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");

  const float* bias_ptr = nullptr;
  c10::MaybeOwned<at::Tensor> bias_contig;
  if (this->bias_.has_value()) {
    auto& bias = this->bias_.value();
    bias_contig = bias.expect_contiguous();
    TORCH_CHECK(bias_contig->dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_contig->sizes()[0] == N, "bias should have N elements: " + std::to_string(N));
    bias_ptr = bias_contig->data_ptr<float>();
  }

  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  // Allocate output Tensor and a buffer for fbgemmPacked to use
  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));
  auto buffer = at::empty_like(
      output,
      output.options().dtype(at::kInt),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto output_data = output.data_ptr<float>();

  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    fbgemm::PackAWithQuantRowOffset<uint8_t> packA(
        /*trans=*/fbgemm::matrix_op_t::NoTranspose,
        /*nRow=*/M,
        /*nCol=*/K,
        /*smat=*/input_ptr,
        /*ld=*/K,
        /*pmat=*/nullptr,
        /*scale=*/input_scale_float,
        /*zero_pt=*/input_zero_point_int32);

    fbgemm::DoNothing<float, float> doNothingObj{};
    for (const auto task_id : c10::irange(begin, end)) {
      if (q_scheme == c10::kPerTensorAffine) {
        // Process the per tensor quantization.
        //
        // After the uint8 * int8 matrix multiplication is performed, this
        // operation does:
        //  1) Add in row and column offsets to the rows and columns,
        //  respectively.
        //  2) Add in the bias term.
        fbgemm::ReQuantizeForFloat<ReluFused>
            outputProcObj(
                doNothingObj,
                input_scale_float,
                w_scale.data(),
                input_zero_point_int32,
                w_zp.data(),
                packA.getRowOffsetBuffer(),
                col_offsets.data(),
                bias_ptr,
                N /* nCol */);

        // Do the GEMM
        fbgemm::fbgemmPacked(
            /*packA=*/packA,
            /*packB=*/*packB,
            /*C=*/output_data,
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
        fbgemm::ReQuantizeForFloat<
            ReluFused,
            fbgemm::QuantizationGranularity::OUT_CHANNEL>
            outputProcObj(
                doNothingObj,
                input_scale_float,
                w_scale.data(),
                input_zero_point_int32,
                w_zp.data(),
                packA.getRowOffsetBuffer(),
                col_offsets.data(),
                bias_ptr,
                N /* nCol */);

        // Do the GEMM
        fbgemm::fbgemmPacked(
            /*packA=*/packA,
            /*packB=*/*packB,
            /*C=*/output_data,
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

#ifdef USE_XNNPACK
// TODO: add per_channel support in the future when xnnp supports it
template <typename scalar_t, bool kReluFused>
at::Tensor PackedLinearWeightsQnnp::apply_impl_xnnp(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  using underlying_t = typename scalar_t::underlying;

  std::lock_guard<std::mutex> lock(qnnp_mutex_);

  const std::string func_name = kReluFused ? "quantized::linear_relu (xnnpack)"
                                           : "quantized::linear (xnnpack)";
  TORCH_CHECK(
      input.dim() >= 2, func_name, ": Input tensor rank should be >= 2.");
  TORCH_CHECK(
      !per_channel(),
      func_name,
      ": xnnpack does not currently have per_channel support.");

  const auto input_contig = input.contiguous();
  const auto input_scale = input_contig.q_scale();

  const size_t rows_w = bias_.size(0);
  const size_t cols_w = input_contig.size(input_contig.dim() - 1);

  auto status = xnn_status_invalid_state;

  // Create an operator iff not already created
  if (!xnnp_linear_op ||
      (!this->input_scale.has_value() ||
       this->input_scale.value() != input_scale)) {
    // Update the input scale so we may cache the op
    this->input_scale = input_scale;

    xnn_operator_t xnnp_op = nullptr;

    const float* weight_scales_data = w_scales.const_data_ptr<float>();

    // prepare weights
    underlying_t w_zp = static_cast<underlying_t>(
        orig_weight.q_zero_point() +
        (std::is_same<underlying_t, uint8_t>::value ? 128 : 0));

   at::Tensor xnnp_weight = at::_empty_affine_quantized(
        orig_weight.sizes(),
        c10::CppTypeToScalarType<scalar_t>::value,
        weight_scales_data[0],
        w_zp);

    // copy from the original weight and take care of dtype change if necessary
    at::native::xnnp_utils::q8_copy_int8_weight_and_add_offset<scalar_t>(
        orig_weight, xnnp_weight);

    // Original bias was float, so we requantize it here.
    at::Tensor qbias = quant_utils::QuantizeBias(false, bias_, orig_weight, input_scale);

    // output limits
   auto output_min = kReluFused
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).first
        : std::numeric_limits<underlying_t>::min();
    auto output_max = kReluFused
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        ? activationLimits<underlying_t>(output_scale, output_zero_point, Activation::RELU).second
        : std::numeric_limits<underlying_t>::max();

    // Create an operator
    status = at::native::xnnp_utils::xnnp_create_fully_connected_nc(
        cols_w, /* input_channels */
        rows_w, /* output_channels */
        cols_w, /* input_stride */
        rows_w, /* output_stride */
        input_contig.q_zero_point(),
        input_contig.q_scale(),
        w_zp,
        weight_scales_data[0],
        reinterpret_cast<const underlying_t*>(
            xnnp_weight.template data_ptr<scalar_t>()),
        reinterpret_cast<int32_t*>(qbias.data_ptr<c10::qint32>()),
        output_zero_point,
        output_scale,
        output_min,
        output_max,
        0, /* flags */
        &xnnp_op);
    xnnp_linear_op = xnnpack_operator(xnnp_op);

    TORCH_CHECK(
        status == xnn_status_success,
        func_name,
        ": xnn create operator failed(",
        status,
        ")");
  }

  /*
   * Allocate output Tensor and a buffer for XNNPACK to use
   * The resulting matrix here is 2-D, let's view it with the original
   * left hand dimensions of the input. Here are two examples:
   * 1. If the input tensor is {M, K}, the output tensor is {M, N}.
   * 2. If the input tensor is {b, M, K}, the output tensor is {b, M, N}.
   */
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = static_cast<int64_t>(rows_w);
  at::Tensor output = at::native::empty_affine_quantized(
      out_sizes,
      c10::CppTypeToScalarType<scalar_t>::value,
      std::nullopt /* layout */,
      c10::kCPU,
      std::nullopt /* pin_memory */,
      output_scale,
      output_zero_point,
      input.suggest_memory_format());

  // calculate batch_size
  size_t rows_input = 1;
  for (const auto i : c10::irange(input_contig.dim() - 1)) {
    rows_input *= input_contig.size(i);
  }

  // Reshape the operator
  status = at::native::xnnp_utils::xnnp_reshape_fully_connected_nc(
      xnnp_linear_op.get(),
      rows_input, /* batch_size */
      caffe2::pthreadpool_());

  // Setup the operator
  status = at::native::xnnp_utils::xnnp_setup_fully_connected_nc(
      xnnp_linear_op.get(),
      reinterpret_cast<const underlying_t*>(
          input_contig.template data_ptr<scalar_t>()),
      reinterpret_cast<underlying_t*>(output.template data_ptr<scalar_t>())
    );

  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn setup operator failed(",
      status,
      ")");

  // Run the operator
  status = xnn_run_operator(
      xnnp_linear_op.get(), // Linear op
      caffe2::pthreadpool_() // threadpool
  );
  TORCH_CHECK(
      status == xnn_status_success,
      func_name,
      ": xnn run operator failed(",
      status,
      ")");

  return output;
}
#endif // USE_XNNPACK

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
    at::Tensor qbias = quant_utils::QuantizeBias(is_per_channel, bias_fp32, weight_contig, input_scale);

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

#ifdef USE_XNNPACK
static bool can_use_xnnp(c10::ScalarType dtype, bool per_channel) {
  if(!at::native::xnnpack::available()) {
    return false;
  }

  bool supported_dtypes = dtype == c10::kQInt8;
  bool invalid_config = per_channel; /* xnnp does not currently support
                                        per-channel fully connected op */
  if (supported_dtypes && invalid_config) {
    /* don't want this to fall through to QNNPACK */
    TORCH_CHECK(
        false,
        "quantized::linear (xnnpack): Unsupported config for dtype KQInt8");
  }
  return supported_dtypes && !invalid_config;
}
#endif // USE_XNNPACK

at::Tensor PackedLinearWeightsQnnp::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
#ifdef USE_XNNPACK
  if (can_use_xnnp(input.scalar_type(), per_channel())) {
    return apply_impl_xnnp<c10::qint8, false>(
        input, output_scale, output_zero_point);
  } /* fall through for unsupported types, configs, or shapes */
#endif // USE_XNNPACK
  return apply_impl<false>(std::move(input), output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightsQnnp::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
#ifdef USE_XNNPACK
  if (can_use_xnnp(input.scalar_type(), per_channel())) {
    return apply_impl_xnnp<c10::qint8, true>(
        input, output_scale, output_zero_point);
  } /* fall through for unsupported types, configs, or shapes */
#endif // USE_XNNPACK
  return apply_impl<true>(std::move(input), output_scale, output_zero_point);
}

#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()
template <PostOps post_op>
at::Tensor PackedLinearWeightsOnednn::apply_impl(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point,
    torch::List<at::Scalar> post_op_args) {
  const int64_t dim = input.dim();
  TORCH_CHECK(
      dim != 0,
      "qlinear (ONEDNN): input dim should be at least 1, but got 0");
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::QUInt8,
      "qlinear (ONEDNN): data type of input should be QUint8.");

  auto input_contig = input.expect_contiguous();
  auto& w = *(weight_.get());
  auto K = input.size(dim - 1), M = input.numel() / K, N = w.get_dim(1);
  auto input_dims = {M, K};
  auto input_data_type = dnnl::memory::data_type::u8;
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  ideep::attr_t op_attr = ideep::attr_t();
  if (post_op == Relu) {
    op_attr = ideep::attr_t::fuse_relu();
  } else if (post_op == LeakyRelu) {
    op_attr = ideep::attr_t::fuse_relu(/*scale=*/1.0f, /*alpha=*/post_op_args.get(0).to<double>());
  } else if (post_op == Tanh) {
    op_attr = ideep::attr_t::fuse_tanh();
  }
  ideep::tensor x(input_desc, input_contig->data_ptr<c10::quint8>());
  auto dst_dims = {M, N};
  double input_scale = input.q_scale();
  int64_t input_zero_point = input.q_zero_point();
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/input_scale);
  const ideep::scale_t& weights_scales = w.get_scale();
  // Scales of ONEDNN and PyTorch are reciprocal
  const ideep::scale_t& dst_scales = ideep::scale_t(1, 1.0/output_scale);
  const ideep::zero_point_t& src_zero_point = ideep::zero_point_t(1, input_zero_point);
  const ideep::zero_point_t& dst_zero_point = ideep::zero_point_t(1, output_zero_point);
  // Compute: Use ideep::matmul_forward to support asymmetric quantization
  // Allocate output Tensor
  at::Tensor output = at::_empty_affine_quantized(
      dst_dims,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  if (output.numel() == 0) {
    return output;
  }
  ideep::tensor y({dst_dims, ideep::tensor::data_type::u8,
                   {output.strides().cbegin(), output.strides().cend()}},
                  output.data_ptr());
  bool with_bias = bias_.has_value();
  if (with_bias) {
    // Bias might be modified outside (e.g. by quantization bias correction).
    // If so, update the prepacked bias as well.
    if (bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
      bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
    }
  }
  const auto& b = with_bias ? bias_.value() : ideep::tensor();
  // Primitive cache is initialized when called for the first time
  // and won't be updated afterwards.
  int num_threads = at::get_num_threads();
  PrimitiveCacheKey cache_key = std::make_tuple(
      input_scale, input_zero_point, input_dims, output_scale, output_zero_point, num_threads, /*accum scale*/1.0, /*accum zero point*/0);
  c10::call_once(*cache_initialized_flag, [&](){
      LinearParams params;
      ideep::matmul_forward::prepare</*is_dynamic=*/false>(
          params, x, w, b, y,
          src_scales, weights_scales, dst_scales,
          src_zero_point, dst_zero_point, 1.0f, 1.0f, op_attr);
      get_cache() = LinearPrimitiveCache(cache_key, params);
      w = w.reorder_if_differ_in(params.pd.weights_desc());
  });
  if (get_cache().hit(cache_key)) {
    LinearParams& params = get_cache().get_param();
    ideep::matmul_forward::compute<false, false>(params, x, w, b, y);
  } else {
    ideep::matmul_forward::compute(x, w, b, y, src_scales, weights_scales,
                                   dst_scales, src_zero_point, dst_zero_point,
                                   1.0f, 1.0f, op_attr);
  }
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  if (output.sizes().vec() == out_sizes)
    return output;
  return output.reshape(out_sizes);
}

at::Tensor PackedLinearWeightsOnednn::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<NoPostOp>(
      std::move(input), output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightsOnednn::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<Relu>(
      std::move(input), output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightsOnednn:: apply_leaky_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point,
    double negative_slope) {
  torch::List<at::Scalar> post_op_args =
      {at::Scalar(negative_slope)};
  return apply_impl<LeakyRelu>(
      std::move(input), output_scale, output_zero_point, post_op_args);
}

at::Tensor PackedLinearWeightsOnednn:: apply_tanh(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<Tanh>(
      std::move(input), output_scale, output_zero_point);
}

static at::Tensor linear_int8_with_onednn_weight(
    at::Tensor input, // int8 CPU Tensor, not QTensor
    double input_scale,
    int64_t input_zero_point,
    at::Tensor onednn_weight, // int8 tensor from MkldnnCPU
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<at::Tensor> bias, // plain tensor
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<at::Tensor> other, // extra input for binary post-op
    double other_scale,
    int64_t other_zero_point,
    const c10::string_view& binary_post_op, // e.g. "none", "sum", "add"
    double binary_alpha,
    const c10::string_view& unary_post_op, // e.g. "none", "relu"
    torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    c10::string_view& unary_post_op_algorithm) {
  using ideep::tensor;
  const int64_t dim = input.dim();
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Byte,
      "qlinear with mkldnn tensor: data type of input should be uint8 (unsigned char).");
  TORCH_CHECK(onednn_weight.scalar_type() == c10::ScalarType::Char,
      "qlinear with mkldnn tensor: data type of weight should be int8 (char).");
  TORCH_CHECK(
      weight_scales.scalar_type() == c10::ScalarType::Float, "weight scales should be dtype c10::ScalarType::Float.");
  TORCH_CHECK(
      binary_alpha == 1.0f, "onednn qlinear: alpha != 1 for binary post op is not yet supported.");
  bool fp32_output = output_dtype.has_value() && (output_dtype.value() == c10::kFloat);
  bool bf16_output = output_dtype.has_value() && (output_dtype.value() == c10::kBFloat16);
  if (fp32_output || bf16_output) {
    TORCH_CHECK(
        output_scale == 1.0f && output_zero_point == 0, "onednn qlinear: expect scale=1 and zero point=0 for fp32 output");
  }
  if (binary_post_op != "none") {
    /* Supported cases for binary post op:
      +-------------------+--------------+---------------+
      | Extra input dtype | Output dtype | Post op       |
      +-------------------+--------------+---------------+
      | Fp32/bf16         | fp32/bf16    | sum           |
      +-------------------+--------------+---------------+
      | Fp32/bf16         | int8         | add           |
      +-------------------+--------------+---------------+
      | int8              | fp32/bf16    | not supported |
      +-------------------+--------------+---------------+
      | int8              | int8         | sum           |
      +-------------------+--------------+---------------+
    */
    TORCH_CHECK(other.has_value(), "onednn qlinear: the extra input is missing for post op ", binary_post_op);
    if (fp32_output || bf16_output) {
      TORCH_CHECK(
          other_scale == 1.0f && other_zero_point == 0,
          "onednn qlinear: expect extra input scale = 1.0 and zero point = 0 when output dtype is ", output_dtype.value(),
          ", but got ", other_scale, " and ", other_zero_point, ", respectively"
      );
    }
    if (binary_post_op == "sum") {
      auto expected_dtype = output_dtype.has_value() ? output_dtype.value() : c10::kByte;
      TORCH_CHECK(
          other.value().scalar_type() == expected_dtype,
          "onednn qlinear: the dtype of extra input for binary post op should be ", expected_dtype,
          " (same as output dtype), but got ", other.value().scalar_type()
      );
    }
  }

  // If the input has more than two dimensions, we will reshape it to a 2-dimensional form
  // for calculation and subsequently reshape the output back.
  auto input_contig =
      dim == 2 ? input.contiguous() : input.reshape({-1, input.size(dim - 1)}).contiguous();

  auto src = at::native::itensor_from_tensor(input_contig);
  auto packed_weight = at::native::itensor_from_mkldnn(onednn_weight);
  int64_t K = input.size(dim - 1), M = input.numel() / K, N = packed_weight.get_dim(1);

  auto output_size = input.sizes().vec();
  output_size[dim - 1] = N;

  std::optional<ideep::tensor> onednn_bias{std::nullopt};
  bool with_bias = bias.has_value();
  at::Tensor bias_val_float;
  if (with_bias) {
    bias_val_float = bias.value().to(at::kFloat);
    if (bias_val_float.dim() == 1) {
      auto b_reshape = bias_val_float.reshape({1, bias_val_float.size(0)});
      onednn_bias = at::native::itensor_view_from_dense(b_reshape);
    } else {
      onednn_bias = at::native::itensor_view_from_dense(bias_val_float);
    }
  }
  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};
  at::Tensor output = binary_post_op == "sum" ?
      other.value() :
      at::empty(
        dst_dims,
        device(c10::kCPU)
            .dtype(fp32_output ? c10::kFloat : (bf16_output ? c10::kBFloat16 : c10::kByte))
      );
  if (output.numel() == 0) {
    return output;
  }
  tensor dst = at::native::itensor_view_from_dense(output);
  static tensor empty_tensor;
  static tensor::desc empty_tensor_desc;
  tensor src1 = binary_post_op == "add" ?
      at::native::itensor_view_from_dense(other.value().reshape({-1, other.value().size(dim - 1)})) :
      empty_tensor;

  // Create onednn primitive
  auto src_desc = tensor::desc(src_dims, ideep::data_type::u8, ideep::format_tag::any);
  auto weights_desc = packed_weight.get_desc();
  auto dst_dtype = dst.get_data_type();
  auto dst_desc = tensor::desc(dst_dims, dst_dtype, ideep::format_tag::any);
  auto bias_desc = with_bias ?
      tensor::desc(onednn_bias.value().get_dims(), ideep::data_type::f32, ideep::format_tag::any) :
      empty_tensor_desc;
  // Get op attr for primitive
  // Note: output_scale & output_zero_point are for re-quantization of the final output.
  // And other_scale & other_zero_point are for dequantization of other.
  auto other_desc = binary_post_op == "add" ? src1.get_desc() : empty_tensor_desc;
  auto op_attr = onednn_utils::create_attr_by_post_op(
    binary_post_op,
    binary_alpha,
    other_scale,
    other_zero_point,
    other_desc,
    unary_post_op,
    unary_post_op_args,
    unary_post_op_algorithm
  );
  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }
  if (input_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
  }
  op_attr.set_scales_mask(DNNL_ARG_WEIGHTS, ideep::utils::op_scale_mask(weight_scales.numel()));
  if (output_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_DST, 0);
  }
  if (output_zero_point != 0) {
    op_attr.set_zero_points_mask(DNNL_ARG_DST, 0);
  }
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto engine = ideep::engine::cpu_engine();
  auto primitive_desc = with_bias ?
      dnnl::matmul::primitive_desc(engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr) :
      dnnl::matmul::primitive_desc(engine, src_desc, weights_desc, dst_desc, op_attr);
  auto primitive = dnnl::matmul(primitive_desc);

  // Reorder weight if needed
  auto expected_weight = packed_weight.reorder_if_differ_in(primitive_desc.weights_desc());

  // Prepare args and execute primitive
  tensor scratchpad(primitive_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weight});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, onednn_bias.value()});
  }
  tensor src_scales_t = tensor(ideep::scale_t(1, input_scale));
  tensor wei_scales_t = at::native::itensor_from_tensor(weight_scales);
  tensor dst_scales_t = tensor(ideep::scale_t(1, output_scale));
  tensor src_zp_t = tensor(ideep::zero_point_t(1, input_zero_point));
  tensor dst_zp_t = tensor(ideep::zero_point_t(1, output_zero_point));
  if (input_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  if (output_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_t});
  }
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});
  if (input_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zp_t});
  }
  if (output_zero_point != 0) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_t});
  }
  if (binary_post_op == "add") {
    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, src1});
  }
  primitive.execute(ideep::stream::default_stream(), args);
  return dim == 2 ? output : output.reshape(output_size);
}
#endif // #if AT_MKLDNN_ENABLED()

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

class QLinearLeakyReluInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point,
      double negative_slope) {
#if AT_MKLDNN_ENABLED() || !defined(STRIP_ERROR_MESSAGES)
    auto& ctx = at::globalContext();
#endif
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return dynamic_cast<PackedLinearWeightsOnednn*>(packed_weight.get())->apply_leaky_relu(
          std::move(input), output_scale, output_zero_point, negative_slope);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_leaky_relu ",
        toString(ctx.qEngine()));
  }
};


class QLinearTanhInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
#if AT_MKLDNN_ENABLED() || !defined(STRIP_ERROR_MESSAGES)
    auto& ctx = at::globalContext();
#endif
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return dynamic_cast<PackedLinearWeightsOnednn*>(packed_weight.get())->apply_tanh(
          std::move(input), output_scale, output_zero_point);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_tanh ",
        toString(ctx.qEngine()));
  }
};

template <bool ReluFused>
class QLinearInt8FusedQDQ final {
 public:
  static at::Tensor run(
      at::Tensor input,
      double input_scale,
      int64_t input_zero_point,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    if (ReluFused) {
      return packed_weight->apply_with_input_q_dq_qweight_dq_relu_output_fp32(
          std::move(input), input_scale, input_zero_point);
    } else {
      return packed_weight->apply_with_input_q_dq_qweight_dq_output_fp32(
          std::move(input), input_scale, input_zero_point);
    }
  }
};

class QLinearOnednn final {
 public:
  static Tensor run_pointwise(
      Tensor act, // int8 CPU tensor, not QTensor
      double act_scale,
      int64_t act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      c10::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      c10::string_view post_op_algorithm) {
#if AT_MKLDNN_ENABLED()
    static std::optional<at::Tensor> other = std::nullopt;
    static const c10::string_view binary_post_op = "none";
    return linear_int8_with_onednn_weight(
        act, act_scale, act_zero_point,
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, /*other scale*/1.0, /*other zp*/0,
        binary_post_op, /*binary alpha*/1.0,
        post_op_name, post_op_args, post_op_algorithm
    );
#endif
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }

  static Tensor run_pointwise_tensor(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      c10::string_view post_op_name,
      torch::List<std::optional<at::Scalar>> post_op_args,
      c10::string_view post_op_algorithm) {
#if AT_MKLDNN_ENABLED()
    TORCH_CHECK(act_scale.numel() == 1 && act_zero_point.numel() == 1,
        "onednn int8 linear: act scale/zp size should be 1");
    static std::optional<at::Tensor> other = std::nullopt;
    static const c10::string_view binary_post_op = "none";
    return linear_int8_with_onednn_weight(
        act, act_scale.item().toDouble(), act_zero_point.item().toLong(),
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, /*other scale*/1.0, /*other zp*/0,
        binary_post_op, /*binary alpha*/1.0,
        post_op_name, post_op_args, post_op_algorithm
    );
#endif
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }

  static Tensor run_pointwise_binary(
      Tensor act, // int8 CPU tensor, not QTensor
      double act_scale,
      int64_t act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other, // extra input for binary post-op
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      c10::string_view binary_post_op, // e.g. "none", "sum", "add"
      double binary_alpha,
      c10::string_view unary_post_op, // e.g. "none", "relu"
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      c10::string_view unary_post_op_algorithm) {
#if AT_MKLDNN_ENABLED()
    return linear_int8_with_onednn_weight(
        act, act_scale, act_zero_point,
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, other_scale, other_zero_point,
        binary_post_op, binary_alpha,
        unary_post_op, unary_post_op_args, unary_post_op_algorithm
    );
#endif
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }

  static Tensor run_pointwise_binary_tensor(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor act_scale,
      Tensor act_zero_point,
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      Tensor weight_scales,
      Tensor weight_zero_points,
      std::optional<at::Tensor> other, // extra input for binary post-op
      std::optional<Tensor> bias,
      double output_scale,
      int64_t output_zero_point,
      std::optional<c10::ScalarType> output_dtype,
      double other_scale,
      int64_t other_zero_point,
      c10::string_view binary_post_op, // e.g. "none", "sum", "add"
      double binary_alpha,
      c10::string_view unary_post_op, // e.g. "none", "relu"
      torch::List<std::optional<at::Scalar>> unary_post_op_args,
      c10::string_view unary_post_op_algorithm) {
#if AT_MKLDNN_ENABLED()
    TORCH_CHECK(act_scale.numel() == 1 && act_zero_point.numel() == 1,
        "onednn int8 linear: act scale/zp size should be 1");
    return linear_int8_with_onednn_weight(
        act, act_scale.item().toDouble(), act_zero_point.item().toLong(),
        onednn_weight, weight_scales, weight_zero_points,
        bias, output_scale, output_zero_point, output_dtype,
        other, other_scale, other_zero_point,
        binary_post_op, binary_alpha,
        unary_post_op, unary_post_op_args, unary_post_op_algorithm
    );
#endif
    TORCH_CHECK(false, "Unimplemented (int8 linear with packed weight and bias)");
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear"), TORCH_FN(QLinearInt8<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu"), TORCH_FN(QLinearInt8<true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_leaky_relu"), TORCH_FN(QLinearLeakyReluInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_tanh"), TORCH_FN(QLinearTanhInt8::run));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear"), TORCH_FN(QLinearInt8<false>::run));
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_with_input_q_dq_qweight_dq_output_fp32"), TORCH_FN(QLinearInt8FusedQDQ<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32"), TORCH_FN(QLinearInt8FusedQDQ<true>::run));
}

TORCH_LIBRARY_IMPL(onednn, MkldnnCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise"),
      TORCH_FN(QLinearOnednn::run_pointwise));
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.tensor"),
      TORCH_FN(QLinearOnednn::run_pointwise_tensor));
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary"),
      TORCH_FN(QLinearOnednn::run_pointwise_binary));
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary_tensor"),
      TORCH_FN(QLinearOnednn::run_pointwise_binary_tensor));
}

} // namespace
} // namespace native
} // namespace at
