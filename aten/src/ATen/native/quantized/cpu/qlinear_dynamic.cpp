#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/ACLUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/library.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_native.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_native.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16_native.h>
#include <ATen/ops/quantize_per_tensor.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <string>
#include <type_traits>

#ifdef USE_FBGEMM
template <bool ReluFused>
at::Tensor PackedLinearWeight::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {
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
  const auto* input_ptr = input_contig.const_data_ptr<float>();

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
  // matrices, respectively.
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int64_t M = size_to_dim_(input.dim() - 1, input.sizes());

  auto packB = w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = input.size(input.dim() - 1);
  TORCH_CHECK(
      K == static_cast<int64_t>(packB->numRows()),
      "The number of rows in the packB should be equal to K: " +
          std::to_string(K));

  // Calculate statistics for quantization of the input Tensor
  float x_min = std::numeric_limits<float>::quiet_NaN(), x_max = std::numeric_limits<float>::quiet_NaN();
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
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        /*scale=*/q_params.scale,
        /*zero_pt=*/q_params.zero_point);
    // TODO: Consider a way to pre-allocate and reuse
    // pmat buffer.

    // This is the end of the pipeline, pass the resulting matrix through.
    fbgemm::DoNothing<float, float> doNothingObj{};

    for (const auto task_id : c10::irange(begin, end)) {
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

at::Tensor PackedLinearWeight::apply_dynamic(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(
      std::move(input), reduce_range);
}

at::Tensor PackedLinearWeight::apply_dynamic_relu(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input), reduce_range);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <bool ReluFused>
at::Tensor PackedLinearWeightsQnnp::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {
  if (reduce_range) {
    TORCH_WARN_ONCE("Currently, qnnpack incorrectly ignores reduce_range when it is set to true; this may change in a future release.");
  }

  using at::Tensor;
  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  auto input_contig = input.contiguous();
  // C(output) = A(input) x B(weight), where C, A, B are M x N, M x K, K x N
  // matrices, respectively.

  // Weight packing is not thread safe
  std::lock_guard<std::mutex> lock(qnnp_mutex_);
  auto packB = w.get();
  size_t rows_w = bias_.size(0);
  size_t cols_w = input_contig.size(input_contig.dim() - 1);

  at::Tensor bias_vec = bias_;

  TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");

  auto bias_contig = bias_vec.contiguous();
  const float* bias_ptr = bias_contig.const_data_ptr<float>();

  // Calculate statistics for quantization of input Tensor
  // TODO: optimized kernel
  float x_min = 0;
  float x_max = 0;
  if (input.numel() > 0) {
    x_min = input_contig.min().item<float>();
    x_max = input_contig.max().item<float>();
  } else {
    // On empty input, no output data will be generated,
    // so use arbitrary qparams.
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255);
  float* weight_scales_data = w_scales.data_ptr<float>();

  if (!input_scale.has_value() || input_scale.value() != q_params.scale) {
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales,
        q_params.scale,
        1.f,
        requantization_scales);
  }

  if (!input_scale.has_value()) {
    // Get the original weight and adjust it to uint8 from int8
    auto weight_contig = orig_weight;

    // TODO(kimishpatel), we are allocating affine_quantized regardless of per
    // channel or not. This allocation is actually used only for packing weight
    // and thus will be freed. Still we should be consistent. Fix this.
    Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(c10::kCPU).dtype(c10::kQUInt8),
        weight_scales_data[0],
        w_zero_points[0]);
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
    auto wt_numel = weight_contig.numel();
    for (const auto i : c10::irange(wt_numel)) {
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
      // On mobile, we release the original weight by resetting the
      // intrusive_ptr. Calling unpack after this will throw an assertion.
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
  for (const auto i : c10::irange(input_contig.dim() - 1)) {
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

  // Call the relu operator here until qlinear dynamic in QNNPACK
  // supports it natively.
  if (ReluFused) {
    output.relu_();
  }
  return output;
}

at::Tensor PackedLinearWeightsQnnp::apply_dynamic(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(std::move(input), reduce_range);
}

at::Tensor PackedLinearWeightsQnnp::apply_dynamic_relu(
    at::Tensor input,
    bool reduce_range ) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input), reduce_range);
}

#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM

template <bool ReluFused>
at::Tensor& PackedLinearWeightFp16::apply_dynamic_impl(
    const at::Tensor& input,
    at::Tensor& output) {
  const at::Tensor input_contig = input.contiguous();
  const float* input_ptr = input_contig.const_data_ptr<float>();

  auto& packed_weight_fp16 = *w;

  TORCH_CHECK(input.size(input.dim() - 1) == packed_weight_fp16.numRows())
  TORCH_CHECK(input.dim() >= 2);

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int64_t M = size_to_dim_(input.dim() - 1, input.sizes());
  const int64_t N = packed_weight_fp16.numCols();
  std::vector<int64_t> output_sizes = input.sizes().vec();
  TORCH_CHECK(!output_sizes.empty())
  output_sizes.back() = N;
  // Resize output Tensor
  output.resize_(output_sizes);

  auto output_data = output.data_ptr<float>();

  int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    for (const auto task_id : c10::irange(begin, end)) {
      // Call the fp16 gemm interface
      fbgemm::cblas_gemm_compute(
          /*transa=*/fbgemm::matrix_op_t::NoTranspose,
          /*m=*/static_cast<int>(M),
          /*A=*/input_ptr,
          /*Bp=*/packed_weight_fp16,
          /*beta=*/0.0f,
          /*C=*/output_data,
          /*thread_id=*/static_cast<int>(task_id),
          /*num_threads=*/num_tasks);
    }
  });

  // Add bias term
  if (bias_.has_value()) {
    TORCH_CHECK(bias_->dim() == 1);
    output.add_(*bias_);
  }

  return output;
}

at::Tensor PackedLinearWeightFp16::apply_dynamic(
    at::Tensor input,
    bool /* reduce_range */) {
  at::Tensor output = at::empty({0}, input.options().dtype(at::kFloat));
  return apply_dynamic_impl</*ReluFused=*/false>(input, output);
}

at::Tensor PackedLinearWeightFp16::apply_dynamic_relu(
    at::Tensor input,
    bool /* reduce_range */) {
  at::Tensor output = at::empty({0}, input.options().dtype(at::kFloat));
  return apply_dynamic_impl</*ReluFused=*/true>(input, output);
}

at::Tensor& PackedLinearWeightFp16::apply_dynamic_out(
    const at::Tensor& input,
    at::Tensor& output,
    bool /* reduce_range */) {
  TORCH_CHECK((output.device() == c10::kCPU) && (output.dtype() == at::kFloat));
  return apply_dynamic_impl<false>(input, output);
}

at::Tensor& PackedLinearWeightFp16::apply_dynamic_relu_out(
    const at::Tensor& input,
    at::Tensor& output,
    bool /* reduce_range */) {
  TORCH_CHECK((output.device() == c10::kCPU) && (output.dtype() == at::kFloat));
  return apply_dynamic_impl<true>(input, output);
}

void PackedLinearWeightFp16::set_bias(std::optional<at::Tensor> bias) {
  bias_ = std::move(bias);
}

#endif // USE_FBGEMM

#if AT_MKLDNN_ENABLED()
template <bool ReluFused>
at::Tensor PackedLinearWeightsOnednn::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {
  // Dynamic: fp32 * int8 -> fp32
  using at::Tensor;

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float,
      "qlinear_dynamic (ONEDNN): data type of input should be float.");

  // Input -> uint8
  auto input_contig = input.contiguous();
  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});
  auto input_dims = input_reshaped.sizes().vec();
  auto input_data_type = dnnl::memory::data_type::f32;
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  ideep::attr_t op_attr = ReluFused ? ideep::attr_t::fuse_relu() : ideep::attr_t();
  ideep::tensor x;
  x.init(input_desc, input_contig.data_ptr());
  // Find quantization parameters
  float x_max = 0, x_min = 0;
#ifdef USE_FBGEMM
  // Use FBGEMM's FindMinMax if available since it's faster
  fbgemm::FindMinMax(
      /*m=*/input_contig.data_ptr<float>(),
      /*min=*/&x_min,
      /*max=*/&x_max,
      /*len=*/input.numel());
#else
  if (input_contig.numel() > 0) {
    auto [t_min, t_max] = at::aminmax(input_contig);
    x_max = t_max.item<float>();
    x_min = t_min.item<float>();
  }
#endif

#if defined(__aarch64__) && AT_MKLDNN_ACL_ENABLED()
  // oneDNN+ACL has optimized kernels for s8s8 matmul, so input is signed
  using input_qtype = int8_t;
#else
  using input_qtype = uint8_t;
#endif

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/std::numeric_limits<input_qtype>::min(),
      /*qmax=*/std::numeric_limits<input_qtype>::max(),
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);
  const std::vector<int32_t>& src_zero_point = std::vector<int32_t>(1, q_params.zero_point);
  // weights, dst
  auto w = *(weight_.get());
  auto dst_dims = {x.get_dim(0), w.get_dim(1)};
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/q_params.scale);
  const ideep::scale_t& weights_scales = w.get_scale();
  // Compute -> f32
  // Use ideep::matmul_forward instead of ideep::inner_product_forward,
  // since the latter does not support asymmetric quantization
  // Allocate output Tensor
  at::Tensor output = at::empty(dst_dims, input.options().dtype(at::kFloat));
  if (output.numel() == 0) return output;
  ideep::tensor y({dst_dims, ideep::tensor::data_type::f32,
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
      q_params.scale, q_params.zero_point, input_dims, 1.0, 0, num_threads, /*accum scale*/1.0, /*accum zero point*/0);
  c10::call_once(*cache_initialized_flag, [&](){
      LinearParams params;
      ideep::matmul_forward::prepare</*is_dynamic=*/true>(
          params, x, w, b, y,
          src_scales, weights_scales, ideep::scale_t(),
          src_zero_point, ideep::zero_point_t(), 1.0f, 1.0f, op_attr,
          ideep::tensor::data_type::f32, std::is_signed_v<input_qtype> ? ideep::s8s8 : ideep::u8s8);
      get_cache() = LinearPrimitiveCache(cache_key, params);
      w = w.reorder_if_differ_in(params.pd.weights_desc());
  });
  if (get_cache().hit_dynamic(cache_key)) {
    LinearParams& params = get_cache().get_param();
    ideep::matmul_forward::compute(params, x, w, b, y, src_scales, src_zero_point);
  } else {
    ideep::matmul_forward::compute(x, w, b, y,
                                   src_scales, weights_scales, ideep::scale_t(),
                                   src_zero_point, ideep::zero_point_t(),
                                   1.0f, 1.0f, op_attr);
  }
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = w.get_dim(1);
  if (output.sizes().vec() == out_sizes)
    return output;
  return output.reshape(out_sizes);
}

at::Tensor PackedLinearWeightsOnednn::apply_dynamic(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(
      std::move(input), reduce_range);
}

at::Tensor PackedLinearWeightsOnednn::apply_dynamic_relu(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(
      std::move(input), reduce_range);
}

static at::Tensor linear_dynamic_fp16_with_onednn_weight(
    at::Tensor input,
    at::Tensor onednn_weight, // fp16 tensor from MkldnnCPU
    std::optional<at::Tensor> bias,
    bool relu_fused) {
  using ideep::tensor;
  const int64_t dim = input.dim();
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float,
      "onednn linear dynamic fp16: data type of input should be float.");
  TORCH_CHECK(onednn_weight.scalar_type() == c10::ScalarType::Half,
      "onednn linear dynamic fp16: data type of weight should be half.");

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
  at::Tensor output = at::empty(
        dst_dims,
        at::device(c10::kCPU)
            .dtype(c10::kFloat)
      );
  if (output.numel() == 0) {
    return output;
  }
  tensor dst = at::native::itensor_view_from_dense(output);
  static tensor empty_tensor;
  static tensor::desc empty_tensor_desc;

  // Create matmul primitive
  auto src_dtype = ideep::data_type::f32;
  auto src_desc = tensor::desc(src_dims, src_dtype, ideep::format_tag::any);
  // onednn does not support f32f16f32 matmul, so we get primitive with f32 weight desc
  // weight is stored in f16 and reordered to f32 below by `reorder_if_differ_in`
  auto weights_desc = tensor::desc(packed_weight.get_dims(), ideep::data_type::f32, ideep::format_tag::any);
  auto dst_dtype = dst.get_data_type();
  auto dst_desc = tensor::desc(dst_dims, dst_dtype, ideep::format_tag::any);
  auto bias_desc = with_bias ?
      tensor::desc(onednn_bias.value().get_dims(), ideep::data_type::f32, ideep::format_tag::any) :
      empty_tensor_desc;
  // Get op attr for primitive
  auto op_attr = relu_fused ? ideep::attr_t::fuse_relu() : ideep::attr_t();
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto engine = ideep::engine::cpu_engine();
  auto primitive_desc = with_bias ?
      dnnl::matmul::primitive_desc(engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr) :
      dnnl::matmul::primitive_desc(engine, src_desc, weights_desc, dst_desc, op_attr);
  auto primitive = dnnl::matmul(primitive_desc);

  // Convert weight from f16 to f32 with layout changes
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
  primitive.execute(ideep::stream::default_stream(), args);
  return dim == 2 ? output : output.reshape(output_size);
}

#if AT_MKLDNN_ACL_ENABLED()

template <bool ReluFused>
at::Tensor PackedLinearWeightsACL::apply_dynamic_impl(
    at::Tensor input,
    bool reduce_range) {
  // Dynamic: fp32 * int8 -> fp32
  using at::Tensor;

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  TORCH_CHECK(
      input.scalar_type() == c10::ScalarType::Float,
      "qlinear_dynamic (ACL): data type of input should be float.");

  auto input_contig = input.contiguous();
  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});
  auto input_dims = input_reshaped.sizes().vec();

  int64_t m = input_dims[0];
  auto key = std::make_tuple(
      m, /* M */
      ReluFused, /* FUSE_RELU */
      static_cast<int64_t>(at::get_num_threads()), /* NUM_THREADS */
      1, /* INPUT_SCALE */
      0, /* INPUT_OFFSET */
      1, /* OUTPUT_SCALE */
      0, /* OUTPUT_OFFSET */
      true /* SIGNED_INPUT */
  );
  auto acl_gemm =
      get_acl_quant_matmul<at::native::acl_utils::DynamicQuantMatmul>(key);

  if (acl_gemm) {
    // Find quantization parameters
    float x_max = 0, x_min = 0;

#ifdef USE_FBGEMM
    // Use FBGEMM's FindMinMax if available since it's faster
    fbgemm::FindMinMax(
        /*m=*/input_contig.data_ptr<float>(),
        /*min=*/&x_min,
        /*max=*/&x_max,
        /*len=*/input.numel());
#else
    if (input_contig.numel() > 0) {
      auto [t_min, t_max] = at::aminmax(input_contig);
      x_max = t_max.item<float>();
      x_min = t_min.item<float>();
    }
#endif

    auto q_params = quant_utils::ChooseQuantizationParams(
        /*min=*/x_min,
        /*max=*/x_max,
        /*qmin=*/std::numeric_limits<int8_t>::min(),
        /*qmax=*/std::numeric_limits<int8_t>::max(),
        /*preserve_sparsity=*/false,
        /*force_scale_power_of_two=*/false,
        /*reduce_range=*/reduce_range);

    acl_gemm->src_tensor.allocator()->import_memory(
        (float*)input_contig.data_ptr());

    acl_gemm->src_q_tensor.info()->set_quantization_info(
        arm_compute::QuantizationInfo(
            q_params.scale, q_params.zero_point, true));

    // quantize src tensor: fp32 -> s8
    acl_gemm->quant.run();

    // allocation for fp32 out tensor
    auto output = at::empty({m, n_}, input.options().dtype(at::kFloat));
    if (output.numel() == 0)
      return output;

    // We set the offset to "-zero_point" for the GEMM, but to "zero_point" for
    // the quantization layer This is a known inconsistency in ACL.
    acl_gemm->src_q_tensor.info()->set_quantization_info(
        arm_compute::QuantizationInfo(
            q_params.scale, -q_params.zero_point, true));

    acl_gemm->dst_tensor.allocator()->import_memory((float*)output.data_ptr());

    // s8 src, s8 wei -> f32 dst
    acl_gemm->gemm.run();

    if (acl_gemm->relu.has_value()) {
      acl_gemm->relu->run();
    }

    // this will not free memory, it will just tell ACL that we're no longer
    // using the pointer
    acl_gemm->src_tensor.allocator()->free();
    acl_gemm->dst_tensor.allocator()->free();

    auto out_sizes = input.sizes().vec();
    out_sizes.back() = n_;
    if (output.sizes().vec() == out_sizes)
      return output;
    return output.reshape(out_sizes);
  }

  // fallback to oneDNN in the unlikely scinario that ACL's validation fails
  if (ReluFused) {
    return PackedLinearWeightsOnednn::apply_dynamic_relu(input, reduce_range);
  } else {
    return PackedLinearWeightsOnednn::apply_dynamic(input, reduce_range);
  }
}

at::Tensor PackedLinearWeightsACL::apply_dynamic(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(
      std::move(input), reduce_range);
}

at::Tensor PackedLinearWeightsACL::apply_dynamic_relu(
    at::Tensor input,
    bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input), reduce_range);
}

#endif // #if AT_MKLDNN_ACL_ENABLED()
#endif // #if AT_MKLDNN_ENABLED()

namespace at::native {
namespace {

template <bool ReluFused>
class QLinearDynamicInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      bool reduce_range) {
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

    auto output = packed_weight->apply_dynamic(std::move(input));

    // Call the relu operator here until fp16 linear dynamic in FBGEMM
    // supports it natively.
    if (ReluFused) {
      output.relu_();
    }
    return output;
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

class QLinearUnpackedDynamicFp16 final {
 public:
#ifdef USE_FBGEMM
  static at::Tensor run(
      at::Tensor input,
      const at::Tensor& weight,
      const at::Tensor& bias) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

    TORCH_CHECK(
        weight.dim() == 2,
        "The dimension of weight tensor should be equal to 2");

    auto packed_weight = PackedLinearWeightFp16::prepack(weight, bias);
    auto output = packed_weight->apply_dynamic(std::move(input));

    return output;
  }

  static at::Tensor meta(
      at::Tensor input,
      const at::Tensor& weight,
      const at::Tensor& bias) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        fbgemm::fbgemmSupportedCPU(), "Your CPU doesn't support FBGEMM.");

    TORCH_CHECK(
        weight.dim() == 2,
        "The dimension of weight tensor should be equal to 2");

    auto out_channel = weight.sym_sizes().vec()[0];
    auto out_sizes = input.sym_sizes().vec();
    out_sizes[out_sizes.size() - 1] = out_channel;

    return at::empty_symint(out_sizes, input.options());
  }
#else // USE_FBGEMM
  static at::Tensor run(
      at::Tensor /* input */,
      const at::Tensor& weight,
      const at::Tensor& bias) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }

  static at::Tensor meta(
      at::Tensor /* input */,
      const at::Tensor& weight,
      const at::Tensor& bias) {
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

at::Tensor wrapped_fbgemm_pack_gemm_matrix_fp16(const at::Tensor& weight) {
#ifdef USE_FBGEMM
  TORCH_CHECK(
      weight.dim() == 2,
      "fbgemm weight packing only packs matrices not vectors.");
  return at::native::fbgemm_pack_gemm_matrix_fp16(weight);
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor wrapped_fbgemm_pack_gemm_matrix_fp16_meta(const at::Tensor& weight) {
#ifdef USE_FBGEMM
  // Strictly speaking this is not correct. However we do not know the exact
  // size of the packed matrix as it's being maintained by the object itself,
  // therefore we return the view we have here.
  return at::empty({8}, weight.options().dtype(at::kByte));
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor wrapped_fbgemm_linear_fp16_weight(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, int64_t out_channel) {
#ifdef USE_FBGEMM
  return at::native::fbgemm_linear_fp16_weight(input, weight, bias);
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor wrapped_fbgemm_linear_fp16_weight_meta(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, int64_t out_channel) {
#ifdef USE_FBGEMM
  // For the meta function, we need users to provide the dimension explicitly
  // as we don't have access to the weight.
  auto out_sizes = input.sym_sizes().vec();
  if (out_channel == -1) {
    out_sizes.pop_back();
  } else {
    out_sizes.back() = out_channel;
  }
  return at::empty_symint(out_sizes, input.options());
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

class LinearDynamicFp16Onednn final {
 public:
  static Tensor run(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      std::optional<Tensor> bias) {
#if AT_MKLDNN_ENABLED()
    return linear_dynamic_fp16_with_onednn_weight(
        act, onednn_weight, bias, /*relu_fused*/false);
#endif
    TORCH_CHECK(false, "Unimplemented (linear_dynamic_fp16_with_onednn_weight)");
  }

  static Tensor run_relu(
      Tensor act, // int8 CPU tensor, not QTensor
      Tensor onednn_weight, // int8 tensor from MkldnnCPU
      std::optional<Tensor> bias) {
#if AT_MKLDNN_ENABLED()
    return linear_dynamic_fp16_with_onednn_weight(
        act, onednn_weight, bias, /*relu_fused*/true);
#endif
    TORCH_CHECK(false, "Unimplemented (linear_dynamic_fp16_with_onednn_weight)");
  }

};


TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  register_linear_params();
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic"),
      TORCH_FN(QLinearDynamicInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic"),
      TORCH_FN(QLinearDynamicInt8<true>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16"),
      TORCH_FN(QLinearDynamicFp16<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16_unpacked_weight"),
      TORCH_FN(QLinearUnpackedDynamicFp16::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic_fp16"),
      TORCH_FN(QLinearDynamicFp16<true>::run));
}

TORCH_LIBRARY_IMPL(quantized, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic_fp16_unpacked_weight"),
      TORCH_FN(QLinearUnpackedDynamicFp16::meta));
}

TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  register_linear_params();
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::linear_dynamic"),
      TORCH_FN(QLinearDynamicInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16"),
      wrapped_fbgemm_pack_gemm_matrix_fp16);
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_linear_fp16_weight"),
      wrapped_fbgemm_linear_fp16_weight);
}

TORCH_LIBRARY_IMPL(_quantized, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16"),
      wrapped_fbgemm_pack_gemm_matrix_fp16_meta);
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::wrapped_fbgemm_linear_fp16_weight"),
      wrapped_fbgemm_linear_fp16_weight_meta);
}

TORCH_LIBRARY_IMPL(onednn, MkldnnCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::linear_dynamic_fp16"),
      TORCH_FN(LinearDynamicFp16Onednn::run));
  m.impl(TORCH_SELECTIVE_NAME("onednn::linear_relu_dynamic_fp16"),
      TORCH_FN(LinearDynamicFp16Onednn::run_relu));
}
} // namespace
} // namespace at::native
