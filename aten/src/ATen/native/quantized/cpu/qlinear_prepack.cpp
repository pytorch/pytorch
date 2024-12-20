#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_saturate_weight_to_fp16.h>
#include <ATen/ops/_saturate_weight_to_fp16_native.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <c10/util/irange.h>

#include <algorithm>
#include <utility>
#include <vector>

int register_linear_params();

#ifdef USE_FBGEMM
namespace {
// Calculate the column offsets.
// Note this includes the sum of the columns as well as the scalar term
// B_zero_point * K, whereas the row_offsets created by
// PackAWithQuantRowOffset is only the sum of the A rows.
void calc_col_offsets_transpose(
    int K,
    int N,
    const int8_t* Bint8,
    int32_t* B_zero_point,
    int32_t* col_offsets,
    c10::QScheme qtype) {
  for (const auto i : c10::irange(N)) {
    int32_t sum = 0;
    for (const auto j : c10::irange(K)) {
      sum += Bint8[i * K + j];
    }
    if (qtype == c10::kPerTensorAffine) {
      col_offsets[i] = sum - B_zero_point[0] * K;
    } else {
      col_offsets[i] = sum - B_zero_point[i] * K;
    }
  }
}
} // namespace

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeight::prepack(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    at::Tensor weight,
    std::optional<at::Tensor> bias) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack (fbgemm) should"
      " be 2-dimensional.");

  auto N = weight.size(0);
  auto K = weight.size(1);

  // TODO: contiguous is called for further JIT optimizations.
  auto weight_contig = weight.contiguous();
  const auto qtype = weight.qscheme();
  std::vector<int32_t> weight_zero_points_int32(1, 0);
  if (qtype == c10::kPerTensorAffine) {
    weight_zero_points_int32[0] = {static_cast<int32_t>(weight.q_zero_point())};
  } else if (qtype == c10::kPerChannelAffine) {
    weight_zero_points_int32.resize(N, 0);
    for (const auto i : c10::irange(N)) {
      weight_zero_points_int32[i] =
          weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  }
  std::vector<float> weight_scales_float(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    weight_scales_float[0] = {static_cast<float>(weight.q_scale())};
  } else if (qtype == c10::kPerChannelAffine) {
    weight_scales_float.resize(N, 0.0);
    for (const auto i : c10::irange(N)) {
      weight_scales_float[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

  std::vector<int32_t> col_offsets(N);
  calc_col_offsets_transpose(
      /*K=*/static_cast<int>(K),
      /*N=*/static_cast<int>(N),
      /*Bint8=*/weight_ptr_int8,
      /*B_zero_point=*/weight_zero_points_int32.data(),
      /*col_offsets=*/col_offsets.data(),
      /*qtype=*/qtype);

  std::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == N,
        "bias should have N elements: " + std::to_string(N));
    bias_contig = bias->contiguous();
  }
  auto ret_ptr = c10::make_intrusive<PackedLinearWeight>(
      std::make_unique<fbgemm::PackBMatrix<int8_t>>(
          /*trans=*/fbgemm::matrix_op_t::Transpose,
          /*nRow=*/K,
          /*nCol=*/N,
          /*smat=*/weight_ptr_int8,
          /*ld=*/K,
          /*pmat=*/nullptr, // PackBMatrix manages ownership of pmat
          /*groups=*/1),
      bias_contig,
      col_offsets,
      weight_scales_float,
      weight_zero_points_int32,
      qtype);
  return ret_ptr;
}
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightsQnnp::prepack(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    at::Tensor weight,
    std::optional<at::Tensor> bias_in) {
  TORCH_CHECK(
      weight.dim() == 2,
      "quantized::linear_prepack (qnnpack): Weight tensor rank should be == 2");

  int64_t rows_w = weight.size(0);
  at::Tensor bias_fp32;
  if (bias_in.has_value()) {
    bias_fp32 = bias_in.value();
  } else {
    bias_fp32 = at::zeros(rows_w, weight.options().dtype(at::kFloat));
  }
  TORCH_CHECK(
      !bias_fp32.defined() ||
          (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == rows_w),
      "quantized::linear_prepack (qnnpack): Given weight of size ",
      weight.sizes(),
      ", expected bias to be 1-dimensional with ",
      rows_w,
      " elements",
      ", but got bias of size ",
      bias_fp32.sizes(),
      " instead");

  at::Tensor weight_contig = weight.contiguous();
  auto [w_zero_points, w_scales] =
      make_zero_points_and_scales_tensor(weight_contig);

  at::native::initQNNPACK();

  // We set the pre-packed linear weights to nullptr below as we call pre-pack
  // during the first invocation of operator run. Refer to Linear.cpp for more
  // details. TODO Update to actually call pre-pack here once bias is removed
  // from pre-packing step.
  auto wt_ptr = c10::make_intrusive<PackedLinearWeightsQnnp>(
      nullptr,
      weight_contig, /* int8_t weight */
      bias_fp32.contiguous(), /* fp32 bias */
      std::nullopt, /* input_scale */
      w_scales,
      std::move(w_zero_points));
  return wt_ptr;
}
#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightFp16::prepack(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    at::Tensor weight,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::optional<at::Tensor> bias) {

  weight = at::_saturate_weight_to_fp16(weight);

  const int64_t K = weight.size(1);
  const int64_t N = weight.size(0);
  at::Tensor weight_contig = weight.contiguous();
  float* weight_contig_ptr = weight_contig.data_ptr<float>();

  // TODO(mingzhe09088):
  // Consider using a functor here in PackedGemmMatrixFP16
  // Comments from (XQ): Not entirely sure this make_unique is safe.
  // make_unique is created with regular "new", and freed through
  // TypeMetaData::deleteFn in this function. This is perfectly fine if the
  // tensors are created and freed within this translation unit. It might be
  // very problematic if that tensor flows across dll boundaries.
  auto ptr = c10::make_intrusive<PackedLinearWeightFp16>(
      std::make_unique<fbgemm::PackedGemmMatrixFP16>(
          fbgemm::matrix_op_t::Transpose, K, N, 1, weight_contig_ptr),
      bias);
  return ptr;
}
#endif // USE_FBGEMM

#if AT_MKLDNN_ENABLED()
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightsOnednn::prepack(
    at::Tensor weight,
    std::optional<at::Tensor> bias) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack (onednn) should"
      " be 2-dimensional.");
  // Weight
  std::vector<int64_t> dims = weight.sizes().vec();
  auto N = weight.size(0);
  std::vector<int32_t> wgt_zero_points;
  ideep::scale_t wgt_scales;
  const auto qtype = weight.qscheme();
  if (qtype == c10::kPerTensorAffine) {
    TORCH_CHECK(
        weight.q_zero_point() == 0,
        "quantized::linear_prepack: ONEDNN only supports symmetric quantization of weight,"
        " whose zero point must be 0, but got ", weight.q_zero_point());
    wgt_zero_points = std::vector<int32_t>(1, weight.q_zero_point());
    wgt_scales = ideep::scale_t(1, 1.0/weight.q_scale()); // Scales of ONEDNN and PyTorch are reciprocal
  } else if (qtype == c10::kPerChannelAffine) {
    wgt_zero_points.resize(N);
    wgt_scales.resize(N);
    for (int i = 0; i < N; ++i) {
      wgt_zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
      TORCH_CHECK(
          wgt_zero_points[i] == 0,
          "quantized::linear_prepack: ONEDNN only supports symmetric quantization of weight,"
          " whose zero point must be 0, but got ",  wgt_zero_points[i], ", at index ", i);
      wgt_scales[i] = 1.0f / weight.q_per_channel_scales()[i].item<float>(); // Scales of ONEDNN and PyTorch are reciprocal
    }
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // Prepack weight
  auto weight_copy = weight.clone();
  ideep::tensor wgt = ideep::tensor({dims, dnnl::memory::data_type::s8}, weight_copy.data_ptr());
  wgt.transpose_(0, 1); // ONEDNN requires transposed weight
  auto src_dims = ideep::dims(); // Unknown when prepacking
  ideep::attr_t op_attr;
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
  auto w_desc = ideep::matmul_forward::expected_weights_desc(wgt.get_dims(), src_dims, dnnl::memory::data_type::s8,
                                                             dnnl::memory::data_type::u8, op_attr);
  ideep::tensor exp_wgt(w_desc);
  exp_wgt.feed_from(wgt);
  ideep::tensor * packed_weight_p = new ideep::tensor(std::move(exp_wgt));
  packed_weight_p->set_scale(wgt_scales);
  packed_weight_p->set_zero_point(wgt_zero_points);
  std::unique_ptr<ideep::tensor> weight_ptr(packed_weight_p);
  // Bias
  std::optional<ideep::tensor> onednn_bias{std::nullopt};
  if (bias.has_value()) {
    auto& b = bias.value();
    auto bias_size = b.sizes().vec();
    bias_size.insert(bias_size.begin(), 1);
    TORCH_CHECK(
        bias_size[1] == weight_ptr->get_dim(1),
        "bias should have N elements: ",
        std::to_string(weight_ptr->get_dim(1)),
        ", but got ", bias_size[1]);
    auto bias_desc = ideep::tensor::desc(bias_size, dnnl::memory::data_type::f32);
    ideep::tensor packed_bias;
    packed_bias.init(bias_desc, b.data_ptr());
    onednn_bias = std::optional<ideep::tensor>(packed_bias);
  }
  auto ret_ptr = c10::make_intrusive<PackedLinearWeightsOnednn>(
      PackedLinearWeightsOnednn{
        std::move(weight_ptr),
        onednn_bias,
        weight,
        bias});
  return ret_ptr;
}

inline at::Tensor pack_weight_to_onednn_tensor(
    const at::Tensor& weight,
    std::optional<torch::List<int64_t>>& input_shape) {
  std::vector<int64_t> w_dims = weight.sizes().vec();
  ideep::tensor wei = ideep::tensor({w_dims, dnnl::memory::data_type::s8}, weight.data_ptr());
  wei.transpose_(0, 1); // oneDNN requires transposed weight
  ideep::dims input_dims = input_shape.has_value() ? input_shape.value().vec() : ideep::dims();
  ideep::attr_t op_attr;
  op_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
  auto w_desc = ideep::matmul_forward::expected_weights_desc(
      wei.get_dims(), input_dims, dnnl::memory::data_type::s8, dnnl::memory::data_type::u8, op_attr);
  ideep::tensor expected_weight(w_desc);
  expected_weight.feed_from(wei);
  auto packed_weight = at::native::new_with_itensor_mkldnn(
      std::move(expected_weight),
      c10::optTypeMetaToScalarType(weight.options().dtype_opt()),
      weight.options().device_opt());
  return packed_weight;
}

inline at::Tensor pack_weight_to_fp16_onednn_tensor(
    at::Tensor& weight,
    std::optional<torch::List<int64_t>>& input_shape) {
  TORCH_CHECK(weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kFloat, "Weight should be of type float or float16");
  weight = weight.scalar_type() == at::kHalf ? weight : at::_saturate_weight_to_fp16(weight);
  std::vector<int64_t> w_dims = weight.sizes().vec();
  auto weight_fp16 = weight.to(at::kHalf);
  ideep::tensor wei = ideep::tensor({w_dims, dnnl::memory::data_type::f16}, weight_fp16.data_ptr());
  auto expected_weight = wei.transpose(0, 1); // oneDNN requires transposed weight
  // Onednn does not support f32f16f32 matmul, so we need to convert weight to f32 before compute
  // Therefore, we just return weight in plain format
  auto packed_weight = at::native::new_with_itensor_mkldnn(
      std::move(expected_weight),
      c10::kHalf,
      weight.options().device_opt());
  return packed_weight;
}

#endif // #if AT_MKLDNN_ENABLED()

namespace at::native {

at::Tensor _saturate_weight_to_fp16(const Tensor& weight) {
  Tensor weight_contig = weight.contiguous();
  float* weight_contig_ptr = weight_contig.data_ptr<float>();
  quant_utils::HandleWeightsSaturation(weight.size(0) * weight.size(1), weight_contig_ptr);
  return weight;
}

template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

template <class... Args>
inline std::vector<c10::IValue> callOpByHandle(
    const c10::OperatorHandle& op,
    Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  c10::Dispatcher::singleton().callBoxed(op, &stack);
  return stack;
}

template <class... Args>
inline std::vector<c10::IValue> callOpByName(
    const char* func_name,
    const char* overload_name,
    Args... args) {
  const std::optional<c10::OperatorHandle> op_handle =
      c10::Dispatcher::singleton().findSchema({func_name, overload_name});
  assert(op_handle.has_value());
  return callOpByHandle(op_handle.value(), std::forward<Args>(args)...);
}

at::Tensor wrapped_quantized_linear(
    at::Tensor input,
    const at::Tensor& input_scale,
    const at::Tensor& input_zero_point,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_zero_point,
    const at::Tensor& bias,
    const at::Tensor& output_scale,
    const at::Tensor& output_zero_point,
    [[maybe_unused]] const int64_t out_channel);

at::Tensor wrapped_quantized_linear(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    at::Tensor input,
    const at::Tensor& input_scale,
    const at::Tensor& input_zero_point,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_zero_point,
    const at::Tensor& bias,
    const at::Tensor& output_scale,
    const at::Tensor& output_zero_point,
    [[maybe_unused]] const int64_t out_channel) {
  //This op does four things:
  // 1. Use quantize_per_tensor to quantize the input
  // 2. Use quantized::linear_prepack to prepack the weight and bias
  // 3. Use quantized::linear to do the int8 linear quantized computation
  // 4. Use dequantize to dequantize the result of quantized::linear
  // The reason we do this is because we want to have such wrapper op to
  // bypass the issue from torch.export
#ifdef USE_FBGEMM
  auto qw = at::quantize_per_tensor(
      weight, weight_scale, weight_zero_point, c10::ScalarType::QInt8);
  auto op = Dispatcher::singleton()
                .findSchemaOrThrow("quantized::linear_prepack", "")
                .typed<c10::intrusive_ptr<LinearPackedParamsBase>(
                    at::Tensor, std::optional<at::Tensor>)>();
  auto packed_params = op.call(qw, bias);

  auto qx = at::quantize_per_tensor(
      input, input_scale, input_zero_point, c10::ScalarType::QUInt8);

  const auto scale_val = output_scale.item().toFloat();
  const auto zero_point_val = output_zero_point.item().toLong();

  auto result = callOpByName(
      "quantized::linear", "", qx, packed_params, scale_val, zero_point_val);

  return at::dequantize(result[0].toTensor());
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor wrapped_quantized_linear_meta(
    at::Tensor input,
    [[maybe_unused]] const at::Tensor& input_scale,
    [[maybe_unused]] const at::Tensor& input_zero_point,
    const at::Tensor& weight,
    [[maybe_unused]] const at::Tensor& weight_scale,
    [[maybe_unused]] const at::Tensor& weight_zero_point,
    [[maybe_unused]] const at::Tensor& bias,
    [[maybe_unused]] const at::Tensor& output_scale,
    [[maybe_unused]] const at::Tensor& output_zero_point,
    [[maybe_unused]] const int64_t out_channel);

at::Tensor wrapped_quantized_linear_meta(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    at::Tensor input,
    [[maybe_unused]] const at::Tensor& input_scale,
    [[maybe_unused]] const at::Tensor& input_zero_point,
    const at::Tensor& weight,
    [[maybe_unused]] const at::Tensor& weight_scale,
    [[maybe_unused]] const at::Tensor& weight_zero_point,
    [[maybe_unused]] const at::Tensor& bias,
    [[maybe_unused]] const at::Tensor& output_scale,
    [[maybe_unused]] const at::Tensor& output_zero_point,
    [[maybe_unused]] const int64_t out_channel) {
#ifdef USE_FBGEMM
  const at::SymInt M = input.sym_size(0);
  const at::SymInt N = weight.sym_size(0);
  auto Y = at::empty_symint({M, N}, input.options().dtype(at::kFloat));
  return Y;
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor _wrapped_linear_prepack(const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_zero_point,
    const at::Tensor& bias);

at::Tensor _wrapped_linear_prepack(const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_zero_point,
    const at::Tensor& bias) {
  // This op does two things
  // 1. Use quantize_per_tensor to quantize the weight
  // 2. Use quantized::linear_prepack to prepack the weight and bias
  // The reason we do this is because we want to have such wrapper op to
  // save the quantized weight as constants for AOTI
#ifdef USE_FBGEMM
  TORCH_CHECK(
      weight.dim() == 2,
      "fbgemm weight packing only packs matrices not vectors.");
  auto qw = at::quantize_per_tensor(
      weight, weight_scale, weight_zero_point, c10::ScalarType::QInt8);

  auto op = Dispatcher::singleton()
                .findSchemaOrThrow("quantized::linear_prepack", "")
                .typed<c10::intrusive_ptr<LinearPackedParamsBase>(
                    at::Tensor, std::optional<at::Tensor>)>();
  auto packed_params = op.call(qw, bias);

  auto unique_ptr_wrapper =
      std::make_unique<decltype(packed_params)>(std::move(packed_params));
  auto ret = cpp_custom_type_hack::create(
      std::move(unique_ptr_wrapper), weight.options());
  return ret;
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor _wrapped_quantized_linear_prepacked(const at::Tensor& input, const at::Tensor& input_scale,
    const at::Tensor& input_zero_point,
    const at::Tensor& packed_weight,
    const at::Tensor& output_scale,
    const at::Tensor& output_zero_point,
    [[maybe_unused]] const int64_t out_channel);

at::Tensor _wrapped_quantized_linear_prepacked(const at::Tensor& input, const at::Tensor& input_scale,
    const at::Tensor& input_zero_point,
    const at::Tensor& packed_weight,
    const at::Tensor& output_scale,
    const at::Tensor& output_zero_point,
    [[maybe_unused]] const int64_t out_channel) {
  // This op is similar to wrapped_quantized_linear, but it takes the prepacked weight
#ifdef USE_FBGEMM
  auto qx = at::quantize_per_tensor(
      input, input_scale, input_zero_point, c10::ScalarType::QUInt8);
  const auto scale_val = output_scale.item().toFloat();
  const auto zero_point_val = output_zero_point.item().toLong();
  auto packed_weight_ptr =
      // @lint-ignore CLANGTIDY facebook-hte-Deprecated
      cpp_custom_type_hack::cast<c10::intrusive_ptr<LinearPackedParamsBase>>(
          packed_weight);
  auto result = callOpByName(
      "quantized::linear", "", qx, packed_weight_ptr, scale_val, zero_point_val);

  return at::dequantize(result[0].toTensor());
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor _wrapped_linear_prepack_meta(const at::Tensor& weight,
    [[maybe_unused]] const at::Tensor& weight_scale,
    [[maybe_unused]] const at::Tensor& weight_zero_point,
    [[maybe_unused]] const at::Tensor& bias);

at::Tensor _wrapped_linear_prepack_meta(const at::Tensor& weight,
    [[maybe_unused]] const at::Tensor& weight_scale,
    [[maybe_unused]] const at::Tensor& weight_zero_point,
    [[maybe_unused]] const at::Tensor& bias) {
#ifdef USE_FBGEMM
  TORCH_CHECK(
      weight.dim() == 2,
      "fbgemm weight packing only packs matrices not vectors.");
  const at::SymInt M = weight.sym_size(0);
  const at::SymInt N = weight.sym_size(1);
  auto Y = at::empty_symint({M, N}, weight.options().dtype(at::kFloat));
  return Y;
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

at::Tensor _wrapped_quantized_linear_prepacked_meta(const at::Tensor& input,
    [[maybe_unused]] const at::Tensor& input_scale,
    [[maybe_unused]] const at::Tensor& input_zero_point,
    [[maybe_unused]] const at::Tensor& packed_weight,
    [[maybe_unused]] const at::Tensor& output_scale,
    [[maybe_unused]] const at::Tensor& output_zero_point,
    const int64_t out_channel);

at::Tensor _wrapped_quantized_linear_prepacked_meta(const at::Tensor& input,
    [[maybe_unused]] const at::Tensor& input_scale,
    [[maybe_unused]] const at::Tensor& input_zero_point,
    [[maybe_unused]] const at::Tensor& packed_weight,
    [[maybe_unused]] const at::Tensor& output_scale,
    [[maybe_unused]] const at::Tensor& output_zero_point,
    const int64_t out_channel) {
#ifdef USE_FBGEMM
  auto out_sizes = input.sym_sizes().vec();
  TORCH_CHECK(
        out_sizes.size() == 2,
        "The dimension of weight tensor should be equal to 2");
  out_sizes[out_sizes.size() - 1] = out_channel;

  return at::empty_symint(out_sizes, input.options());
#else // USE_FBGEMM
  TORCH_CHECK(
      false, "This PyTorch installation was not built with FBGEMM operators");
#endif // USE_FBGEMM
}

namespace {

class QLinearPackWeightInt8 final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      std::optional<Tensor> bias) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      return PackedLinearWeight::prepack(std::move(weight), std::move(bias));
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return PackedLinearWeightsQnnp::prepack(
          std::move(weight), std::move(bias));
    }
#endif
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      return PackedLinearWeightsOnednn::prepack(std::move(weight), std::move(bias));
    }
#endif // #if AT_MKLDNN_ENABLED()
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack ",
        toString(ctx.qEngine()));
  }
};

class QLinearPackWeightFp16 final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      std::optional<Tensor> bias) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
    // temporarily convert weight back to fp32, needs to be fixed
    // after fbgemm fixes the interface for their prepacking op (take fp16 input0
    weight = weight.to(ScalarType::Float);
    if (ctx.qEngine() == at::QEngine::FBGEMM ||
        ctx.qEngine() == at::QEngine::X86) {
      return PackedLinearWeightFp16::prepack(
          std::move(weight), std::move(bias));
    }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          false,
          "quantized::linear_prepack_fp16 is currently "
          "not supported by QNNPACK");
    }
#endif // USE_PYTORCH_QNNPACK
#if AT_MKLDNN_ENABLED()
    if (ctx.qEngine() == at::QEngine::ONEDNN) {
      TORCH_CHECK(
          false,
          "quantized::linear_prepack_fp16 is currently "
          "not supported by ONEDNN");
    }
#endif // #if AT_MKLDNN_ENABLED()
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack_fp16 ",
        toString(ctx.qEngine()));
  }
};

class QLinearPackWeightInt8Legacy final {
 public:
  static Tensor run(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] at::Tensor weight,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] std::optional<Tensor> bias) {
    TORCH_CHECK(false,
        "This model uses an outdated version of quantized.linear_prepack. "
        "Please re-export your model using the newer definitions in torch.jit.quantized");
  }
};

class QLinearPackWeightFp16Legacy final {
 public:
  static Tensor run(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] at::Tensor weight,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] std::optional<Tensor> bias) {
    TORCH_CHECK(false,
        "This model uses an outdated version of quantized.linear_prepack_fp16. "
        "Please re-export your model using the newer definitions in torch.jit.quantized");
  }
};

class QLinearPackWeightInt8Onednn final {
 public:
  static at::Tensor run(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] at::Tensor weight, // Not QTensor
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] std::optional<torch::List<int64_t>> input_shape) {
#if AT_MKLDNN_ENABLED()
    return pack_weight_to_onednn_tensor(weight, input_shape);
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.");
#endif
  }
};

class QLinearPackWeightFp16Onednn final {
 public:
  static at::Tensor run(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] at::Tensor weight, // Not QTensor
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] std::optional<torch::List<int64_t>> input_shape) {
#if AT_MKLDNN_ENABLED()
    return pack_weight_to_fp16_onednn_tensor(weight, input_shape);
#else
    TORCH_CHECK(false, "Unimplemented as onednn is not available.");
#endif
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_legacy"), TORCH_FN(QLinearPackWeightInt8Legacy::run));
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8::run));
}

TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16::run));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_prepack_fp16_legacy"), TORCH_FN(QLinearPackWeightFp16Legacy::run));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::wrapped_quantized_linear"), TORCH_FN(wrapped_quantized_linear));
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::_wrapped_linear_prepack"),
      _wrapped_linear_prepack);
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::_wrapped_quantized_linear_prepacked"),
      _wrapped_quantized_linear_prepacked);
}

TORCH_LIBRARY_IMPL(_quantized, Meta, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::wrapped_quantized_linear"), TORCH_FN(wrapped_quantized_linear_meta));
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::_wrapped_linear_prepack"),
      _wrapped_linear_prepack_meta);
  m.impl(
      TORCH_SELECTIVE_NAME("_quantized::_wrapped_quantized_linear_prepacked"),
      _wrapped_quantized_linear_prepacked_meta);
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_prepack"), TORCH_FN(QLinearPackWeightInt8Onednn::run));
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::linear_prepack_fp16"), TORCH_FN(QLinearPackWeightFp16Onednn::run));
}

} // namespace
} // namespace at::native
