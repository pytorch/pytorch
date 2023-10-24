#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
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
    at::Tensor weight,
    c10::optional<at::Tensor> bias) {
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
    weight_zero_points_int32[0] = weight.q_zero_point();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_zero_points_int32.resize(N, 0);
    for (const auto i : c10::irange(N)) {
      weight_zero_points_int32[i] =
          weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  }
  std::vector<float> weight_scales_float(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    weight_scales_float[0] = weight.q_scale();
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
      /*K=*/K,
      /*N=*/N,
      /*Bint8=*/weight_ptr_int8,
      /*B_zero_point=*/weight_zero_points_int32.data(),
      /*col_offsets=*/col_offsets.data(),
      /*qtype=*/qtype);

  c10::optional<at::Tensor> bias_contig;
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
    at::Tensor weight,
    c10::optional<at::Tensor> bias_in) {
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
  std::vector<uint8_t> w_zero_points;
  at::Tensor  w_scales;
  std::tie(w_zero_points, w_scales) =
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
      c10::nullopt, /* input_scale */
      w_scales,
      std::move(w_zero_points));
  return wt_ptr;
}
#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightFp16::prepack(
    at::Tensor weight,
    c10::optional<at::Tensor> bias) {

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
    c10::optional<at::Tensor> bias) {
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
  c10::optional<ideep::tensor> onednn_bias{c10::nullopt};
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
    onednn_bias = c10::optional<ideep::tensor>(packed_bias);
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
    c10::optional<torch::List<int64_t>>& input_shape) {
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
      optTypeMetaToScalarType(weight.options().dtype_opt()),
      weight.options().device_opt());
  return packed_weight;
}

#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {

at::Tensor _saturate_weight_to_fp16(const Tensor& weight) {
  Tensor weight_contig = weight.contiguous();
  float* weight_contig_ptr = weight_contig.data_ptr<float>();
  quant_utils::HandleWeightsSaturation(weight.size(0) * weight.size(1), weight_contig_ptr);
  return weight;
}

namespace {

class QLinearPackWeightInt8 final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      c10::optional<Tensor> bias) {
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
      c10::optional<Tensor> bias) {
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
  static Tensor run(at::Tensor weight, c10::optional<Tensor> bias) {
    TORCH_CHECK(false,
        "This model uses an outdated version of quantized.linear_prepack. "
        "Please re-export your model using the newer definitions in torch.jit.quantized");
  }
};

class QLinearPackWeightFp16Legacy final {
 public:
  static Tensor run(at::Tensor weight, c10::optional<Tensor> bias) {
    TORCH_CHECK(false,
        "This model uses an outdated version of quantized.linear_prepack_fp16. "
        "Please re-export your model using the newer definitions in torch.jit.quantized");
  }
};

class QLinearPackWeightInt8Onednn final {
 public:
  static at::Tensor run(
    at::Tensor weight, // Not QTensor
    c10::optional<torch::List<int64_t>> input_shape) {
#if AT_MKLDNN_ENABLED()
    return pack_weight_to_onednn_tensor(weight, input_shape);
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
}

TORCH_LIBRARY_IMPL(onednn, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("onednn::qlinear_prepack"), TORCH_FN(QLinearPackWeightInt8Onednn::run));
}

} // namespace
} // namespace native
} // namespace at
