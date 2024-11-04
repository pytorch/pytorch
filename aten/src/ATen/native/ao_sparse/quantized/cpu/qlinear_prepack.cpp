#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <c10/util/irange.h>
#include <torch/custom_class.h>

#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/zeros.h>
#endif

#include <algorithm>

namespace ao {
namespace sparse {

int register_linear_params();

#ifdef USE_FBGEMM
namespace {
// Calculate the column offsets.
// Note this includes the sum of the columns as well as the scalar term
// B_zero_point * K, whereas the row_offsets created by
// packing of activation is only the sum of the A rows.
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

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeight::
    prepack(
        const at::Tensor& weight,
        const std::optional<at::Tensor>& bias,
        const int64_t out_features_block_size,
        const int64_t in_features_block_size) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for ao::sparse::qlinear_prepack (fbgemm) should"
      " be 2-dimensional.");

  TORCH_CHECK(
      out_features_block_size == 1 && in_features_block_size == 4,
      "The out and in features block sizes for ao::sparse::qlinear_prepack",
      " (fbgemm) should be 1 and 4 respectively (got ", out_features_block_size,
      " and ", in_features_block_size, ")");

  auto N = weight.size(0);
  auto K = weight.size(1);

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
  TORCH_CHECK(
      std::all_of(
          weight_zero_points_int32.cbegin(),
          weight_zero_points_int32.cend(),
          [](int32_t i) { return i == 0; }),
      "zero point(s) should be 0 for the weight tensor of ao::sparse::qlinear op");
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

  std::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    const at::Tensor& bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == N,
        "bias should have N elements: " + std::to_string(N));
    bias_contig = bias->contiguous();
  }

  auto bcsr = fbgemm::fbgemmDenseToBCSR<int8_t>(N, K, weight_ptr_int8);
  auto ret_ptr = c10::make_intrusive<PackedLinearWeight>(
      std::move(bcsr),
      bias_contig,
      col_offsets,
      weight_scales_float,
      weight_zero_points_int32,
      qtype,
      out_features_block_size,
      in_features_block_size);
  return ret_ptr;
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightQnnp::
    prepack(
        const at::Tensor& weight,
        const std::optional<at::Tensor>& bias,
        const int64_t out_features_block_size,
        const int64_t in_features_block_size) {
  at::native::initQNNPACK();
  return c10::make_intrusive<PackedLinearWeightQnnp>(
      weight, bias, out_features_block_size, in_features_block_size);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
PackedLinearWeightQnnp::PackedLinearWeightQnnp(
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    const int64_t out_features_block_size,
    const int64_t in_features_block_size)
    : LinearPackedParamsBase(out_features_block_size, in_features_block_size),
      orig_bias_(bias),
      q_scheme_(weight.qscheme()),
      output_channels_(weight.size(0)),
      input_channels_(weight.size(1)) {
  TORCH_CHECK(
      weight.dim() == 2,
      "ao::sparse::qlinear (qnnpack): Weight tensor rank should be == 2");
  TORCH_CHECK(out_features_block_size > 0, "Row block size must be > 0.");
  TORCH_CHECK(in_features_block_size > 0, "Row block size must be > 0.");

  if (bias.has_value()) {
    bias_ = bias.value();
  } else {
    bias_ = at::zeros(output_channels_, weight.options().dtype(at::kFloat));
  }
  TORCH_CHECK(
      (bias_.ndimension() == 1 && bias_.size(0) == output_channels_),
      "ao::sparse::qlinear_prepack (qnnpack): Given weight of size ",
      weight.sizes(),
      ", expected bias to be 1-dimensional with ",
      output_channels_,
      " elements",
      ", but got bias of size ",
      bias_.sizes(),
      " instead");

  // Given bias is supposed to be 1 dim, it is already contiguous,
  // but the weight might be non-contiguous.
  at::Tensor weight_contig = weight.contiguous();

  std::tie(w_zero_points_, w_scales_) =
      make_zero_points_and_scales_tensor(weight_contig);
  const float* weight_scales_data = w_scales_.const_data_ptr<float>();
  at::Tensor qnnp_weight = at::_empty_affine_quantized(
      weight_contig.sizes(),
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      weight_scales_data[0],
      w_zero_points_[0]);
  auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
  auto wt_numel = weight_contig.numel();
  int8_t* w_data =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());
  for (const auto i : c10::irange(wt_numel)) {
    qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
  }
  bcsr_matrix_ = qnnpack::generateBlockCSRMatrix<uint32_t>(
      reinterpret_cast<uint8_t*>(qnnp_w_data),
      output_channels_,
      input_channels_,
      out_features_block_size,
      in_features_block_size,
      w_zero_points_.data());
}
#endif // USE_PYTORCH_QNNPACK

namespace {

class QLinearPackWeightInt8 final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      const at::Tensor& weight,
      const std::optional<at::Tensor>& bias,
      const int64_t out_features_block_size,
      const int64_t in_features_block_size) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return PackedLinearWeight::prepack(
          weight, bias, out_features_block_size, in_features_block_size);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return PackedLinearWeightQnnp::prepack(
          weight, bias, out_features_block_size, in_features_block_size);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation ao::sparse::qlinear_prepack ",
        toString(ctx.qEngine()));
  }
};

TORCH_LIBRARY_IMPL(sparse, QuantizedCPU, m) {
  register_linear_params();
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_prepack"),
      TORCH_FN(QLinearPackWeightInt8::run));
}
}  // namespace
}}  // namespace ao::sparse
