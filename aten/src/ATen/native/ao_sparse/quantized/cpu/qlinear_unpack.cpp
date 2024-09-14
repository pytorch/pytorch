#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <torch/custom_class.h>

#include <ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#endif

namespace ao {
namespace sparse {
int register_linear_params();

#ifdef USE_FBGEMM

LinearPackedSerializationType PackedLinearWeight::unpack() {
  auto packW = w.get();

  const int64_t N = static_cast<int64_t>(packW->R);
  const int64_t K = static_cast<int64_t>(packW->C);

  at::Tensor weight_origin;
  if (q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K}, at::device(c10::kCPU).dtype(c10::kQInt8), w_scale[0], w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    at::Tensor scales = at::empty(
        {static_cast<long>(w_scale.size())},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(w_scale.begin(), w_scale.end(), scales.mutable_data_ptr<float>());

    at::Tensor zero_points = at::empty(
        {static_cast<long>(w_zp.size())},
        at::device(c10::kCPU).dtype(c10::kInt));
    std::copy(w_zp.begin(), w_zp.end(), zero_points.mutable_data_ptr<int>());

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales,
        zero_points,
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  packW->unpack(weight_ptr_int8);

  const std::vector<int64_t> block_pattern(
      {out_features_block_size_, in_features_block_size_});

  return std::make_tuple(std::move(weight_origin), bias_, block_pattern);
}

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

LinearPackedSerializationType PackedLinearWeightQnnp::unpack() {
  const int64_t N = static_cast<int64_t>(output_channels_);
  const int64_t K = static_cast<int64_t>(input_channels_);

  float* w_scales_ptr = w_scales_.data_ptr<float>();

  at::Tensor weight_origin;
  if (q_scheme_ == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K},
        at::device(c10::kCPU).dtype(c10::kQInt8),
        w_scales_ptr[0],
        w_zero_points_[0] - 128);
  } else if (q_scheme_ == c10::kPerChannelAffine) {
    at::Tensor scales = at::empty(
        {static_cast<long>(output_channels_)},
        at::device(c10::kCPU).dtype(c10::kFloat));
    std::copy(
        w_scales_ptr,
        w_scales_ptr + output_channels_,
        scales.mutable_data_ptr<float>());

    at::Tensor zero_points = at::empty(
        {static_cast<long>(output_channels_)},
        at::device(c10::kCPU).dtype(c10::kInt));
    std::transform(
        w_zero_points_.begin(),
        w_zero_points_.begin() + output_channels_,
        zero_points.mutable_data_ptr<int>(),
        [](uint8_t v) { return static_cast<int>(v) - 128; });

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales,
        zero_points,
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  bcsr_matrix_->unpack(
      weight_ptr_int8,
      output_channels_,
      input_channels_,
      w_zero_points_.data());

  std::vector<int64_t> block_pattern(
      {out_features_block_size_, in_features_block_size_});

  return std::make_tuple(
      std::move(weight_origin), bias_, std::move(block_pattern));
}

#endif // USE_FBGEMM

namespace {

class QLinearUnpackWeightInt8 final {
 public:
  static LinearPackedSerializationType run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

TORCH_LIBRARY_IMPL(sparse, CatchAll, m) {
  register_linear_params();
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_unpack"),
      TORCH_FN(QLinearUnpackWeightInt8::run));
}
}  // namespace
}}  // namespace ao::sparse
