#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#endif

int register_linear_params();

#ifdef USE_FBGEMM
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeight::unpack() {
  auto packB = w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = static_cast<int64_t>(packB->numRows());

  at::Tensor weight_origin;
  if (q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K}, at::device(c10::kCPU).dtype(c10::kQInt8), w_scale[0], w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kInt));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales.toType(c10::kDouble),
        zero_points.toType(c10::kLong),
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  // packB->printPackedMatrix("packedB inside fbgemm_unpack
  // (QLinearUnpackWeightInt8): ");
  packB->unpack(weight_ptr_int8);

  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      weight_origin, bias_);
}
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeightsQnnp::
    unpack() {
  TORCH_CHECK(
      orig_weight.defined(),
      "Cannot unpack weights. "
      "Call at::globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(orig_weight, bias_);
}
#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeightFp16::
    unpack() {
  auto& packed_weight_ptr = w;

  auto nrows = packed_weight_ptr->numRows();
  auto ncols = packed_weight_ptr->numCols();

  at::Tensor unpacked_weight =
      at::empty({ncols, nrows}, at::kHalf, c10::MemoryFormat::Contiguous);
  packed_weight_ptr->unpack(
      static_cast<fbgemm::float16*>(unpacked_weight.data_ptr()),
      fbgemm::matrix_op_t::Transpose);

  return std::make_tuple(unpacked_weight.to(at::kFloat), bias_);
}
#endif // USE_FBGEMM

#if AT_MKLDNN_ENABLED()
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeightsOnednn::unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      orig_weight_, orig_bias_);
}
#endif // #if AT_MKLDNN_ENABLED()
