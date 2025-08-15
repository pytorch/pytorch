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

#ifdef USE_FBGEMM
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeight::unpack() {
  auto packB = w.get();

  int64_t N = static_cast<int64_t>(packB->numCols());
  int64_t K = static_cast<int64_t>(packB->numRows());

  at::Tensor weight_origin;
  if (q_scheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K}, at::device(c10::kCPU).dtype(c10::kQInt8), w_scale[0], w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), at::device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), at::device(c10::kCPU).dtype(c10::kInt));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales.toType(c10::kDouble),
        zero_points.toType(c10::kLong),
        0, // The output channel axis is 0
        at::device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  // packB->printPackedMatrix("packedB inside fbgemm_unpack
  // (QLinearUnpackWeightInt8): ");
  packB->unpack(weight_ptr_int8);

  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      weight_origin, bias_);
}
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightsQnnp::
    unpack() {
  if (orig_weight.defined()) {
    return std::tuple<at::Tensor, std::optional<at::Tensor>>(
        orig_weight, bias_);
  } else {
    // Unpacking requires reverting *make_zero_points_and_scales_tensor*
    // function in QnnpackUtils.h Please refer for a detail mechanism.
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/QnnpackUtils.h#L469
    // w_scales and w_zero_points are different from original scales & zero
    // points with padding & casting etc
    at::Tensor weight_origin;

    float* weight_scales_data = w_scales.data_ptr<float>();
    if (q_scheme == c10::kPerTensorAffine) {
      weight_origin = at::_empty_affine_quantized(
          weight_sizes,
          at::device(c10::kCPU).dtype(c10::kQInt8),
          static_cast<double>(weight_scales_data[0]),
          (int64_t)w_zero_points[0] - 128);
    } else if (q_scheme == c10::kPerChannelAffine) {
      auto scales = at::from_blob(
          weight_scales_data,
          w_scales.sizes()[0] - kPaddingChannels,
          at::device(c10::kCPU).dtype(c10::kFloat));

      at::Tensor zero_points = at::empty(
          static_cast<int64_t>(w_zero_points.size() - kPaddingChannels), at::device(c10::kCPU).dtype(c10::kLong));
      for (const auto i : c10::irange(zero_points.numel())) {
        zero_points[i] = ((int64_t)w_zero_points[i] - 128);
      }
      weight_origin = at::_empty_per_channel_affine_quantized(
                          weight_sizes,
                          scales,
                          zero_points.toType(c10::kLong),
                          0, // The output channel axis is 0
                          at::device(c10::kCPU).dtype(c10::kQInt8))
                          .contiguous();
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
    }
    int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());
    w->unpackWeights(w_zero_points.data(), weight_ptr_int8);
    // See for the subtraction 128
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/qlinear_dynamic.cpp#L319
    auto wt_numel = weight_origin.numel();
    for (const auto i : c10::irange(wt_numel)) {
      weight_ptr_int8[i] = (int8_t)(weight_ptr_int8[i] - 128);
    }

    return std::tuple<at::Tensor, std::optional<at::Tensor>>(
        weight_origin, bias_);
  }
}
#endif // USE_PYTORCH_QNNPACK

#ifdef USE_FBGEMM
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightFp16::
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
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedLinearWeightsOnednn::unpack() {
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      orig_weight_, orig_bias_);
}
#endif // #if AT_MKLDNN_ENABLED()
