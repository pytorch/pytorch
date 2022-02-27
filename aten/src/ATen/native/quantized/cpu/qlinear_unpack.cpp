#include <ATen/ATen.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

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

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8 final {
 public:
  static std::tuple<at::Tensor, c10::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightFp16 final {
 public:
  static std::tuple<at::Tensor, c10::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    auto& ctx = at::globalContext();

    TORCH_CHECK(
        ctx.qEngine() != at::QEngine::QNNPACK,
        "quantized::linear_unpack_fp16 is currently "
        "not supported by QNNPACK");

    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightInt8Legacy final {
 public:
  static std::tuple<at::Tensor, c10::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    TORCH_WARN_ONCE(
        "quantized.linear_unpack(Tensor) is deprecated! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
    return cpp_custom_type_hack::cast<
               c10::intrusive_ptr<LinearPackedParamsBase>>(packed_weight)
        ->unpack();
  }
};

class QLinearUnpackWeightFp16Legacy final {
 public:
  static std::tuple<at::Tensor, c10::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    TORCH_WARN_ONCE(
        "quantized.linear_unpack(Tensor) is deprecated! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
    auto& ctx = at::globalContext();

    TORCH_CHECK(
        ctx.qEngine() != at::QEngine::QNNPACK,
        "quantized::linear_unpack_fp16 is currently "
        "not supported by QNNPACK");

    return cpp_custom_type_hack::cast<
               c10::intrusive_ptr<LinearPackedParamsBase>>(packed_weight)
        ->unpack();
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack.legacy"), TORCH_FN(QLinearUnpackWeightInt8Legacy::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16.legacy"), TORCH_FN(QLinearUnpackWeightFp16Legacy::run));
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack"), TORCH_FN(QLinearUnpackWeightInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16"), TORCH_FN(QLinearUnpackWeightFp16::run));
}

} // namespace
} // namespace native
} // namespace at
