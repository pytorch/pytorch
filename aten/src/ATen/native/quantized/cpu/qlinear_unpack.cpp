#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  std::tuple<at::Tensor, c10::optional<Tensor>> fbgemm_linear_unpack(
      at::Tensor packed_weight) {
    // Pull out the PackBMatrix instance from the owning tensor.
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeight>(packed_weight);
    auto packB = pack_ptr.w.get();

    int64_t N = static_cast<int64_t>(packB->numCols());
    int64_t K = static_cast<int64_t>(packB->numRows());

    Tensor weight_origin;
    if (pack_ptr.q_scheme == kPerTensorAffine) {
      weight_origin = _empty_affine_quantized(
          {N, K},
          at::device(kCPU).dtype(kQInt8),
          pack_ptr.w_scale[0],
          pack_ptr.w_zp[0]);
    } else if (pack_ptr.q_scheme == kPerChannelAffine) {
      auto scales = from_blob(
          pack_ptr.w_scale.data(),
          pack_ptr.w_scale.size(),
          device(kCPU).dtype(kFloat));
      auto zero_points = from_blob(
          pack_ptr.w_zp.data(), pack_ptr.w_zp.size(), device(kCPU).dtype(kInt));

      weight_origin = _empty_per_channel_affine_quantized(
          {N, K},
          scales.toType(kDouble),
          zero_points.toType(kLong),
          0, // The output channel axis is 0
          device(kCPU).dtype(kQInt8));
    }

    int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

    // packB->printPackedMatrix("packedB inside fbgemm_unpack
    // (QLinearUnpackWeightInt8): ");
    packB->unpack(weight_ptr_int8);

    return std::tuple<at::Tensor, c10::optional<Tensor>>(
        weight_origin, pack_ptr.bias);
  }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> qnnpack_linear_unpack(
      at::Tensor packed_weight) {
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeightsQnnp>(packed_weight);
    return std::tuple<at::Tensor, c10::optional<Tensor>>(
        pack_ptr.orig_weight, pack_ptr.bias);
  }
#endif // USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> operator()(
      at::Tensor packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear_unpack(packed_weight);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_unpack(packed_weight);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_unpack ",
        toString(ctx.qEngine()));
  }
};

class QLinearUnpackWeightFp16 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  std::tuple<at::Tensor, c10::optional<Tensor>> fbgemm_linear_unpack(
      at::Tensor packed_weight) {
    // Pull out the PackBMatrix instance from the owning tensor.
    auto& packed_struct =
        cpp_custom_type_hack::cast<PackedLinearWeightFp16>(packed_weight);
    auto& packed_weight_ptr = packed_struct.w;
    auto& bias = packed_struct.bias;

    auto nrows = packed_weight_ptr->numRows();
    auto ncols = packed_weight_ptr->numCols();

    at::Tensor unpacked_weight =
        at::empty({ncols, nrows}, at::dtype(at::kHalf).memory_format(MemoryFormat::Contiguous));
    packed_weight_ptr->unpack(
        static_cast<fbgemm::float16*>(unpacked_weight.data_ptr()),
        fbgemm::matrix_op_t::Transpose);

    return std::make_tuple(unpacked_weight.to(at::kFloat), bias);
  }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> qnnpack_linear_unpack(
      at::Tensor packed_weight) {
    TORCH_CHECK(
        false,
        "quantized::linear_unpack_fp16 is currently "
        "not supported by QNNPACK");
  }
#endif // USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> operator()(
      at::Tensor packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear_unpack(packed_weight);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_unpack(packed_weight);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_unpack_fp16 ",
        toString(ctx.qEngine()));
  }
};

static auto registry =
    c10::RegisterOperators()
        .op("quantized::linear_unpack(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)",
            c10::RegisterOperators::options().kernel<QLinearUnpackWeightInt8>(
                DispatchKey::CPUTensorId))
        .op("quantized::linear_unpack_fp16(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)",
            c10::RegisterOperators::options().kernel<QLinearUnpackWeightFp16>(
                DispatchKey::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
