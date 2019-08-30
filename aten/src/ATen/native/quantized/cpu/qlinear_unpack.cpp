#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor operator()(at::Tensor packed_weight) {
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

      weight_origin = _empty_per_channel_affine_quantized_like(
          scales.toType(kDouble),
          zero_points.toType(kLong),
          {N, K},
          {0}, // The output channel axis is 0
          device(kCPU).dtype(kQInt8));
    }

    int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

    // packB->printPackedMatrix("packedB inside fbgemm_unpack
    // (QLinearUnpackWeightInt8): ");
    packB->unpack(weight_ptr_int8);

    return weight_origin;
  }
#else // USE_FBGEMM
  at::Tensor operator()(at::Tensor /* weight */
  ) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_linear_unpack(Tensor W_prepack) -> Tensor W_origin",
    c10::RegisterOperators::options().kernel<QLinearUnpackWeightInt8>(
        TensorTypeId::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
