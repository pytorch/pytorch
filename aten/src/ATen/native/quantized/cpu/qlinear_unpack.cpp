#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/quantized/Quantizer.h>

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

    int32_t weight_zero_point_int32 = pack_ptr.w_zp;

    auto weight_origin = _empty_affine_quantized(
        {N, K},
        at::device(kCPU).dtype(kQInt8),
        pack_ptr.w_scale,
        weight_zero_point_int32);
    int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_origin.data<c10::qint8>());

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
        CPUTensorId()));

} // namespace
} // namespace native
} // namespace at
