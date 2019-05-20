#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {
namespace {

class QFCUnpackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  at::Tensor operator()(at::Tensor packed_weight) {
    // Pull out the PackBMatrix instance from the owning tensor.
    auto& pack_ptr = cpp_custom_type_hack::cast<PackedFCWeight>(packed_weight);
    auto packB = pack_ptr.w.get();

    int64_t N = static_cast<int64_t>(packB->numCols());
    int64_t K = static_cast<int64_t>(packB->numRows());

    float weight_scale_float = pack_ptr.w_scale;
    int32_t weight_zero_point_int32 = pack_ptr.w_zp + 128;

    std::vector<int8_t> weight_int8(K * N);
    int8_t* weight_ptr_int8 = weight_int8.data();

    auto weight_origin = _empty_affine_quantized(
        {N, K},
        at::device(kCPU).dtype(kQUInt8),
        weight_scale_float,
        weight_zero_point_int32);
    uint8_t* weight_ptr_uint8 =
        reinterpret_cast<uint8_t*>(weight_origin.data<c10::quint8>());

    packB->unpack(weight_ptr_int8);
    convert_int8_uint8(K, N, weight_ptr_int8, weight_ptr_uint8);

    return weight_origin;
  }
#else // USE_FBGEMM
  at::Tensor operator()(at::Tensor /* weight */
  ) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    AT_ASSERTM(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_linear_unpack(Tensor W_prepack) -> Tensor W_origin",
    c10::RegisterOperators::options()
    .kernel<QFCUnpackWeightInt8>()
    .dispatchKey(CPUTensorId()));

} // namespace
} // namespace native
} // namespace at
