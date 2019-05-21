#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/quantized/Quantizer.h>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedConvWeight);
#endif
} // namespace caffe2

namespace at {
namespace native {
namespace {
class QConvPackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  Tensor operator()(Tensor weight, int64_t groups) {
    TORCH_CHECK(
        weight.ndimension() == 4, "Weights are expected to have 4 dimensions");
    TORCH_CHECK(groups == 1, "Groupwise convolutions are not supported yet");
    // weights in RS(C/G)K format
    // matrix dimensions after im2col
    int NDim = weight.size(3) / groups;
    int KDim = weight.size(0) * weight.size(1) * groups * weight.size(2);
    auto weight_config = weight.contiguous();
    int weight_zero_point_int32 = weight.q_zero_point().toInt();
    TORCH_CHECK(
        weight_zero_point_int32 == 0,
        "Only symmetric quantization is supported for weights yet");
    const int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_config.data<c10::quint8>());

    std::vector<int32_t> col_offsets(NDim * groups);
    std::vector<int32_t> kernel{static_cast<int>(weight.size(0)),
                                static_cast<int>(weight.size(1))};
    std::vector<int8_t> weight_int8(KDim * NDim * groups);
    auto ret_ptr = guts::make_unique<PackedConvWeight>(
        PackedConvWeight{guts::make_unique<fbgemm::PackBMatrix<int8_t>>(
                             fbgemm::matrix_op_t::NoTranspose,
                             KDim,
                             NDim,
                             weight_ptr_int8,
                             NDim,
                             nullptr, // PackBMatrix manages ownership of pmat
                             groups),
                         col_offsets,
                         kernel,
                         weight.q_scale().toFloat(),
                         weight_zero_point_int32});
    // TODO: we will need to replace this with torchscript classes at a later
    // point.
    return cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
  }
#else // USE_FBGEMM
  Tensor operator()(
      Tensor, /* weight */
      int64_t /* groups */
  ) {
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_conv_prepack",
    c10::RegisterOperators::options()
    .kernel<QConvPackWeightInt8>()
    .dispatchKey(QuantizedCPUTensorId()));

} // namespace
} // namespace native
} // namespace at
