#include <ATen/ATen.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

torch::class_<LinearPackedParamsBase> register_linear_params();

#ifdef USE_FBGEMM


  * weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  // packB->printPackedMatrix("packedB inside fbgemm_unpack
  // (QLinearUnpackWeightInt8): ");


  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      weight_origin, bias_);
}
#endif // USE_FBGEMMint8_t

#ifdef USE_PYTORCH_QNNPACK

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8 final {
 public:
  
};

class QLinearUnpackWeightFp16 final {
 public:
  
};

class QLinearUnpackWeightInt8Legacy final {
 public:
};

class QLinearUnpackWeightFp16Legacy final {
 public:
  
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
