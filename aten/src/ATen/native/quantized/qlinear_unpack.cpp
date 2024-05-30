/*
The dispatch registrations at the end of this file applies to fbgemm, qnnpack, and cudnn backends.
The correct unpack backend function is determined using runtime polymorphism through the packed_weight pointer,
which is of type intrusive_ptr<LinearPackedParamsBase> and points to either a PackedLinearWeightsQnnp,
PackedLinearWeights (Fbgemm), or PackedLinearWeightsCudnn at runtime, which all inherit from LinearPackedParamsBase.
The implementations for the unpack functions can be found in /cpu/LinearUnpackImpl.cpp, for fbgemm&qnnpack
and /cudnn/linear_unpack_impl.cpp, for cudnn.
*/
#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

int register_linear_params();

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8 final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

class QLinearUnpackWeightFp16 final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
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
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    TORCH_CHECK(false,
        "quantized.linear_unpack(Tensor) is unsupported! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
  }
};

class QLinearUnpackWeightFp16Legacy final {
 public:
  static std::tuple<at::Tensor, std::optional<Tensor>> run(
      const at::Tensor& packed_weight) {
    TORCH_CHECK(false,
        "quantized.linear_unpack(Tensor) is unsupported! Please "
        "upgrade your model to use the newer quantized.linear_"
        "unpack(LinearPackedParamsBase) overload");
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack.legacy"), TORCH_FN(QLinearUnpackWeightInt8Legacy::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16.legacy"), TORCH_FN(QLinearUnpackWeightFp16Legacy::run));
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack"), TORCH_FN(QLinearUnpackWeightInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_fp16"), TORCH_FN(QLinearUnpackWeightFp16::run));
}

} // namespace
} // namespace native
} // namespace at
