#include "caffe2/operators/enforce_finite_op.h"
#include <ATen/core/Tensor.h>

namespace caffe2 {
namespace detail {
void LogBlobFiniteness(Workspace *ws) {
  // This uses the aten interfaces to compute the sum and finiteness of the
  // tensors which are not present by default on xplat and mobile builds.
#if defined(EXPOSE_C2_OPS) ||                               \
  !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  for (const std::string& blob_name : ws->Blobs()) {
    try {
      const auto& blob = ws->GetBlob(blob_name);
      if (blob != nullptr && blob->IsType<Tensor>()) {
        Tensor* c2Tensor = blob->GetMutable<Tensor>();
        const at::Tensor& tensor = static_cast<at::Tensor>(*c2Tensor);
        bool blob_finite = tensor.sum().isfinite().cpu().data_ptr<bool>()[0];
        LOG(INFO) << "blob " << blob_name << " isfinite=" << (blob_finite ? "true" : "false");
      }
    } catch (const std::exception& ex) {
      LOG(ERROR) << "failed to check finiteness for " << blob_name << ": " << ex.what();
    }
  }
#endif
}
}

template <>
template <typename T>
bool EnforceFiniteOp<CPUContext>::DoRunWithType() {
  EnforceOnCPU<T>(Input(0));
  return true;
}

REGISTER_CPU_OPERATOR(EnforceFinite, EnforceFiniteOp<CPUContext>);

OPERATOR_SCHEMA(EnforceFinite)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Raise if there is NaN or Inf values in the input tensor.
)DOC")
    .Input(0, "input", "Input tensor");

SHOULD_NOT_DO_GRADIENT(EnforceFinite);

} // namespace caffe2
