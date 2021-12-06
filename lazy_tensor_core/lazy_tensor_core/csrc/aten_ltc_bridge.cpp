#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"

#include "lazy_tensor_core/csrc/tensor.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"

namespace torch_lazy_tensors {
namespace bridge {

c10::optional<torch::lazy::BackendDevice> GetBackendDevice(const at::Tensor& tensor) {
  if (auto lt = TryGetLtcTensor(tensor)) {
    return lt.GetDevice();
  }
  return c10::nullopt;
}

c10::optional<torch::lazy::BackendDevice> GetBackendDevice() {
  return c10::nullopt;
}

}  // namespace bridge
}  // namespace torch_lazy_tensors
