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

// TODO(whc) refactor this: we need to support non-zero default ordinal for torch/XLA.
torch::lazy::BackendDevice AtenDeviceToBackendDevice(const c10::Device& device) {
  CHECK_EQ(device.type(), at::kLazy) << device;
  int ordinal = device.has_index() ? device.index() : 0;
  return torch::lazy::BackendDevice(
      torch::lazy::getBackend()->GetDefaultDeviceType(), ordinal);
}

// TODO(whc) refactor this: we need to support non 1 on 1 mapping for torch/XLA.
c10::Device BackendDeviceToAtenDevice(const torch::lazy::BackendDevice& device) {
  return c10::Device(at::kLazy, device.ordinal());
}

}  // namespace bridge
}  // namespace torch_lazy_tensors
