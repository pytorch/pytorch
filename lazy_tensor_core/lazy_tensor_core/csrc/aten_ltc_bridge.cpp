#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"

#include <map>
#include <string>
#include <vector>

#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"

namespace torch_lazy_tensors {
namespace bridge {
c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::Tensor& tensor) {
  auto xtensor = TryGetLtcTensor(tensor);
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor.GetDevice();
}

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::optional<c10::Device>& device) {
  if (!device) {
    return c10::nullopt;
  }
  if (device->type() != at::kLazy) {
    return c10::nullopt;
  }
  return AtenDeviceToLtcDevice(*device);
}

// TODO(whc) do we need 'orUseDefault?'
// or can we require that one of the input tensors is lazy?, and the caller
// should know if they are promoting a tensor to Lazy for the first time?

// Needed zero-arg version to make variadic template work
c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault() { 
  return torch::lazy::BackendDevice(compiler::getBackend()->GetDefaultDeviceType(), 0);
}
c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault(const at::TensorList& tensors) {
  for (const auto& tensor : tensors) {
    if (c10::optional<LazyTensor> lt = TryGetLtcTensor(tensor)) {
      return lt->GetDevice();
    }
  }
  return GetSameBackendDeviceOrUseDefault();
}

c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault(const at::Tensor& tensor) {
  if (c10::optional<LazyTensor> lt = TryGetLtcTensor(tensor)) {
    return lt->GetDevice();
  }
  return GetSameBackendDeviceOrUseDefault();
}

// TODO(whc) refactor this: we need to support non-zero default ordinal for torch/XLA
torch::lazy::BackendDevice AtenDeviceToLtcDevice(const c10::Device& device) {
  CHECK_EQ(device.type(), at::kLazy) << device;
  int ordinal = device.has_index() ? device.index() : 0;
  return torch::lazy::BackendDevice(compiler::getBackend()->GetDefaultDeviceType(), ordinal);
}

c10::Device LtcDeviceToAtenDevice(const torch::lazy::BackendDevice& device) {
  return c10::Device(at::kLazy, device.ordinal());
}

}  // namespace bridge
}  // namespace torch_lazy_tensors
