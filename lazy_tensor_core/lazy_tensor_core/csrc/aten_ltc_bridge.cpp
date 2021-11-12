#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"

#include <map>
#include <string>
#include <vector>

#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/torch_util.h"

namespace torch_lazy_tensors {
namespace bridge {
namespace {

// TODO(alanwaketan): Move it to the backend interface.
class AtenLtcDeviceMapper {
 public:
  static AtenLtcDeviceMapper* Get();

  size_t GetDeviceOrdinal(const torch::lazy::BackendDevice& device) const {
    auto it = devices_ordinals_.find(device);
    CHECK(it != devices_ordinals_.end()) << device;
    return it->second;
  }

  const torch::lazy::BackendDevice& GetDeviceFromOrdinal(size_t ordinal) const {
    return devices_.at(ordinal);
  }

 private:
  AtenLtcDeviceMapper() {
    for (auto& device_str :
         compiler::getBackend()->GetLocalDevices()) {
      // TODO(alanwaketan): The backend should do the mapping themselves, and construct the device accordingly.
      devices_.emplace_back();
      devices_ordinals_[devices_.back()] = devices_.size() - 1;
    }
  }

  std::vector<torch::lazy::BackendDevice> devices_;
  std::map<torch::lazy::BackendDevice, size_t> devices_ordinals_;
};

AtenLtcDeviceMapper* AtenLtcDeviceMapper::Get() {
  static AtenLtcDeviceMapper* device_mapper = new AtenLtcDeviceMapper();
  return device_mapper;
}

}  // namespace

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

torch::lazy::BackendDevice AtenDeviceToLtcDevice(const c10::Device& device) {
  CHECK_EQ(device.type(), at::kLazy) << device;
  // Ordinal doesn't make any sense currently given
  // distributed training/multi-device is not supported.
  int ordinal = device.has_index() ? device.index() : 0;
  return AtenLtcDeviceMapper::Get()->GetDeviceFromOrdinal(ordinal);
}

c10::Device LtcDeviceToAtenDevice(const torch::lazy::BackendDevice& device) {
  return c10::Device(at::kLazy,
                     AtenLtcDeviceMapper::Get()->GetDeviceOrdinal(device));
}

}  // namespace bridge
}  // namespace torch_lazy_tensors
