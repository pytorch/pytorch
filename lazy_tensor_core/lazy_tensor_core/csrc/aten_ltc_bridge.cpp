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

LTCTensorImpl* GetLtcTensorImpl(const at::Tensor& tensor) {
  return dynamic_cast<LTCTensorImpl*>(tensor.unsafeGetTensorImpl());
}

}  // namespace

c10::optional<LazyTensor> TryGetLtcTensor(const at::Tensor& tensor) {
  LTCTensorImpl* impl = GetLtcTensorImpl(tensor);
  if (impl == nullptr) {
    return c10::nullopt;
  }
  return impl->tensor();
}

bool IsLtcTensor(const at::Tensor& tensor) {
  return GetLtcTensorImpl(tensor) != nullptr;
}

LazyTensor GetLtcTensor(const at::Tensor& tensor) {
  auto xtensor = TryGetLtcTensor(tensor);
  CHECK(xtensor) << "Input tensor is not a lazy tensor: " << tensor.toString();
  return *xtensor;
}

std::vector<LazyTensor> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors) {
  std::vector<LazyTensor> ltc_tensors;
  ltc_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    ltc_tensors.push_back(bridge::GetLtcTensor(tensor));
  }
  return ltc_tensors;
}

LazyTensor GetOrCreateLtcTensor(const at::Tensor& tensor,
                                const torch::lazy::BackendDevice& device) {
  if (!tensor.defined()) {
    return LazyTensor();
  }
  auto xtensor = TryGetLtcTensor(tensor);
  return xtensor ? *xtensor : LazyTensor::Create(tensor, device);
}

LazyTensor GetOrCreateLtcTensor(const c10::optional<at::Tensor>& tensor,
                                const torch::lazy::BackendDevice& device) {
  if (!IsDefined(tensor)) {
    return LazyTensor();
  }
  auto xtensor = TryGetLtcTensor(*tensor);
  return xtensor ? *xtensor : LazyTensor::Create(*tensor, device);
}

LazyTensor GetLtcTensorOrCreateForWrappedNumber(const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ?
      GetOrCreateLtcTensor(tensor, device) : GetLtcTensor(tensor);
}

std::vector<at::Tensor> LtcCreateTensorList(const at::TensorList& tensors) {
  std::vector<at::Tensor> aten_ltc_tensors(tensors.size());
  std::vector<LazyTensor> ltc_tensors;
  // We need to separate out the defined tensors first, GetLtcTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (tensor.defined()) {
      auto xtensor = TryGetLtcTensor(tensor);
      if (xtensor) {
        to_translate[i] = true;
        ltc_tensors.push_back(*xtensor);
      } else {
        aten_ltc_tensors[i] = tensor;
      }
    }
  }
  auto defined_aten_ltc_tensors =
      LazyGraphExecutor::Get()->GetTensors(&ltc_tensors);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      aten_ltc_tensors[i] = std::move(defined_aten_ltc_tensors[defined_pos++]);
    }
  }
  return aten_ltc_tensors;
}

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::Tensor& tensor) {
  auto xtensor = TryGetLtcTensor(tensor);
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor->GetDevice();
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

at::Tensor AtenFromLtcTensor(LazyTensor ltc_tensor) {
  return ltc_tensor.is_null() ? at::Tensor()
                              : at::Tensor(c10::make_intrusive<LTCTensorImpl>(
                                    std::move(ltc_tensor)));
}

at::Tensor CreateLtcTensor(at::Tensor tensor,
                           const c10::optional<torch::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    LazyTensor ltc_tensor = LazyTensor::Create(std::move(tensor), *device);
    tensor = AtenFromLtcTensor(ltc_tensor);
  }
  return tensor;
}

}  // namespace bridge
}  // namespace torch_lazy_tensors
