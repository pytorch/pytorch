#include <torch/csrc/lazy/core/tensor_util.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/helpers.h>

#include <cstring>

namespace torch::lazy {

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (size_t i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
}

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<BackendDataPtr> data_handles,
    at::ScalarType dest_element_type) {
  std::vector<at::Tensor> tensors;
  for (const auto& handle : data_handles) {
    tensors.push_back(
        getBackend()->MakeTensorFromComputationData(handle, dest_element_type));
  }
  return tensors;
}

BackendDataPtr TensorToDataHandle(
    const at::Tensor& tensor,
    const BackendDevice& device) {
  return getBackend()->MakeComputationDataFromTensor(
      tensor, Shape(tensor.scalar_type(), tensor.sizes()), device);
}

std::vector<BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<BackendDevice>& devices) {
  TORCH_CHECK(tensors.size() == devices.size());
  std::vector<BackendDataPtr> result;
  result.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    result.push_back(TensorToDataHandle(tensors[i], devices[i]));
  }
  return result;
}

bool IsSpecialScalar(const at::Scalar& value) {
  if (FLAGS_torch_lazy_handle_special_scalars &&
      (value.isIntegral(false) || value.isFloatingPoint())) {
    if (FLAGS_torch_lazy_all_numbers_special_scalars) {
      return true;
    }
    double scalar_value = value.toDouble();
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

} // namespace torch::lazy
