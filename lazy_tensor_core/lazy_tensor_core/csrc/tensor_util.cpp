#include "lazy_tensor_core/csrc/tensor_util.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <torch/csrc/lazy/backend/backend_device.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <numeric>
#include <thread>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
}

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<compiler::BackendDataPtr> data_handles,
    at::ScalarType dest_element_type) {
  std::vector<at::Tensor> tensors;
  for (const auto& handle : data_handles) {
    tensors.push_back(compiler::getBackend()->MakeTensorFromComputationData(
        handle, dest_element_type));
  }
  return tensors;
}

compiler::BackendDataPtr TensorToDataHandle(const at::Tensor& tensor,
                                     const torch::lazy::BackendDevice& device) {
  return compiler::getBackend()
      ->MakeComputationDataFromTensor(
          tensor, torch::lazy::Shape(tensor.scalar_type(), tensor.sizes()),
          device);
}

std::vector<compiler::BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<torch::lazy::BackendDevice>& devices) {
  CHECK_EQ(tensors.size(), devices.size());
  std::vector<compiler::BackendDataPtr> result;
  result.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    result.push_back(TensorToDataHandle(tensors[i], devices[i]));
  }
  return result;
}

bool IsSpecialScalar(const at::Scalar& value) {
  static bool no_scalars =
      lazy_tensors::sys_util::GetEnvBool("NO_SPECIAL_SCALARS", false);
  if (!no_scalars && (value.isIntegral() || value.isFloatingPoint())) {
    double scalar_value = value.toDouble();
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

}  // namespace torch_lazy_tensors
