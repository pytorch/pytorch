
#include "lazy_tensors/computation_client/nnc_computation_client.h"

#include <ATen/Functions.h>

#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/sys_util.h"

namespace lazy_tensors {

at::DeviceType NNCComputationClient::HardwareDeviceType() {
  static auto device_type =
      sys_util::GetEnvBool("NNC_CUDA", false) ? at::kCUDA : at::kCPU;
  // The first CUDA usage could happen via lazy tensors. Initialize CUDA here to
  // account for that, at::scalar_tensor constructor triggers everything we
  // need.
  static c10::optional<at::Tensor> init_cuda =
      device_type == at::kCUDA ? c10::optional<at::Tensor>(at::scalar_tensor(
                                     0, at::TensorOptions().device(at::kCUDA)))
                               : c10::nullopt;
  return device_type;
}

}  // namespace lazy_tensors
