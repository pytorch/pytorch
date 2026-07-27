#include <ATen/cuda/PeerToPeerAccess.h>

#include <ATen/Context.h>
#include <c10/cuda/PeerToPeerAccess.h>

namespace at::cuda {

bool get_p2p_access(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  // Ensure CUDA is lazily initialized before forwarding to c10
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  return c10::cuda::get_p2p_access(dev, dev_to_access);
}

bool get_fabric_access(c10::DeviceIndex dev) {
  // Ensure CUDA is lazily initialized before forwarding to c10
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  return c10::cuda::get_fabric_access(dev);
}

} // namespace at::cuda
