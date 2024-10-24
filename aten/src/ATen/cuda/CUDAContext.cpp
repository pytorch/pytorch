#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>

#include <ATen/cuda/CUDAConfig.h>
#include <deque>
#include <vector>

namespace at::cuda {

namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
  num_gpus = c10::cuda::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::cuda::current_device();
  return getDeviceProperties(device);
}

cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", num_gpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(c10::DeviceIndex device, c10::DeviceIndex peer_device) {
  c10::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", num_gpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus, "peer_device=", static_cast<int>(peer_device), ", num_gpus=", num_gpus);
  int can_access = 0;
  AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getCUDADeviceAllocator() {
  return c10::cuda::CUDACachingAllocator::get();
}

} // namespace at::cuda
