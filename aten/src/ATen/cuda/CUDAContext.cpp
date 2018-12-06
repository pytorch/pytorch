#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.hpp>

#include <ATen/cuda/CUDAConfig.h>

namespace at { namespace cuda {

namespace {

static DeviceIndex num_gpus = -1;
static std::once_flag init_flag;
static std::deque<std::once_flag> device_flags;
static std::vector<cudaDeviceProp> device_properties;

static void initDevicePropertiesVector() {
  num_gpus = c10::cuda::device_count();
  device_flags.resize(num_gpus);
}

static void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop;
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::cuda::current_device();
  return getDeviceProperties(device);
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  std::call_once(init_flag, initDevicePropertiesVector);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

Allocator* getCUDADeviceAllocator() {
  return at::globalContext().getTHCState()->cudaDeviceAllocator;
}

/* Handles */
cusparseHandle_t getCurrentCUDASparseHandle() {
  return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
}

cublasHandle_t getCurrentCUDABlasHandle() {
  return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

} // namespace cuda

} // namespace at
