#include <c10/cuda/CUDAFunctions.h>

namespace c10 {
namespace cuda {

DeviceIndex current_device() {
  int cur_device;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  C10_CUDA_CHECK(cudaSetDevice(static_cast<int>(device)));
}

}} // namespace c10::cuda
