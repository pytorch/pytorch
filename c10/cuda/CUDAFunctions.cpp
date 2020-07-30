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

std::pair<int32_t, std::string> driver_version() {
  int driver_version = -1;
  cudaError_t err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    std::string err_str = std::to_string(err) + " " + cudaGetErrorString(err);
    return {-1, err_str};
  }
  return {driver_version, {}};
}

void device_synchronize() {
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda
} // namespace c10
