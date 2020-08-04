#include <c10/cuda/CUDAFunctions.h>

namespace c10 {
namespace cuda {

DeviceIndex device_count() noexcept {
  int count;
  // NB: In the past, we were inconsistent about whether or not this reported
  // an error if there were driver problems are not.  Based on experience
  // interacting with users, it seems that people basically ~never want this
  // function to fail; it should just return zero if things are not working.
  // Oblige them.
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    // Clear out the error state, so we don't spuriously trigger someone else.
    // (This shouldn't really matter, since we won't be running very much CUDA
    // code in this regime.)
    cudaError_t last_err = cudaGetLastError();
    (void)last_err;
    return 0;
  }
  return static_cast<DeviceIndex>(count);
}

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
