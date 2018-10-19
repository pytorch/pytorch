#include "CUDAUtils.hpp"

#include <c10d/private/CUDAUtils.hpp>

namespace c10d {

CUDAEvent CUDAEvent::create(unsigned int flags) {
  int current_device;
  C10D_CUDA_CHECK(cudaGetDevice(&current_device));
  CUDAEvent event(nullptr, current_device);

  C10D_CUDA_CHECK(cudaEventCreateWithFlags(&event.event_, flags));
  return event;
}

CUDAEvent::~CUDAEvent() noexcept(false) {
  if (event_ != nullptr) {
    // cudaEventDestroy must run on the same device of the event,
    // otherwise it creates a context on default device as well.
    at::DeviceGuard guard(device_);

    C10D_CUDA_CHECK(cudaEventDestroy(event_));
  }
}

} // namespace c10d
