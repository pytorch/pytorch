#include "CUDAUtils.hpp"

#include <c10d/private/CUDAUtils.hpp>

namespace c10d {

CUDAEvent CUDAEvent::create(unsigned int flags) {
  CUDAEvent event;
  C10D_CUDA_CHECK(cudaEventCreateWithFlags(&event.event_, flags));
  int current_device;
  C10D_CUDA_CHECK(cudaGetDevice(&current_device));
  event.setDevice(current_device);
  return event;
}

CUDAEvent::~CUDAEvent() {
  // cudaEventDestroy must run on the same device of the event,
  // otherwise it creates a context on default device as well.
  at::DeviceGuard guard(device_);
  if (event_ != nullptr) {
    C10D_CUDA_CHECK(cudaEventDestroy(event_));
  }
}

CUDAStream CUDAStream::create() {
  CUDAStream stream;
  stream.stream_ = THCStream_new(cudaStreamNonBlocking);
  return stream;
}

CUDAStream::~CUDAStream() {
  if (stream_ != nullptr) {
    THCStream_free(stream_);
    stream_ = nullptr;
  }
}

cudaStream_t CUDAStream::getStream() const {
  return THCStream_stream(stream_);
}

} // namespace c10d
