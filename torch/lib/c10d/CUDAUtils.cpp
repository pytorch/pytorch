#include "CUDAUtils.hpp"

#include "./private/CUDAUtils.hpp"

namespace c10d {

CUDADevice::CUDADevice(int device) {
  setDevice(device);
}

CUDADevice::~CUDADevice() {
  setDevice(originalDevice_);
}

void CUDADevice::setDevice(int device) {
  if (device >= 0) {
    if (originalDevice_ == -1) {
      C10D_CUDA_CHECK(cudaGetDevice(&originalDevice_));
      if (device != originalDevice_) {
        C10D_CUDA_CHECK(cudaSetDevice(device));
      }
    } else {
      C10D_CUDA_CHECK(cudaSetDevice(device));
    }
  }
}

CUDAEvent CUDAEvent::create(unsigned int flags) {
  CUDAEvent event;
  C10D_CUDA_CHECK(cudaEventCreateWithFlags(&event.event_, flags));
  return event;
}

CUDAEvent::~CUDAEvent() {
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
