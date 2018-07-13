#include "CUDAUtils.hpp"

#include <c10d/private/CUDAUtils.hpp>

namespace c10d {

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
  int loPri, hiPri;
  C10D_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
  stream.stream_ = THCStream_newWithPriority(cudaStreamNonBlocking, hiPri);
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
