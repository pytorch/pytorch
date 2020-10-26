#pragma once

#include <aten/src/ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/custom_class.h>
/*namespace torch {
namespace jit {

class CUDAEvent;
class CUDAStream final : public CustomClassHolder {
public:
  CUDAStream(int64_t device=-1, int64_t priority=0) {
    stream_ = std::make_unique<c10::cuda::CUDAStream>(c10::cuda::getStreamFromPool(priority > 50, device));
  }

  bool query() {
    return stream_->query();
  }
  c10::intrusive_ptr<CUDAEvent> recordEvent(c10::intrusive_ptr<CUDAEvent> event);
  void synchronize() {
    stream_->synchronize();
  }
  void waitEvent(c10::intrusive_ptr<CUDAEvent> event);
  void waitStream(c10::intrusive_ptr<CUDAStream> stream);

private:
  std::unique_ptr<c10::cuda::CUDAStream> stream_;
  friend class CUDAEvent;
};

class CUDAEvent final : public CustomClassHolder {
public:
  CUDAEvent(bool enable_timing=false, bool blocking=false, bool interprocess=false) {
    int flags = cudaEventDisableTiming;
    if (enable_timing) {
      flags = cudaEventDefault;
    }
    if (blocking) {
      flags |= cudaEventBlockingSync;
    }
    if (interprocess) {
      TORCH_CHECK(!enable_timing);
      flags |= cudaEventInterprocess;
    }

    event_ = std::make_unique<at::cuda::CUDAEvent>(flags);
  }
  double elapsedTime(c10::intrusive_ptr<CUDAEvent> end) {
    return event_->elapsed_time(*end->event_);
  }
  std::string ipcHandle() {
    cudaIpcEventHandle_t handle;
    event_->ipc_handle(&handle);
    std::string str_handle((const char*)&handle, sizeof(handle));
    return str_handle;
  }
  bool query() {
    return event_->query();
  }
  void record(c10::intrusive_ptr<CUDAStream> stream);
  void synchronize() {
    event_->synchronize();
  }
  void wait(c10::intrusive_ptr<CUDAStream> stream);
private:
  void recordInternal(CUDAStream *stream);
  std::unique_ptr<at::cuda::CUDAEvent> event_;

  friend class CUDAStream;
};
}
}*/
