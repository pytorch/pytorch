#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/custom_class.h>

namespace torch::jit {

class CUDAEvent;
// This class is a wrapper around c10::cuda::CUDAStream.
// It is needed because TorchBind does not support all of the argument types
// for c10::cuda::CUDAStream. For more details, please refer to
// c10/cuda/CUDAStream.h.
class CUDAStream final : public CustomClassHolder {
 public:
  CUDAStream(
      std::optional<c10::Device> device = std::nullopt,
      int64_t priority = 0) {
    c10::DeviceIndex device_index =
        device.has_value() ? device->index() : c10::cuda::current_device();
    stream_ = std::make_unique<c10::cuda::CUDAStream>(
        c10::cuda::getStreamFromPool(static_cast<int>(priority), device_index));
  }

  CUDAStream(c10::cuda::CUDAStream s) {
    stream_ = std::make_unique<c10::cuda::CUDAStream>(s);
  }

  bool query() {
    return stream_->query();
  }

  c10::intrusive_ptr<CUDAEvent> recordEvent(
      c10::intrusive_ptr<CUDAEvent> event);

  void synchronize() {
    stream_->synchronize();
  }

  void waitEvent(const c10::intrusive_ptr<CUDAEvent>& event);

  void waitStream(const c10::intrusive_ptr<CUDAStream>& stream);

  /// Get the CUDA device index that this stream is associated with.
  int64_t device_index() const {
    return stream_->device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a CUDA device.
  c10::Device device() const {
    return stream_->device();
  }

  /// Return the stream ID corresponding to this particular stream.
  int64_t id() const {
    return stream_->id();
  }

 private:
  std::unique_ptr<c10::cuda::CUDAStream> stream_;
  friend class CUDAEvent;
};

// This class is a wrapper around at::cuda::CUDAStream.
// It is needed because TorchBind does not support all of the argument types
// for at::cuda::CUDAEvent. For more details, please refer to
// aten/src/ATen/cuda/CUDAEvent.h.
class CUDAEvent final : public CustomClassHolder {
 public:
  CUDAEvent(
      bool enable_timing = false,
      bool blocking = false,
      bool interprocess = false) {
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

  double elapsedTime(const c10::intrusive_ptr<CUDAEvent>& end) {
    return event_->elapsed_time(*end->event_);
  }

  std::string ipcHandle() {
    cudaIpcEventHandle_t handle{};
    event_->ipc_handle(&handle);
    std::string str_handle((const char*)&handle, sizeof(handle));
    return str_handle;
  }

  bool query() {
    return event_->query();
  }

  void record(const c10::intrusive_ptr<CUDAStream>& stream);

  void synchronize() {
    event_->synchronize();
  }
  void wait(const c10::intrusive_ptr<CUDAStream>& stream);

 private:
  void recordInternal(CUDAStream* stream);
  std::unique_ptr<at::cuda::CUDAEvent> event_;

  friend class CUDAStream;
};

inline c10::intrusive_ptr<CUDAEvent> CUDAStream::recordEvent(
    c10::intrusive_ptr<CUDAEvent> event) {
  if (!event) {
    event = c10::make_intrusive<CUDAEvent>();
  }

  event->recordInternal(this);
  return event;
}

inline void CUDAStream::waitEvent(const c10::intrusive_ptr<CUDAEvent>& event) {
  event->event_->block(*stream_);
}

inline void CUDAStream::waitStream(
    const c10::intrusive_ptr<CUDAStream>& stream) {
  auto ev = c10::make_intrusive<CUDAEvent>();
  stream->recordEvent(ev);
  waitEvent(ev);
}

inline void CUDAEvent::record(const c10::intrusive_ptr<CUDAStream>& stream) {
  event_->record(*stream->stream_);
}

inline void CUDAEvent::recordInternal(CUDAStream* stream) {
  event_->record(*stream->stream_);
}

inline void CUDAEvent::wait(const c10::intrusive_ptr<CUDAStream>& stream) {
  event_->block(*stream->stream_);
}

TORCH_LIBRARY(cuda, m) {
  auto stream_class = m.class_<torch::jit::CUDAStream>("Stream").def(
      torch::init<std::optional<c10::Device>, int64_t>(),
      "",
      {torch::arg("device") = std::nullopt, torch::arg("priority") = 0});
  auto event_class = m.class_<torch::jit::CUDAEvent>("Event").def(
      torch::init<bool, bool, bool>(),
      "",
      {torch::arg("enable_timing") = false,
       torch::arg("blocking") = false,
       torch::arg("interprocess") = false});

  stream_class.def("query", &CUDAStream::query)
      .def("record_event", &CUDAStream::recordEvent)
      .def("synchronize", &CUDAStream::synchronize)
      .def("wait_event", &CUDAStream::waitEvent)
      .def("wait_stream", &CUDAStream::waitStream)
      .def("device_index", &CUDAStream::device_index)
      .def_property("device", &CUDAStream::device)
      .def("id", &CUDAStream::id);

  event_class.def("elapsed_time", &CUDAEvent::elapsedTime)
      .def("query", &CUDAEvent::query)
      .def("record", &CUDAEvent::record)
      .def("synchronize", &CUDAEvent::synchronize)
      .def("wait", &CUDAEvent::wait);
};

} // namespace torch::jit
