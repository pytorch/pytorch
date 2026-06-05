#pragma once

// GIL-free buffer plumbing for the experimental CUPTI monitor
// (torch.profiler._cupti_monitor).
//
// CUPTI invokes the Activity-API buffer-requested / buffer-completed callbacks
// synchronously while it holds internal locks (during record generation and
// flush). If those callbacks acquired the Python GIL, a thread that holds the
// GIL and then enters CUPTI/CUDA could deadlock against them (GIL <-> CUPTI
// lock inversion). So the callbacks here are pure C++: they only touch a
// mutex-guarded buffer pool + completed queue, never Python. The Python decode
// thread pulls completed buffers via get_completed() (with the GIL released
// while it blocks) and hands them back with return_buffer().
//
// Registration (cuptiActivityRegisterCallbacks) still happens on the Python
// side via ctypes using the callback addresses exposed in init.cpp; only the
// callback bodies are native. This keeps the design reusable for the
// per-subscriber _v2 Activity APIs: the pool/queue/callbacks are unchanged,
// only the registration call gains a subscriber handle.

#include <c10/macros/Macros.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace torch::profiler::impl {

struct CompletedCuptiBuffer {
  uint8_t* ptr;
  size_t valid_size;
  uint64_t ctx;
  uint32_t stream;
};

// Process-wide singleton. The v1 Activity-API buffer callbacks carry no user
// data, so the callbacks reach the pool through a global; there is at most one
// CUPTI monitor per process.
class TORCH_API CuptiMonitorBuffers {
 public:
  static CuptiMonitorBuffers& get();

  void configure(size_t buffer_size);

  // CUPTI buffer-requested / buffer-completed callback bodies. GIL-free.
  void on_request(uint8_t** buffer, size_t* size, size_t* max_records);
  void on_complete(
      uint64_t ctx,
      uint32_t stream,
      uint8_t* buffer,
      size_t size,
      size_t valid_size);

  // Block until a completed buffer is available or shutdown() is called.
  // Callers must release the GIL before calling this.
  std::optional<CompletedCuptiBuffer> get_completed();

  void return_buffer(uint8_t* ptr);
  size_t pending_count();
  size_t allocated_count();
  void shutdown();
  void reset();

 private:
  CuptiMonitorBuffers() = default;

  std::mutex mutex_;
  std::condition_variable cv_;
  // LIFO free list: reuse the warmest (most recently returned) buffer first.
  std::vector<uint8_t*> free_;
  std::vector<uint8_t*> all_; // every buffer ever allocated (for reset)
  std::deque<CompletedCuptiBuffer> completed_;
  size_t buffer_size_ = 4 * 1024 * 1024;
  size_t allocated_ = 0;
  bool shutdown_ = false;
};

// Free functions matching the CUPTI v1 buffer-callback signatures. CUcontext is
// an opaque pointer, taken as void* to avoid a CUPTI header dependency.
TORCH_API void cuptiMonitorBufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* max_num_records);
TORCH_API void cuptiMonitorBufferCompleted(
    void* context,
    uint32_t stream_id,
    uint8_t* buffer,
    size_t size,
    size_t valid_size);

} // namespace torch::profiler::impl
