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

// Snapshot of a CUPTI v2 user-defined record layout. The layout is fixed by the
// field selection set at enable time (constant for the session), so it is
// captured once. byte offset / size of one selected field within a record:
struct CuptiRecordFieldLayout {
  int field_id;
  size_t offset;
  size_t size;
};

// Layout of one activity kind's user-defined record: total record size plus the
// selected fields. kind is the CUPTI activity-kind enum value.
struct CuptiRecordLayout {
  uint32_t kind;
  size_t record_size;
  std::vector<CuptiRecordFieldLayout> fields;
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

  // Capture the v2 user-defined record layout from a CUPTI
  // CUpti_BufferCallbackCompleteInfo* (taken as void*). Snapshots on the first
  // call and is a no-op afterwards; the layout is constant for a session and
  // the info pointer is only valid for the duration of the completion callback.
  void capture_layouts(void* complete_info);
  // Copy of the captured layouts for the Python decoder. Empty until a v2
  // buffer has completed.
  std::vector<CuptiRecordLayout> record_layouts();

 private:
  CuptiMonitorBuffers() = default;

  std::mutex mutex_;
  std::condition_variable cv_;
  // LIFO free list: reuse the warmest (most recently returned) buffer first.
  std::vector<uint8_t*> free_;
  std::vector<uint8_t*> all_; // every buffer ever allocated (for reset)
  std::deque<CompletedCuptiBuffer> completed_;
  size_t buffer_size_ = 4UL * 1024 * 1024;
  size_t allocated_ = 0;
  bool shutdown_ = false;
  std::vector<CuptiRecordLayout> record_layouts_;
  bool layouts_captured_ = false;
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

// Free functions matching the CUPTI v2 (subscriber-scoped, user-defined
// record) buffer-callback signatures, registered via
// cuptiActivityRegisterCallbacks_v2. They feed the same pool/queue as v1; only
// the CUPTI-side argument lists differ. The trailing info pointers
// (CUpti_BufferCallbackRequestInfo* / CUpti_BufferCallbackCompleteInfo*) are
// taken as void* to avoid a CUPTI v2 header dependency. v2 does not pass
// CUcontext/streamId to the completion callback (they become selectable record
// fields), so buffers completed on this path carry ctx and stream of 0; the
// record-layout descriptor in the complete info is snapshotted once via
// capture_layouts() so the decoder can parse records after the callback.
TORCH_API void cuptiMonitorBufferRequestedV2(
    uint8_t** buffer,
    size_t* size,
    size_t* max_num_records,
    void* request_info);
TORCH_API void cuptiMonitorBufferCompletedV2(
    uint8_t* buffer,
    size_t size,
    size_t valid_size,
    void* complete_info);

} // namespace torch::profiler::impl
