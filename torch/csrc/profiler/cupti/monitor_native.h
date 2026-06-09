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

// Byte offset / size of one selected field within a user-defined record.
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

struct CompletedCuptiBuffer {
  uint8_t* ptr;
  size_t valid_size;
  uint64_t ctx;
  uint32_t stream;
  // The v2 user-defined record layout CUPTI reported for THIS buffer
  // (pBufferCompleteInfo->ppRecordLayouts), parsed at completion. Travels with
  // the buffer so the decoder parses it against the exact field selection
  // active when the buffer was filled -- no epochs, no shared layout state.
  // Empty for v1 (classic records carry no user-defined layout).
  std::vector<CuptiRecordLayout> layouts;
};

// Process-wide singleton. The Activity-API buffer callbacks carry no user data,
// so the callbacks reach the pool through a global; there is at most one CUPTI
// monitor per process.
class TORCH_API CuptiMonitorBuffers {
 public:
  static CuptiMonitorBuffers& get();

  void configure(size_t buffer_size);

  // CUPTI buffer-requested / buffer-completed callback bodies. GIL-free.
  void on_request(uint8_t** buffer, size_t* size, size_t* max_records);
  // Completion: parses the user-defined record layout from a CUPTI
  // CUpti_BufferCallbackCompleteInfo* (taken as void*) and enqueues the buffer
  // with that layout attached, so the decoder reads each buffer against its own
  // captured layout.
  void on_complete(
      void* complete_info,
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
  size_t buffer_size_ = 4UL * 1024 * 1024;
  size_t allocated_ = 0;
  bool shutdown_ = false;
};

// Parse a CUpti_BufferCallbackCompleteInfo* (taken as void*) into per-kind
// record layouts. ppRecordLayouts is indexed by activity kind, null for kinds
// without a user-defined layout. Returns empty if complete_info /
// ppRecordLayouts is null.
std::vector<CuptiRecordLayout> cuptiMonitorParseRecordLayouts(
    void* complete_info);

// Free functions matching the CUPTI subscriber-scoped (user-defined record)
// buffer-callback signatures, registered via cuptiActivityRegisterCallbacks_v2.
// The trailing info pointers (CUpti_BufferCallbackRequestInfo* /
// CUpti_BufferCallbackCompleteInfo*) are taken as void* to avoid a CUPTI header
// dependency. The completion callback does not receive CUcontext/streamId (they
// become selectable record fields), so completed buffers carry ctx and stream
// of 0; the record-layout descriptor in the complete info is parsed and
// attached to the completed buffer so the decoder can parse records after the
// callback.
TORCH_API void cuptiMonitorBufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* max_num_records,
    void* request_info);
TORCH_API void cuptiMonitorBufferCompleted(
    uint8_t* buffer,
    size_t size,
    size_t valid_size,
    void* complete_info);

} // namespace torch::profiler::impl
