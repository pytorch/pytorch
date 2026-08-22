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
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
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
  // Like get_completed(), but waits at most `timeout` -- returns nullopt on
  // timeout as well as on shutdown. Lets the decode thread wake on a flush
  // cadence. Callers must release the GIL before calling this.
  std::optional<CompletedCuptiBuffer> get_completed_for(
      std::chrono::nanoseconds timeout);

  void return_buffer(uint8_t* ptr);
  size_t pending_count();
  size_t allocated_count();
  void shutdown();
  void reset();

 private:
  CuptiMonitorBuffers() = default;
  // Pop the front completed buffer (nullopt if none). Caller holds mutex_.
  std::optional<CompletedCuptiBuffer> pop_completed_locked();

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

// One accumulated column: the raw little-endian bytes of a single record field
// concatenated across all decoded records of its kind. field_size is the per-
// record width, so count == bytes.size() / field_size; Python views the bytes
// as the field's dtype at drain (the native side stays dtype-agnostic).
struct CuptiColumn {
  size_t field_size = 0;
  std::vector<uint8_t> bytes;
};

// A record layout's grouping key: its sorted (field_id, size) pairs. Records
// sharing this key share an identical column layout, so they accumulate
// together; a different key (a kind whose layout changed mid-session) groups
// separately, keeping every group's columns length-aligned. Used only as a map
// key -- never serialized -- so the sorted pairs are the key directly.
using CuptiLayoutKey = std::vector<std::pair<int, size_t>>;

// Native decode worker for the CUPTI monitor. Runs its own thread that pulls
// completed buffers from CuptiMonitorBuffers, iterates their records with
// CUPTI's own v2 record iterator (cuptiActivityGetNextRecord_v2, passed in as a
// function pointer so this file needs no libcupti link), and accumulates every
// field in each buffer's captured layout into per-(kind, field) columns -- all
// GIL-free. Python drains the accumulated columns periodically (one GIL touch),
// so the per-buffer decode never contends with the training thread.
//
// NUMERIC fields only for now (the timer's start/end/graph_node_id): each field
// is copied as raw bytes. const char* (string) fields would need a deref during
// decode and are a follow-up (the profiler's NAME field).
class TORCH_API CuptiMonitorDecoder {
 public:
  static CuptiMonitorDecoder& get();

  // The CUPTI subscriber handle and the address of
  // cuptiActivityGetNextRecord_v2 (both as integers from the Python side, which
  // owns the libcupti handle). fence_kind / fence_end_field identify the
  // SYNCHRONIZATION record + its END field so the decoder can track the max
  // sync timestamp for flush(sync) (the native analog of the old Python
  // _advance_decoded_clock); 0 disables it.
  //
  // When self_flush is true the decode thread also drives the periodic plain
  // flush itself -- calling cuptiActivityFlushAll(0) (completed buffers only)
  // through flush_fn (the function's address, passed from Python like
  // get_next_record_fn) every flush_period_ns -- so no separate flush thread is
  // needed. self_flush false disables it (the caller drives flush()).
  void configure(
      uintptr_t subscriber,
      uintptr_t get_next_record_fn,
      uint32_t fence_kind,
      int fence_end_field,
      bool self_flush,
      uint64_t flush_period_ns,
      uintptr_t flush_fn);

  void start();
  void stop();

  // Move out the accumulated column groups and reset, so the next drain covers
  // only records decoded after this call. Each group is one (kind,
  // field-layout): records are accumulated per layout signature, so a kind
  // whose record layout changed mid-session (e.g. a field-selection
  // reconfigure, or buffers in flight from before an observer enabled a field)
  // yields SEPARATE groups rather than a single set of mismatched-length
  // columns. Within a group every field column is length-aligned (one entry per
  // record), which the columnar consumers require.
  std::vector<std::pair<uint32_t, std::map<int, CuptiColumn>>> drain();

  // Max SYNCHRONIZATION-END timestamp decoded so far (CUPTI native clock);
  // flush(sync) waits on this reaching its fence point.
  uint64_t max_sync_ns() const {
    return max_sync_ns_.load();
  }

  // Cumulative buffers decoded and valid bytes processed since configure() (for
  // stats()).
  uint64_t buffers_decoded() const {
    return buffers_decoded_.load();
  }
  uint64_t valid_bytes() const {
    return valid_bytes_.load();
  }

 private:
  CuptiMonitorDecoder() = default;
  ~CuptiMonitorDecoder();
  void worker_loop();
  void decode_buffer(const CompletedCuptiBuffer& buf);

  std::mutex mutex_; // guards columns_
  // kind -> layout-key -> {field_id -> column}. Grouping by layout key keeps
  // records decoded against different layouts in separate, internally
  // length-consistent groups (see drain()).
  std::map<uint32_t, std::map<CuptiLayoutKey, std::map<int, CuptiColumn>>>
      columns_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> max_sync_ns_{0};
  std::atomic<uint64_t> buffers_decoded_{0};
  std::atomic<uint64_t> valid_bytes_{0};
  uintptr_t subscriber_ = 0;
  uintptr_t get_next_record_fn_ = 0;
  uint32_t fence_kind_ = 0;
  int fence_end_field_ = -1;
  // Whether the decode thread drives the periodic cuptiActivityFlushAll(0)
  // itself, the plain-flush cadence it uses, and the address of
  // cuptiActivityFlushAll (from Python, so this file needs no libcupti link).
  bool self_flush_ = false;
  uint64_t flush_period_ns_ = 0;
  uintptr_t flush_fn_ = 0;
};

// Process-global store of per-annotation metadata. A producer on the training
// thread -- e.g. an in-process NCCL profiler plugin -- puts a JSON object; it
// is keyed by the current external-correlation id on this thread's stack
// (cuptiMonitorCurrentExternalId), which the producer/caller pushed around the
// op
// -- so the producer never passes an id. Repeated puts under the same id MERGE
// (recursive object merge), so several producers can each contribute fields
// (including nested objects) for one op. The Python side drains it (folded into
// drain_decoded's return) and
// joins the metadata onto the monitor's kernel-timing records by id. Keyed only
// by external id -- graph_node_ids surface only retroactively in replayed
// kernel records, so node-id keying is a consumer-side cache the resolver
// builds at first replay. Mutex-guarded; put per op on the training thread,
// drain on the flushing thread. See the colltrace-replacement design.
class TORCH_API CuptiMetadataStore {
 public:
  static CuptiMetadataStore& get();

  // Recursively merge a JSON object into the metadata entry for external_id
  // (nested objects combine; on a leaf conflict the later put wins). No-op when
  // external_id is 0. Callers resolve which collective: the NCCL plugin passes
  // the id it read; the Python binding defaults a 0 to the most-recently-pushed
  // id. Several producers can each contribute fields to the same id (e.g. the
  // plugin's descriptor plus a backend's extra schema fields).
  void put_external(nlohmann::json blob, uint64_t external_id);

  // Move out the accumulated (merged) objects and reset, so the next drain
  // covers only ops recorded after this call (mirrors the decoder's drain).
  std::map<uint64_t, nlohmann::json> drain_external();

 private:
  CuptiMetadataStore() = default;

  std::mutex mutex_;
  std::map<uint64_t, nlohmann::json> by_external_;
};

// Parse a CUpti_BufferCallbackCompleteInfo* (taken as void*) into per-kind
// record layouts. ppRecordLayouts is indexed by activity kind, null for kinds
// without a user-defined layout. Returns empty if complete_info /
// ppRecordLayouts is null.
std::vector<CuptiRecordLayout> cuptiMonitorParseRecordLayouts(
    void* complete_info);

// Benchmark helper (benchmarks/profiler_benchmark/bench_decode.py): run the
// native per-buffer decode -- stride-walk the records and accumulate every
// field into per-(kind, field_id) byte columns, exactly as the decode worker
// does, but with a stride iterator instead of CUPTI's record iterator (so it
// runs on a synthetic buffer, no GPU/CUPTI) -- `iters` times over
// `buffer_addr`, returning the total wall-clock seconds. The columns are
// discarded each iter.
TORCH_API double cuptiMonitorBenchDecode(
    uintptr_t buffer_addr,
    size_t valid_size,
    const std::vector<CuptiRecordLayout>& layouts,
    size_t iters);

// Host-side mirror of CUPTI's per-thread external-correlation id stack. CUPTI
// offers push/pop but no peek, so the monitor records each push/pop here (next
// to the CUPTI call) to make the CURRENT id readable. An in-process consumer on
// the same thread -- e.g. an NCCL profiler plugin tagging a collective's
// metadata with the annotation the Python side pushed -- reads it via
// cuptiMonitorCurrentExternalId instead of managing its own ids. Thread-local:
// only valid on the pushing thread (collectives launch on the training thread,
// the same one that pushes). Ids are monotonic from 1, so 0 means "no id on
// this thread's stack".
TORCH_API void cuptiMonitorNoteExternalPush(uint64_t external_id);
TORCH_API uint64_t
cuptiMonitorNoteExternalPop(); // returns popped id, 0 if empty
TORCH_API uint64_t cuptiMonitorCurrentExternalId(); // top of stack, 0 if empty

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
