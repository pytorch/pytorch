#include <torch/csrc/profiler/cupti/monitor_native.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace torch::profiler::impl {

namespace {

// ABI mirrors of the CUPTI v2 user-defined-record structs (from
// cupti_activity.h, CUPTI >= 13.2). Mirrored here so this file needs no CUPTI
// v2 header; member order and types must match CUPTI exactly. complete_info is
// read by reinterpreting the void* the callback receives.
struct AbiFieldLayoutEntry {
  size_t structSize;
  int fieldId;
  size_t offset;
  size_t size;
  size_t alignment;
};
struct AbiRecordLayout {
  size_t structSize;
  AbiFieldLayoutEntry* pEntries;
  size_t numFields;
  size_t recordSize;
};
struct AbiBufferCompleteInfo {
  size_t structSize;
  uint64_t threadId;
  AbiRecordLayout** ppRecordLayouts;
  size_t numRecordLayouts;
};

// A record layout's grouping key: its sorted (field_id, size) pairs. Records
// sharing this key share an identical column layout, so they accumulate
// together; a different key (a mid-session field-selection change) groups
// separately, keeping every group's columns length-aligned. The sorted pairs
// are the key directly (it is only ever a map key), so there is no string form.
CuptiLayoutKey layoutKey(const CuptiRecordLayout& layout) {
  CuptiLayoutKey fields;
  fields.reserve(layout.fields.size());
  for (const auto& f : layout.fields) {
    fields.emplace_back(f.field_id, f.size);
  }
  std::ranges::sort(fields);
  return fields;
}

} // namespace

std::vector<CuptiRecordLayout> cuptiMonitorParseRecordLayouts(
    void* complete_info) {
  std::vector<CuptiRecordLayout> out;
  if (complete_info == nullptr) {
    return out;
  }
  const auto* info = static_cast<const AbiBufferCompleteInfo*>(complete_info);
  if (info->ppRecordLayouts == nullptr) {
    return out;
  }
  // ppRecordLayouts is indexed by activity kind; entries are null for kinds
  // without a user-defined layout.
  for (size_t kind = 0; kind < info->numRecordLayouts; ++kind) {
    const AbiRecordLayout* layout = info->ppRecordLayouts[kind];
    if (layout == nullptr) {
      continue;
    }
    CuptiRecordLayout parsed;
    parsed.kind = static_cast<uint32_t>(kind);
    parsed.record_size = layout->recordSize;
    parsed.fields.reserve(layout->numFields);
    for (size_t f = 0; f < layout->numFields; ++f) {
      const AbiFieldLayoutEntry& e = layout->pEntries[f];
      parsed.fields.push_back({e.fieldId, e.offset, e.size});
    }
    out.push_back(std::move(parsed));
  }
  return out;
}

CuptiMonitorBuffers& CuptiMonitorBuffers::get() {
  static CuptiMonitorBuffers instance;
  return instance;
}

void CuptiMonitorBuffers::configure(size_t buffer_size) {
  std::lock_guard<std::mutex> guard(mutex_);
  buffer_size_ = buffer_size;
  shutdown_ = false;
}

void CuptiMonitorBuffers::on_request(
    uint8_t** buffer,
    size_t* size,
    size_t* max_records) {
  uint8_t* buf = nullptr;
  size_t bytes = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    bytes = buffer_size_;
    if (!free_.empty()) {
      buf = free_.back();
      free_.pop_back();
    }
  }
  if (buf == nullptr) {
    // malloc gives alignment suitable for any type (>= CUPTI's 8-byte
    // requirement) and returns nullptr (not throwing) on OOM, which is the
    // right behavior inside a C callback. Tracked in all_ so reset() can free.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    buf = static_cast<uint8_t*>(std::malloc(bytes));
    std::lock_guard<std::mutex> guard(mutex_);
    all_.push_back(buf);
    ++allocated_;
  }
  *buffer = buf;
  *size = bytes;
  *max_records = 0;
}

void CuptiMonitorBuffers::on_complete(
    void* complete_info,
    uint8_t* buffer,
    size_t /*size*/,
    size_t valid_size) {
  // Parse the layout outside the lock (it only reads the CUPTI-owned struct),
  // then enqueue the buffer with its layout attached. The completion callback
  // delivers neither CUcontext nor streamId (they are selectable record
  // fields), so ctx/stream are 0.
  std::vector<CuptiRecordLayout> layouts =
      cuptiMonitorParseRecordLayouts(complete_info);
  {
    std::lock_guard<std::mutex> guard(mutex_);
    completed_.push_back({buffer, valid_size, 0, 0, std::move(layouts)});
  }
  cv_.notify_one();
}

std::optional<CompletedCuptiBuffer> CuptiMonitorBuffers::get_completed() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !completed_.empty() || shutdown_; });
  if (completed_.empty()) {
    return std::nullopt; // shutdown
  }
  CompletedCuptiBuffer buf = std::move(completed_.front());
  completed_.pop_front();
  return buf;
}

void CuptiMonitorBuffers::return_buffer(uint8_t* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  free_.push_back(ptr);
}

size_t CuptiMonitorBuffers::pending_count() {
  std::lock_guard<std::mutex> guard(mutex_);
  return completed_.size();
}

size_t CuptiMonitorBuffers::allocated_count() {
  std::lock_guard<std::mutex> guard(mutex_);
  return allocated_;
}

void CuptiMonitorBuffers::shutdown() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    shutdown_ = true;
  }
  cv_.notify_all();
}

void CuptiMonitorBuffers::reset() {
  std::lock_guard<std::mutex> guard(mutex_);
  completed_.clear();
  free_.clear();
  for (uint8_t* p : all_) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    std::free(p);
  }
  all_.clear();
  allocated_ = 0;
  shutdown_ = false;
}

// ---------------------------------------------------------------------------
// Native decode worker.

CuptiMonitorDecoder& CuptiMonitorDecoder::get() {
  static CuptiMonitorDecoder instance;
  return instance;
}

CuptiMonitorDecoder::~CuptiMonitorDecoder() {
  // The clean path is stop(), which joins the worker. If we reach destruction
  // with it still running (process exiting without stop()), wake it out of
  // get_completed() and join: leaving it parked on the pool's condvar would
  // both std::terminate (~thread on a joinable thread) and hang teardown when
  // the pool condvar is destroyed with a waiter. The pool outlives the decoder
  // (constructed first), so this is safe; the join is bounded -- the worker
  // drains and exits.
  running_.store(false);
  CuptiMonitorBuffers::get().shutdown();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void CuptiMonitorDecoder::configure(
    uintptr_t subscriber,
    uintptr_t get_next_record_fn,
    uint32_t fence_kind,
    int fence_end_field) {
  subscriber_ = subscriber;
  get_next_record_fn_ = get_next_record_fn;
  fence_kind_ = fence_kind;
  fence_end_field_ = fence_end_field;
  max_sync_ns_.store(0);
  buffers_decoded_.store(0);
  valid_bytes_.store(0);
}

void CuptiMonitorDecoder::set_cbid_filter(
    int cbid_field_id,
    std::unordered_map<uint32_t, std::pair<bool, std::unordered_set<uint32_t>>>
        filters) {
  cbid_field_id_ = cbid_field_id;
  cbid_filters_ = std::move(filters);
}

void CuptiMonitorDecoder::start() {
  if (running_.exchange(true)) {
    return; // already running
  }
  thread_ = std::thread(&CuptiMonitorDecoder::worker_loop, this);
}

void CuptiMonitorDecoder::stop() {
  if (!running_.exchange(false)) {
    return; // not running
  }
  // Wake the worker if it is blocked in get_completed().
  CuptiMonitorBuffers::get().shutdown();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void CuptiMonitorDecoder::worker_loop() {
  // get_completed() blocks until a buffer is ready and returns nullopt only
  // once the pool is shut down AND drained, so on stop() the worker decodes
  // every already-completed buffer (incl. the fence's trailing flush) before
  // exiting.
  while (true) {
    std::optional<CompletedCuptiBuffer> buf =
        CuptiMonitorBuffers::get().get_completed();
    if (!buf.has_value()) {
      break; // shut down and drained
    }
    decode_buffer(*buf);
    CuptiMonitorBuffers::get().return_buffer(buf->ptr);
  }
}

void CuptiMonitorDecoder::decode_buffer(const CompletedCuptiBuffer& buf) {
  if (get_next_record_fn_ == 0 || buf.valid_size == 0) {
    return;
  }
  buffers_decoded_.fetch_add(1);
  valid_bytes_.fetch_add(buf.valid_size);
  // cuptiActivityGetNextRecord_v2(subscriber, buffer, validSize, &record).
  // Called via the address Python passed (libcupti is loaded Python-side), so
  // this TU needs no CUPTI header/link. CUPTI_SUCCESS == 0; any other status
  // (MAX_LIMIT_REACHED at end-of-buffer, INVALID_KIND, real errors) ends the
  // buffer.
  using GetNextRecordFn = int (*)(void*, uint8_t*, size_t, void**);
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto get_next = reinterpret_cast<GetNextRecordFn>(get_next_record_fn_);
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* subscriber = reinterpret_cast<void*>(subscriber_);

  // Accumulate locally (no lock during the record walk), then merge once.
  // Grouped by kind -> layout signature so records sharing a layout stay
  // together and fields never end up mismatched in length. Within this buffer a
  // kind has a single layout (ppRecordLayouts is per kind), so cache its
  // signature + layout.
  std::map<uint32_t, std::map<CuptiLayoutKey, std::map<int, CuptiColumn>>>
      local;
  std::map<uint32_t, const CuptiRecordLayout*> layout_by_kind;
  std::map<uint32_t, CuptiLayoutKey> key_by_kind;
  // Per-kind cbid field offset for the noisy-cbid filter (-1 = kind has no cbid
  // field).
  std::map<uint32_t, int64_t> cbid_off_by_kind;
  uint64_t local_max_sync = 0;
  void* record = nullptr;
  while (get_next(subscriber, buf.ptr, buf.valid_size, &record) == 0) {
    const auto* rec = static_cast<const uint8_t*>(record);
    uint32_t kind = *reinterpret_cast<const uint32_t*>(rec); // KIND @ offset 0
    auto cached = layout_by_kind.find(kind);
    const CuptiRecordLayout* layout = nullptr;
    if (cached == layout_by_kind.end()) {
      for (const auto& cand : buf.layouts) {
        if (cand.kind == kind) {
          layout = &cand;
          break;
        }
      }
      layout_by_kind[kind] = layout;
      key_by_kind[kind] =
          (layout != nullptr) ? layoutKey(*layout) : CuptiLayoutKey{};
    } else {
      layout = cached->second;
    }
    if (layout == nullptr) {
      continue; // no captured layout for this kind; skip
    }
    // Drop noisy runtime/driver records by cbid (see set_cbid_filter) so they
    // never reach the columns: the runtime blocklist / driver allowlist,
    // applied here because CUPTI's own per-cbid filter is NOT_COMPATIBLE under
    // user-defined records.
    if (!cbid_filters_.empty()) {
      auto filt = cbid_filters_.find(kind);
      if (filt != cbid_filters_.end()) {
        auto offit = cbid_off_by_kind.find(kind);
        int64_t cbid_off = -1;
        if (offit == cbid_off_by_kind.end()) {
          for (const auto& f : layout->fields) {
            if (f.field_id == cbid_field_id_) {
              cbid_off = static_cast<int64_t>(f.offset);
              break;
            }
          }
          cbid_off_by_kind[kind] = cbid_off;
        } else {
          cbid_off = offit->second;
        }
        if (cbid_off >= 0) {
          uint32_t cbid = 0;
          std::memcpy(&cbid, rec + cbid_off, sizeof(uint32_t));
          const bool in_set = filt->second.second.contains(cbid);
          const bool keep = filt->second.first ? in_set : !in_set;
          if (!keep) {
            continue; // noisy record: drop before accumulating its columns
          }
        }
      }
    }
    auto& kind_cols = local[kind][key_by_kind[kind]];
    for (const auto& field : layout->fields) {
      CuptiColumn& col = kind_cols[field.field_id];
      col.field_size = field.size;
      const uint8_t* src = rec + field.offset;
      col.bytes.insert(col.bytes.end(), src, src + field.size);
      // Track the fence (SYNCHRONIZATION-END) timestamp for flush(sync).
      if (kind == fence_kind_ && field.field_id == fence_end_field_ &&
          field.size == sizeof(uint64_t)) {
        uint64_t end_ns = 0;
        std::memcpy(&end_ns, src, sizeof(uint64_t));
        if (end_ns > local_max_sync) {
          local_max_sync = end_ns;
        }
      }
    }
  }

  if (local_max_sync > 0) {
    uint64_t prev = max_sync_ns_.load();
    while (local_max_sync > prev &&
           !max_sync_ns_.compare_exchange_weak(prev, local_max_sync)) {
    }
  }

  std::lock_guard<std::mutex> guard(mutex_);
  for (auto& [kind, by_sig] : local) {
    auto& dst_kind = columns_[kind];
    for (auto& [sig, kind_cols] : by_sig) {
      auto& dst = dst_kind[sig];
      for (auto& [field_id, col] : kind_cols) {
        CuptiColumn& d = dst[field_id];
        d.field_size = col.field_size;
        d.bytes.insert(d.bytes.end(), col.bytes.begin(), col.bytes.end());
      }
    }
  }
}

std::vector<std::pair<uint32_t, std::map<int, CuptiColumn>>>
CuptiMonitorDecoder::drain() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<std::pair<uint32_t, std::map<int, CuptiColumn>>> out;
  for (auto& [kind, by_sig] : columns_) {
    for (auto& [sig, kind_cols] : by_sig) {
      out.emplace_back(kind, std::move(kind_cols));
    }
  }
  columns_.clear();
  return out;
}

double cuptiMonitorBenchDecode(
    uintptr_t buffer_addr,
    size_t valid_size,
    const std::vector<CuptiRecordLayout>& layouts,
    size_t iters) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  const auto* base = reinterpret_cast<const uint8_t*>(buffer_addr);
  std::map<uint32_t, const CuptiRecordLayout*> by_kind;
  for (const auto& l : layouts) {
    by_kind[l.kind] = &l;
  }
  auto t0 = std::chrono::steady_clock::now();
  for (size_t it = 0; it < iters; ++it) {
    // Fresh columns each iter, matching decode_buffer's per-buffer `local`.
    std::map<uint32_t, std::map<int, CuptiColumn>> cols;
    size_t pos = 0;
    while (pos + sizeof(uint32_t) <= valid_size) {
      uint32_t kind = *reinterpret_cast<const uint32_t*>(base + pos);
      auto found = by_kind.find(kind);
      if (found == by_kind.end()) {
        break; // unknown kind: can't size it, stop (matches the Python walk)
      }
      const CuptiRecordLayout* layout = found->second;
      if (pos + layout->record_size > valid_size) {
        break; // trailing partial record
      }
      auto& kind_cols = cols[kind];
      for (const auto& field : layout->fields) {
        CuptiColumn& col = kind_cols[field.field_id];
        col.field_size = field.size;
        const uint8_t* src = base + pos + field.offset;
        col.bytes.insert(col.bytes.end(), src, src + field.size);
      }
      pos += layout->record_size;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

namespace {
// Per-thread mirror of CUPTI's external-correlation id stack (CUPTI has no
// peek).
thread_local std::vector<uint64_t> g_external_id_stack;
} // namespace

void cuptiMonitorNoteExternalPush(uint64_t external_id) {
  g_external_id_stack.push_back(external_id);
}

uint64_t cuptiMonitorNoteExternalPop() {
  if (g_external_id_stack.empty()) {
    return 0;
  }
  uint64_t id = g_external_id_stack.back();
  g_external_id_stack.pop_back();
  return id;
}

uint64_t cuptiMonitorCurrentExternalId() {
  return g_external_id_stack.empty() ? 0 : g_external_id_stack.back();
}

CuptiMetadataStore& CuptiMetadataStore::get() {
  static CuptiMetadataStore instance;
  return instance;
}

void CuptiMetadataStore::put_external(
    nlohmann::json blob,
    uint64_t external_id) {
  if (external_id == 0) {
    return; // no id to key it by (caller resolves which collective)
  }
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = by_external_.find(external_id);
  if (it == by_external_.end()) {
    by_external_.emplace(external_id, std::move(blob));
  } else {
    // Recursive merge so several producers can each contribute fields (incl.
    // nested objects) for one op; on a leaf conflict, the later put wins.
    it->second.update(blob, /*merge_objects=*/true);
  }
}

std::map<uint64_t, nlohmann::json> CuptiMetadataStore::drain_external() {
  std::lock_guard<std::mutex> guard(mutex_);
  std::map<uint64_t, nlohmann::json> out;
  out.swap(by_external_);
  return out;
}

void cuptiMonitorBufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* max_num_records,
    void* /*request_info*/) {
  CuptiMonitorBuffers::get().on_request(buffer, size, max_num_records);
}

void cuptiMonitorBufferCompleted(
    uint8_t* buffer,
    size_t size,
    size_t valid_size,
    void* complete_info) {
  // The record layout in complete_info is valid only for this call; on_complete
  // parses it and attaches it to the queued buffer.
  CuptiMonitorBuffers::get().on_complete(
      complete_info, buffer, size, valid_size);
}

} // namespace torch::profiler::impl
