#include <torch/csrc/profiler/cupti_monitor_native.h>

#include <cstdlib>

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

} // namespace

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
    uint64_t ctx,
    uint32_t stream,
    uint8_t* buffer,
    size_t /*size*/,
    size_t valid_size) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    completed_.push_back({buffer, valid_size, ctx, stream, current_epoch_});
  }
  cv_.notify_one();
}

void CuptiMonitorBuffers::on_complete_v2(
    void* complete_info,
    uint8_t* buffer,
    size_t /*size*/,
    size_t valid_size) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // Capture + tag under one lock so the buffer's epoch and its layout stay
    // consistent even if next_layout_epoch() races a completion. v2 delivers
    // neither CUcontext nor streamId (they are selectable record fields), so
    // ctx/stream are 0.
    capture_layouts_locked(complete_info);
    completed_.push_back({buffer, valid_size, 0, 0, current_epoch_});
  }
  cv_.notify_one();
}

uint64_t CuptiMonitorBuffers::next_layout_epoch() {
  std::lock_guard<std::mutex> guard(mutex_);
  return ++current_epoch_;
}

std::optional<CompletedCuptiBuffer> CuptiMonitorBuffers::get_completed() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return !completed_.empty() || shutdown_; });
  if (completed_.empty()) {
    return std::nullopt; // shutdown
  }
  CompletedCuptiBuffer buf = completed_.front();
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
  current_epoch_ = 0;
  layouts_.clear();
}

void CuptiMonitorBuffers::capture_layouts_locked(void* complete_info) {
  if (complete_info == nullptr || layouts_.contains(current_epoch_)) {
    return;
  }
  std::vector<CuptiRecordLayout> snapshot;
  const auto* info = static_cast<const AbiBufferCompleteInfo*>(complete_info);
  if (info->ppRecordLayouts != nullptr) {
    // ppRecordLayouts is indexed by activity kind; entries are null for kinds
    // without a user-defined layout.
    for (size_t kind = 0; kind < info->numRecordLayouts; ++kind) {
      const AbiRecordLayout* layout = info->ppRecordLayouts[kind];
      if (layout == nullptr) {
        continue;
      }
      CuptiRecordLayout out;
      out.kind = static_cast<uint32_t>(kind);
      out.record_size = layout->recordSize;
      out.fields.reserve(layout->numFields);
      for (size_t f = 0; f < layout->numFields; ++f) {
        const AbiFieldLayoutEntry& e = layout->pEntries[f];
        out.fields.push_back({e.fieldId, e.offset, e.size});
      }
      snapshot.push_back(std::move(out));
    }
  }
  // Insert even when empty so this epoch is not re-parsed on every buffer.
  layouts_.emplace(current_epoch_, std::move(snapshot));
}

std::vector<CuptiRecordLayout> CuptiMonitorBuffers::record_layouts(
    uint64_t epoch) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = layouts_.find(epoch);
  if (it == layouts_.end()) {
    return {};
  }
  return it->second;
}

void cuptiMonitorBufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* max_num_records) {
  CuptiMonitorBuffers::get().on_request(buffer, size, max_num_records);
}

void cuptiMonitorBufferCompleted(
    void* context,
    uint32_t stream_id,
    uint8_t* buffer,
    size_t size,
    size_t valid_size) {
  CuptiMonitorBuffers::get().on_complete(
      reinterpret_cast<uint64_t>(context), stream_id, buffer, size, valid_size);
}

void cuptiMonitorBufferRequestedV2(
    uint8_t** buffer,
    size_t* size,
    size_t* max_num_records,
    void* /*request_info*/) {
  CuptiMonitorBuffers::get().on_request(buffer, size, max_num_records);
}

void cuptiMonitorBufferCompletedV2(
    uint8_t* buffer,
    size_t size,
    size_t valid_size,
    void* complete_info) {
  // The record layout in complete_info is valid only for this call;
  // on_complete_v2 snapshots it into the current epoch before queuing.
  CuptiMonitorBuffers::get().on_complete_v2(
      complete_info, buffer, size, valid_size);
}

} // namespace torch::profiler::impl
