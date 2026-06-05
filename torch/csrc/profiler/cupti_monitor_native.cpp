#include <torch/csrc/profiler/cupti_monitor_native.h>

#include <cstdlib>

namespace torch::profiler::impl {

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
    // requirement). Tracked in all_ so reset() can free it.
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
    completed_.push_back({buffer, valid_size, ctx, stream});
  }
  cv_.notify_one();
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
    std::free(p);
  }
  all_.clear();
  allocated_ = 0;
  shutdown_ = false;
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

} // namespace torch::profiler::impl
