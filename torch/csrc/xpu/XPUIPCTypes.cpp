#include <ATen/MapAllocator.h>
#include <c10/util/Logging.h>
#include <c10/xpu/XPUStream.h>
#include <torch/csrc/xpu/XPUIPCTypes.h>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace torch {

namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared XPU tensors released.";
    warned = true;
  }
}

struct XpuIPCRefCountersFile final {
  XpuIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }
  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }
  bool have_offsets() const {
    return next_offset_ < size_;
  }
  uint64_t offsets_in_use() const {
    return used_slots_;
  }
  uint64_t get_offset() const {
    return next_offset_;
  }
  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }
  void return_offset() {
    used_slots_--;
  }
  const std::string& handle() const {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

struct XpuIPCSentDataLimbo final {
  ~XpuIPCSentDataLimbo() {
    collect();
    if (size() > 0) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
  }

  bool collect() {
    bool freed_memory = false;
    std::vector<std::unique_ptr<XpuIPCSentData>> reset_blocks;
    {
      std::lock_guard<std::mutex> lock(limbo_mutex_);
      std::vector<std::unique_ptr<XpuIPCSentData>> kept_blocks;
      for (auto& sd : shared_blocks_) {
        if (sd->counter_value() > 0) {
          kept_blocks.push_back(std::move(sd));
        } else {
          freed_memory = true;
          reset_blocks.push_back(std::move(sd));
        }
      }
      shared_blocks_ = std::move(kept_blocks);
    }
    reset_blocks.clear();
    return freed_memory;
  }

  void add(std::unique_ptr<XpuIPCSentData> shared_block) {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    shared_blocks_.push_back(std::move(shared_block));
  }

  uint64_t size() {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    return shared_blocks_.size();
  }

 private:
  std::vector<std::unique_ptr<XpuIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct XpuIPCGlobalEntities {
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::map<std::string, std::shared_ptr<XpuIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<XpuIPCRefCountersFile> next_available_ref_counters_file_;
  XpuIPCSentDataLimbo limbo_;

  XpuIPCGlobalEntities() {
    alive = true;
  }
  XpuIPCGlobalEntities(const XpuIPCGlobalEntities&) = delete;
  XpuIPCGlobalEntities(XpuIPCGlobalEntities&&) = delete;
  XpuIPCGlobalEntities& operator=(const XpuIPCGlobalEntities&) = delete;
  XpuIPCGlobalEntities& operator=(XpuIPCGlobalEntities&&) = delete;
  ~XpuIPCGlobalEntities() {
    alive = false;
    limbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
  }

  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

bool XpuIPCGlobalEntities::alive = false;
XpuIPCGlobalEntities xpu_ipc_global_entities;

void XpuIPCSentDataDelete(void* ptr) {
  std::unique_ptr<XpuIPCSentData> sent_data(static_cast<XpuIPCSentData*>(ptr));
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    xpu_ipc_global_entities.limbo_.add(std::move(sent_data));
  }
  xpu_ipc_global_entities.limbo_.collect();
}

void ReturnRefCounter(const std::string& handle) {
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);
  auto& map = xpu_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset();
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(it);
    }
  }
}

} // namespace

XpuIPCSentData::XpuIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr) {
  c10::xpu::getCurrentXPUStream(device.index()).synchronize();
}

XpuIPCSentData::~XpuIPCSentData() {
  if (!XpuIPCGlobalEntities::alive) {
    original_ptr_.release_context();
  }
  ReturnRefCounter(handle_);
}

at::DataPtr GetNewRefCountedSentXpuData(void* data, at::Device device) {
  std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);
  auto& next_file = xpu_ipc_global_entities.next_available_ref_counters_file_;
  if (!next_file) {
    std::string ref_counter_handle = at::NewProcessWideShmHandle();
    int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
    at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(),
        flags,
        sizeof(int64_t) * XPU_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    next_file = std::make_shared<XpuIPCRefCountersFile>(
        ref_counter_handle, XPU_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
    xpu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = next_file;
  }
  next_file->set_counter(1);
  auto sent_data = new XpuIPCSentData(
      next_file->handle(),
      next_file->get_offset(),
      next_file->counter_ptr(),
      device);
  next_file->rotate_offset();
  if (!next_file->have_offsets()) {
    next_file.reset();
  }
  return at::DataPtr(data, sent_data, XpuIPCSentDataDelete, device);
}

bool XpuIPCCollect() {
  if (!XpuIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = xpu_ipc_global_entities.limbo_.collect();
  if (xpu_ipc_global_entities.limbo_.size() == 0) {
    xpu_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch
