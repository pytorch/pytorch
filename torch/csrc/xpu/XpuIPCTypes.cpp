#include <ATen/MapAllocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Logging.h>
#include <torch/csrc/xpu/XpuIPCTypes.h>
#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace torch {

namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared XPU tensors released. See Note [Sharing CUDA tensors]";
    warned = true;
  }
}

struct XpuIPCGlobalEntities {
  // See CudaIPCGlobalEntities::alive
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<XpuIPCRefCountersFile>>
      ref_counters_files_{};
  std::shared_ptr<XpuIPCRefCountersFile> next_available_ref_counters_file_;
  XpuIPCSentDataLimbo XpuIPCSentDataLimbo_;
  XpuIPCGlobalEntities() {
    alive = true;
  }
  XpuIPCGlobalEntities(const XpuIPCGlobalEntities&) = delete;
  XpuIPCGlobalEntities(XpuIPCGlobalEntities&&) = delete;
  XpuIPCGlobalEntities& operator=(const XpuIPCGlobalEntities&) = delete;
  XpuIPCGlobalEntities& operator=(XpuIPCGlobalEntities&&) = delete;
  ~XpuIPCGlobalEntities() {
    alive = false;
    XpuIPCSentDataLimbo_.collect();
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

XpuIPCSentDataLimbo::~XpuIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool XpuIPCSentDataLimbo::collect() {
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
  // Reset blocks outside of the critical section, otherwise it deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void XpuIPCSentDataLimbo::add(std::unique_ptr<XpuIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO && !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down.";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t XpuIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void XpuIPCSentDataDelete(void* ptr) {
  std::unique_ptr<XpuIPCSentData> sent_data(static_cast<XpuIPCSentData*>(ptr));
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    xpu_ipc_global_entities.XpuIPCSentDataLimbo_.add(std::move(sent_data));
  }
  xpu_ipc_global_entities.XpuIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset) {
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);
  auto& map = xpu_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

bool deviceSupportsIpcEvent([[maybe_unused]] c10::DeviceIndex device_index) {
#if SYCL_COMPILER_VERSION >= 20260200 && !defined(_WIN32)
  return c10::xpu::get_raw_device(device_index)
      .has(sycl::aspect::ext_oneapi_ipc_event);
#else
  return false;
#endif
}

} // namespace

XpuIPCSentData::XpuIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      event_sync_required_(false),
      device_(device) {
  auto stream = c10::xpu::getCurrentXPUStream(device.index());
  // Fall back to stream sync when IPC events are unavailable or too many are
  // in flight, mirroring CudaIPCSentData.
#if SYCL_COMPILER_VERSION >= 20260200
  if (deviceSupportsIpcEvent(device.index()) &&
      xpu_ipc_global_entities.sync_events_used_.load() <
          XPU_IPC_MAXIMUM_EVENTS_TO_USE) {
    xpu_ipc_global_entities.sync_events_used_++;
    event_.emplace(/*enable_timing=*/false, /*enable_ipc=*/true);
    event_->record(stream);
    auto handle_data = event_->ipc_handle();
    event_handle_.assign(
        reinterpret_cast<const char*>(handle_data.data()), handle_data.size());
    event_sync_required_ = true;
    return;
  }
#endif
  stream.synchronize();
}

XpuIPCSentData::~XpuIPCSentData() {
  if (!XpuIPCGlobalEntities::alive) {
    original_ptr_.release_context();
  }
  ReturnRefCounter(handle_, offset_);
  try {
    if (event_sync_required_) {
      c10::DeviceGuard device_guard(device_);
      event_.reset();
      if (!XpuIPCGlobalEntities::alive) {
        return;
      }
      xpu_ipc_global_entities.sync_events_used_--;
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (...) { /* No throw */
  }
}

uint64_t XpuIPCSentData::counter_value() {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentXpuData(void* data, at::Device device) {
  std::string ref_counter_handle;
  uint64_t ref_counter_offset = 0;
  uint64_t* ref_counter_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(
        xpu_ipc_global_entities.ref_counters_mutex_);
    if (!xpu_ipc_global_entities.next_available_ref_counters_file_) {
      ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * XPU_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<XpuIPCRefCountersFile>(
          ref_counter_handle, XPU_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      xpu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      xpu_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
    auto& ref_counter_file =
        xpu_ipc_global_entities.next_available_ref_counters_file_;
    ref_counter_handle = ref_counter_file->handle();
    ref_counter_offset = ref_counter_file->get_offset();
    ref_counter_ptr = ref_counter_file->counter_ptr();
    // Reserve the slot before XpuIPCSentData can trigger IPC collection.
    ref_counter_file->set_counter(1);
    ref_counter_file->rotate_offset();
    if (!ref_counter_file->have_offsets()) {
      ref_counter_file.reset();
    }
  }
  try {
    auto sent_data = std::make_unique<XpuIPCSentData>(
        ref_counter_handle, ref_counter_offset, ref_counter_ptr, device);
    return at::DataPtr(data, sent_data.release(), XpuIPCSentDataDelete, device);
  } catch (...) {
    ReturnRefCounter(ref_counter_handle, ref_counter_offset);
    throw;
  }
}

bool XpuIPCCollect() {
  if (!XpuIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = xpu_ipc_global_entities.XpuIPCSentDataLimbo_.collect();
  if (xpu_ipc_global_entities.XpuIPCSentDataLimbo_.size() == 0) {
    xpu_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch
