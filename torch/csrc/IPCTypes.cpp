#include <ATen/DeviceAccelerator.h>
#include <ATen/MapAllocator.h>
#include <torch/csrc/IPCTypes.h>
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
        << "Producer process has been terminated before all shared device tensors released. See Note [Sharing CUDA tensors]";
    warned = true;
  }
}

struct IPCGlobalEntities {
  // This class is used as a singleton (see ipc_global_entities)
  // This variable is used to track its lifetime to avoid accessing it
  // after it was destroyed which would lead to segmentation faults
  // Note that a trivial type is used which doesn't suffer from construction
  // and destruction order issues
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<IPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<IPCRefCountersFile> next_available_ref_counters_file_;
  IPCSentDataLimbo ipc_sent_data_limbo_;
  IPCGlobalEntities() {
    alive = true;
  }
  IPCGlobalEntities(const IPCGlobalEntities&) = delete;
  IPCGlobalEntities(IPCGlobalEntities&&) = delete;
  IPCGlobalEntities& operator=(const IPCGlobalEntities&) = delete;
  IPCGlobalEntities& operator=(IPCGlobalEntities&&) = delete;
  ~IPCGlobalEntities() {
    alive = false;
    ipc_sent_data_limbo_.collect();
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

bool IPCGlobalEntities::alive = false;
IPCGlobalEntities ipc_global_entities;

IPCSentDataLimbo::~IPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool IPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<IPCSentData>> reset_blocks;
  { // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<IPCSentData>> kept_blocks;
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
  // Need to reset blocks out of the critical section here, otherwise it
  // deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void IPCSentDataLimbo::add(std::unique_ptr<IPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO && !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
        << "We assume it will never going to be the case, but if it is, please file but to https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t IPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void IPCSentDataDelete(void* ptr) {
  std::unique_ptr<IPCSentData> sent_data(static_cast<IPCSentData*>(ptr));
  if (!IPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    ipc_global_entities.ipc_sent_data_limbo_.add(std::move(sent_data));
  }
  ipc_global_entities.ipc_sent_data_limbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!IPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(ipc_global_entities.ref_counters_mutex_);
  auto& map = ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

} // namespace

IPCSentData::IPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      device_(device) {
  auto stream = at::accelerator::getCurrentStream(device_.index());
  // Backends may limit the number of live interprocess events, so fall back to
  // stream sync once IPC_MAXIMUM_EVENTS_TO_USE is reached.
  if (ipc_global_entities.sync_events_used_.load() <
      IPC_MAXIMUM_EVENTS_TO_USE) {
    // TODO: More efficient would be to create event inside of main thread (at
    // the moment of the queue.put). The reason this is more efficient is
    // because the main thread may have queued extra work on the stream, which
    // this event will consequently wait for (uselessly).
    event_.emplace(
        device_.type(),
        c10::EventFlag::INTERPROCESS | c10::EventFlag::BLOCKING);
    event_->record(stream);
    ipc_global_entities.sync_events_used_++;
    event_sync_required_ = true;
  } else {
    stream.synchronize();
    event_ = std::nullopt;
    event_sync_required_ = false;
  }
}

IPCSentData::~IPCSentData() {
  if (!IPCGlobalEntities::alive) {
    original_ptr_.release_context();
  }
  ReturnRefCounter(handle_, offset_);
  try {
    if (event_sync_required_) {
      event_.reset();
      if (!IPCGlobalEntities::alive) {
        return;
      }
      ipc_global_entities.sync_events_used_--;
    }
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (...) { /* No throw */
  }
}

uint64_t IPCSentData::counter_value() {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  std::string ref_counter_handle;
  uint64_t ref_counter_offset;
  uint64_t* ref_counter_ptr;
  {
    std::lock_guard<std::mutex> lock(ipc_global_entities.ref_counters_mutex_);
    if (!ipc_global_entities.next_available_ref_counters_file_) {
      ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<IPCRefCountersFile>(
          ref_counter_handle, IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
    auto& ref_counter_file =
        ipc_global_entities.next_available_ref_counters_file_;
    ref_counter_handle = ref_counter_file->handle();
    ref_counter_offset = ref_counter_file->get_offset();
    ref_counter_ptr = ref_counter_file->counter_ptr();
    // Reserve the slot before IPCSentData can trigger IPC collection.
    ref_counter_file->set_counter(1);
    ref_counter_file->rotate_offset();
    if (!ref_counter_file->have_offsets()) {
      ref_counter_file.reset();
    }
  }
  try {
    auto sent_data = std::make_unique<IPCSentData>(
        ref_counter_handle, ref_counter_offset, ref_counter_ptr, device);
    return at::DataPtr(data, sent_data.release(), IPCSentDataDelete, device);
  } catch (...) {
    ReturnRefCounter(ref_counter_handle, ref_counter_offset);
    throw;
  }
}

bool IPCCollect() {
  if (!IPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = ipc_global_entities.ipc_sent_data_limbo_.collect();
  if (ipc_global_entities.ipc_sent_data_limbo_.size() == 0) {
    ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch
