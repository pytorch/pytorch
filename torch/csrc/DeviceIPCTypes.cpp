#include <torch/csrc/DeviceIPCTypes.h>
#include <ATen/MapAllocator.h>
#include <c10/util/Logging.h>
#include <map>

namespace torch {

namespace {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared device "
           "tensors released. See Note [Sharing device tensors]";
    warned = true;
  }
}

struct DeviceIPCGlobalEntities {
  // Trivial bool avoids static destruction order issues.
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::map<std::string, std::shared_ptr<DeviceIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<DeviceIPCRefCountersFile> next_available_ref_counters_file_;
  DeviceIPCSentDataLimbo DeviceIPCSentDataLimbo_;

  DeviceIPCGlobalEntities() {
    alive = true;
  }
  DeviceIPCGlobalEntities(const DeviceIPCGlobalEntities&) = delete;
  DeviceIPCGlobalEntities(DeviceIPCGlobalEntities&&) = delete;
  DeviceIPCGlobalEntities& operator=(const DeviceIPCGlobalEntities&) = delete;
  DeviceIPCGlobalEntities& operator=(DeviceIPCGlobalEntities&&) = delete;
  ~DeviceIPCGlobalEntities() {
    alive = false;
    DeviceIPCSentDataLimbo_.collect();
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

bool DeviceIPCGlobalEntities::alive = false;
DeviceIPCGlobalEntities device_ipc_global_entities;

DeviceIPCSentDataLimbo::~DeviceIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool DeviceIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<DeviceIPCSentData>> reset_blocks;
  {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<DeviceIPCSentData>> kept_blocks;
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

void DeviceIPCSentDataLimbo::add(
    std::unique_ptr<DeviceIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (static_cast<int64_t>(shared_blocks_.size()) >
          DEVICE_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
      !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << DEVICE_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might "
           "be significantly slowed down. "
        << "We assume it will never going to be the case, but if it is, "
           "please file a bug at https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t DeviceIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void DeviceIPCSentDataDelete(void* ptr) {
  std::unique_ptr<DeviceIPCSentData> sent_data(
      static_cast<DeviceIPCSentData*>(ptr));
  if (!DeviceIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    device_ipc_global_entities.DeviceIPCSentDataLimbo_.add(
        std::move(sent_data));
  }
  device_ipc_global_entities.DeviceIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!DeviceIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(
      device_ipc_global_entities.ref_counters_mutex_);
  auto& map = device_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

} // namespace

DeviceIPCSentData::DeviceIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      device_(device) {}

DeviceIPCSentData::~DeviceIPCSentData() {
  if (!DeviceIPCGlobalEntities::alive) {
    original_ptr_.release_context();
  }
  ReturnRefCounter(handle_, offset_);
}

uint64_t DeviceIPCSentData::counter_value() {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentDataForDevice(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(
        device_ipc_global_entities.ref_counters_mutex_);
    if (!device_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(uint64_t) * DEVICE_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<DeviceIPCRefCountersFile>(
          ref_counter_handle,
          DEVICE_IPC_REF_COUNTER_FILE_SIZE,
          std::move(sptr));
      device_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      device_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  device_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto sent_data = new DeviceIPCSentData(
      device_ipc_global_entities.next_available_ref_counters_file_->handle(),
      device_ipc_global_entities.next_available_ref_counters_file_
          ->get_offset(),
      device_ipc_global_entities.next_available_ref_counters_file_
          ->counter_ptr(),
      device);

  device_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!device_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    device_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, DeviceIPCSentDataDelete, device);
}

bool DeviceIPCCollect() {
  if (!DeviceIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory =
      device_ipc_global_entities.DeviceIPCSentDataLimbo_.collect();
  if (device_ipc_global_entities.DeviceIPCSentDataLimbo_.size() == 0) {
    device_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch