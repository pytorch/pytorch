#include <ATen/MapAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/CudaIPCTypes.h>
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
        << "Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]";
    warned = true;
  }
}

struct CudaIPCGlobalEntities {
  // This class is used as a singleton (see cuda_ipc_global_entities)
  // This variable is used to track its lifetime to avoid accessing it
  // after it was destroyed which would lead to segmentation faults
  // Note that a trvial type is used which doesn't suffer from construction
  // and destruction order issues
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<CudaIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<CudaIPCRefCountersFile> next_available_ref_counters_file_;
  CudaIPCSentDataLimbo CudaIPCSentDataLimbo_;
  CudaIPCGlobalEntities() {
    alive = true;
  }
  ~CudaIPCGlobalEntities() {
    CudaIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
    alive = false;
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

bool CudaIPCGlobalEntities::alive = false;
CudaIPCGlobalEntities cuda_ipc_global_entities;

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool CudaIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<CudaIPCSentData>> reset_blocks;
  { // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<CudaIPCSentData>> kept_blocks;
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

void CudaIPCSentDataLimbo::add(std::unique_ptr<CudaIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
      !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
        << "We assume it will never going to be the case, but if it is, please file but to https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t CudaIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void CudaIPCSentDataDelete(void* ptr) {
  std::unique_ptr<CudaIPCSentData> sent_data(
      static_cast<CudaIPCSentData*>(ptr));
  if (!CudaIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    cuda_ipc_global_entities.CudaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!CudaIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(
      cuda_ipc_global_entities.ref_counters_mutex_);
  auto& map = cuda_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

} // namespace

CudaIPCSentData::CudaIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      original_ptr_(),
      device_(device) {
#if !defined(USE_ROCM)
  // CUDA have the unofficial limit on the number of recorded blocking
  // interprocess events, to prevent using of all events, we are switching to
  // StreamSync before limit reached.
  //
  //  ```python
  //  import torch
  //  a = [ torch.cuda.Event(
  //      enable_timing=False, blocking=True, interprocess=True) for i in
  //      range(30000) ]
  //  [i.record() for i in a]
  //  ```
  //
  if (cuda_ipc_global_entities.sync_events_used_.load() <
      CUDA_IPC_MAXIMUM_EVENTS_TO_USE) {
    // TODO: More efficient would be to create event inside of main thread (at
    // the moment of the queue.put). The reason this is more efficient is
    // because the main thread may have queued extra work on the stream, which
    // this event will consequently wait for (uselessly).
    cuda_ipc_global_entities.sync_events_used_++;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(
        &event_,
        cudaEventDisableTiming | cudaEventInterprocess |
            cudaEventBlockingSync));
    C10_CUDA_CHECK(cudaEventRecord(
        event_, c10::cuda::getCurrentCUDAStream(device.index())));
    event_sync_required_ = true;
  } else {
    auto stream = c10::cuda::getCurrentCUDAStream(device.index());
    at::cuda::stream_synchronize(stream);
    event_ = nullptr;
    event_sync_required_ = false;
  }
#else
  // cuIpcGetEventHandle with HIP is not supported, so we have to sync
  // stream instead of passing event
  auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  at::cuda::stream_synchronize(stream);
  event_sync_required_ = false;
#endif
}

CudaIPCSentData::~CudaIPCSentData() {
  ReturnRefCounter(handle_, offset_);
#if !defined(USE_ROCM)
  try {
    if (event_sync_required_) {
      at::cuda::CUDAGuard device_guard(device_.index());
      C10_CUDA_CHECK(cudaEventDestroy(event_));
      if (!CudaIPCGlobalEntities::alive) {
        return;
      }
      cuda_ipc_global_entities.sync_events_used_--;
    }
  } catch (...) { /* No throw */
  }
#endif
}

uint64_t CudaIPCSentData::counter_value() {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(
        cuda_ipc_global_entities.ref_counters_mutex_);
    if (!cuda_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * CUDA_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<CudaIPCRefCountersFile>(
          ref_counter_handle, CUDA_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      cuda_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      cuda_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  cuda_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto sent_data = new CudaIPCSentData(
      cuda_ipc_global_entities.next_available_ref_counters_file_->handle(),
      cuda_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      cuda_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);

  cuda_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!cuda_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    cuda_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, CudaIPCSentDataDelete, device);
}

bool CudaIPCCollect() {
  if (!CudaIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
  if (cuda_ipc_global_entities.CudaIPCSentDataLimbo_.size() == 0) {
    cuda_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch

namespace c10 {
namespace {
REGISTER_FREE_MEMORY_CALLBACK("cuda_ipc_collect", CudaIPCCollectCallback);
}
} // namespace c10
