#ifdef USE_CUDA
#include <torch/csrc/CudaIPCTypes.h>
#include <map>
#include <mutex>

namespace torch {
void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared CUDA tensors released.";
    warned = true;
  }
}

struct CudaIPCGlobalEntities {
  std::mutex ref_counters_mutex_;
  std::map<std::string, std::shared_ptr<CudaIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<CudaIPCRefCountersFile> next_available_ref_counters_file_;
  CudaIPCSentDataLimbo CudaIPCSentDataLimbo_;
  CudaIPCGlobalEntities() : ref_counters_files_() {}
  ~CudaIPCGlobalEntities() {
    CudaIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
  };
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

CudaIPCGlobalEntities cuda_ipc_global_entities;

void CudaIPCCollect() {
  cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
  if (cuda_ipc_global_entities.CudaIPCSentDataLimbo_.size() == 0) {
    cuda_ipc_global_entities.safe_clean_current_file();
  }
}

CudaIPCReceivedData::CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
    : shared_ptr_(std::move(shared_ptr)) {}

int64_t CudaIPCSentData::counter_value() {
  return *counter_ptr_;
}

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  collect();
  if (shared_blocks_.size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

void CudaIPCSentDataLimbo::collect() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  std::vector<std::unique_ptr<CudaIPCSentData>> kept_blocks;
  for (auto& sd : shared_blocks_) {
    if (sd->counter_value() > 0) {
      kept_blocks.push_back(std::move(sd));
    } else {
      sd.reset();
    }
  }
  shared_blocks_ = std::move(kept_blocks);
}

CudaIPCSentData::~CudaIPCSentData() {
  ReturnRefCounter(handle_, offset_);
#ifndef __HIP_PLATFORM_HCC__
  try {
    at::cuda::CUDAGuard device_guard(device_.index());
    cudaEventDestroy(event_);
  } catch (...) { /* No throw */
  }
#endif
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

void CudaIPCSentDataDelete(void* ptr) {
  std::unique_ptr<CudaIPCSentData> sent_data(
      static_cast<CudaIPCSentData*>(ptr));
  if (sent_data->counter_value() > 0) {
    cuda_ipc_global_entities.CudaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string handle, uint64_t offset /* unused */) {
  std::lock_guard<std::mutex> lock(
      cuda_ipc_global_entities.ref_counters_mutex_);
  cuda_ipc_global_entities.ref_counters_files_[handle]->return_offset(offset);
  if (cuda_ipc_global_entities.ref_counters_files_[handle]->offsets_in_use() ==
          0 &&
      !cuda_ipc_global_entities.ref_counters_files_[handle]->have_offsets()) {
    cuda_ipc_global_entities.ref_counters_files_.erase(handle);
  }
}

bool CudaIPCHaveRefCounter() {
  std::lock_guard<std::mutex> lock(
      cuda_ipc_global_entities.ref_counters_mutex_);
  return (bool)cuda_ipc_global_entities.next_available_ref_counters_file_;
}

void CudaIPCCreateRefCounter(
    const std::string handle,
    uint64_t size,
    at::DataPtr data_ptr) {
  auto rc = std::make_shared<CudaIPCRefCountersFile>(
      handle, size, std::move(data_ptr));
  std::lock_guard<std::mutex> lock(
      cuda_ipc_global_entities.ref_counters_mutex_);
  cuda_ipc_global_entities.ref_counters_files_[handle] = rc;
  cuda_ipc_global_entities.next_available_ref_counters_file_ = rc;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  if (!CudaIPCHaveRefCounter()) {
    AT_ERROR("GetNewRefCountedSentData() requires initialised IPCRefCounter");
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

} // namespace torch

namespace c10 {
REGISTER_FREE_MEMORY_CALLBACK("cuda_ipc_collect", CudaIPCCollectCallback);
}

#endif
