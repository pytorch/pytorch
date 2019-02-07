#include <torch/csrc/CudaIPCTypes.h>
#include <iostream>
#include <map>

namespace torch {

CudaIPCReceivedData::CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
    : shared_ptr_(std::move(shared_ptr)) {}

int64_t CudaIPCSentData::get() {
  return *counter_ptr;
}

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  collect();
  if (shared_blocks_.size() > 0) {
    std::cerr
        << "Producer process has been terminated before all shared CUDA tensors released.\n";
  }
}

void CudaIPCSentDataLimbo::collect() {
  std::vector<std::unique_ptr<CudaIPCSentData>> kept_blocks;
  for (auto& sd : shared_blocks_) {
    if (sd->get() > 0) {
      kept_blocks.push_back(std::move(sd));
    } else {
      sd.reset();
    }
  }
  shared_blocks_ = std::move(kept_blocks);
}

CudaIPCSentDataLimbo CudaIPCSentDataLimbo;

CudaIPCSentData::~CudaIPCSentData() {
  ReturnRefCounter(handle, offset);
}

void CudaIPCSentDataLimbo::add(std::unique_ptr<CudaIPCSentData> shared_block) {
  static bool warned = false;
  if (shared_blocks_.size() > CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
      !warned) {
    std::cerr
        << "Producer process tried to deallocate over "
        << CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down.\n";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

void CudaIPCSentDataDelete(void* ptr) {
  std::unique_ptr<CudaIPCSentData> sent_data(
      static_cast<CudaIPCSentData*>(ptr));
  if (sent_data->get() > 0) {
    // TODO: As alternative we can lock current process until resource get
    // released and cerr to console.
    CudaIPCSentDataLimbo.add(std::move(sent_data));
  }
  CudaIPCSentDataLimbo.collect();
}

std::map<std::string, std::shared_ptr<CudaIPCRefCountersFile>> ref_counters_files_;
std::shared_ptr<CudaIPCRefCountersFile> next_available_ref_counters_file_;

void ReturnRefCounter(std::string handle, uint64_t offset) {
  ref_counters_files_[handle]->used_slots--;
  if (ref_counters_files_[handle]->used_slots == 0) {
    ref_counters_files_.erase(handle);
  }
}

bool CudaIPCHaveRefCounter() {
  return next_available_ref_counters_file_ != nullptr;
}

void CudaIPCCreateRefCounter(
    std::string handle,
    uint64_t size,
    at::DataPtr data_ptr) {
  auto rc = std::shared_ptr<CudaIPCRefCountersFile>(
      new CudaIPCRefCountersFile(handle, size, std::move(data_ptr)));
  ref_counters_files_[handle] = rc;
  next_available_ref_counters_file_ = rc;
}

CudaIPCSentData* GetNewRefCountedSentData() {

  if (!CudaIPCHaveRefCounter()) {
    AT_ERROR("CudaIPCSentData() requires initialised RefCounter");
  }

  auto sent_data = new CudaIPCSentData(
      next_available_ref_counters_file_->handle,
      next_available_ref_counters_file_->next_offset,
      next_available_ref_counters_file_->counter_ptr());

  *next_available_ref_counters_file_->counter_ptr() = 1;
  next_available_ref_counters_file_->next_offset++;
  next_available_ref_counters_file_->used_slots++;
  if (next_available_ref_counters_file_->next_offset == next_available_ref_counters_file_->size) {
    next_available_ref_counters_file_ = nullptr;
  }
  return sent_data;
}

CudaIPCRefCountersFile::~CudaIPCRefCountersFile() {}

} // namespace torch
