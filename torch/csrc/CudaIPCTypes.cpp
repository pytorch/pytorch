#include <torch/csrc/CudaIPCTypes.h>
#include <iostream>
#include <map>
#include <mutex>

namespace torch {

struct Fia {
  std::mutex mutex;
  std::map<std::string, std::shared_ptr<CudaIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<CudaIPCRefCountersFile> next_available_ref_counters_file_;
  CudaIPCSentDataLimbo CudaIPCSentDataLimbo_;
  ~Fia() {
    std::cout << "Fia destroy\n";
    safe_clean_current_file();
  };
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << "clean_all_files\n";
    if (next_available_ref_counters_file_ != nullptr) {
      if (next_available_ref_counters_file_->used_slots_ == 0) {
        ref_counters_files_.erase(next_available_ref_counters_file_->handle_);
        next_available_ref_counters_file_ = nullptr;
      } else {
        std::cout << "cat delete " << next_available_ref_counters_file_->handle_
                  << " with " << next_available_ref_counters_file_->used_slots_
                  << " slots \n";
      }
    }
  }
};

Fia fia;

void CudaIPCCollect() {
  std::cout << "CudaIPCCollect " << fia.CudaIPCSentDataLimbo_.size() << "\n";
  fia.CudaIPCSentDataLimbo_.collect();
  if (fia.CudaIPCSentDataLimbo_.size() == 0) {
    fia.safe_clean_current_file();
  }
  std::cout << "CudaIPCCollect done\n";
}

CudaIPCReceivedData::CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
    : shared_ptr_(std::move(shared_ptr)) {}

int64_t CudaIPCSentData::get() {
  return *counter_ptr_;
}

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  std::cout << "Destroy LIMBO\n";
  collect();

  if (shared_blocks_.size() > 0) {
    std::cerr
        << "Producer process has been terminated before all shared CUDA tensors released.\n";
  }

  fia.safe_clean_current_file();
}

void CudaIPCSentDataLimbo::collect() {
  // std::cout << "Collecting LIMBO\n";
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

CudaIPCSentData::~CudaIPCSentData() {
  // std::cout << "SentData datele ptr" << original_ptr_.get() << "\n";
  ReturnRefCounter(handle_, offset_);
  cudaEventDestroy(event_); // TODO: Trap errors!
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
    fia.CudaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  fia.CudaIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(std::string handle, uint64_t offset /* unused */) {
  std::lock_guard<std::mutex> lock(fia.mutex);
  fia.ref_counters_files_[handle]->used_slots_--;
  if (fia.ref_counters_files_[handle]->used_slots_ == 0 &&
      fia.ref_counters_files_[handle]->next_offset_ ==
          fia.ref_counters_files_[handle]->size_) {
    fia.ref_counters_files_.erase(handle);
  }
}

bool CudaIPCHaveRefCounter() {
  std::lock_guard<std::mutex> lock(fia.mutex);
  return fia.next_available_ref_counters_file_ != nullptr;
}

void CudaIPCCreateRefCounter(
    std::string handle,
    uint64_t size,
    at::DataPtr data_ptr) {
  auto rc = std::make_shared<CudaIPCRefCountersFile>(
      handle, size, std::move(data_ptr));
  std::lock_guard<std::mutex> lock(fia.mutex);
  fia.ref_counters_files_[handle] = rc;
  fia.next_available_ref_counters_file_ = rc;
}

CudaIPCSentData* GetNewRefCountedSentData() {
  if (!CudaIPCHaveRefCounter()) {
    AT_ERROR("CudaIPCSentData() requires initialised RefCounter");
  }

  auto sent_data = new CudaIPCSentData(
      fia.next_available_ref_counters_file_->handle_,
      fia.next_available_ref_counters_file_->next_offset_,
      fia.next_available_ref_counters_file_->counter_ptr());

  *fia.next_available_ref_counters_file_->counter_ptr() = 1;
  fia.next_available_ref_counters_file_->next_offset_++;
  fia.next_available_ref_counters_file_->used_slots_++;
  if (fia.next_available_ref_counters_file_->next_offset_ ==
      fia.next_available_ref_counters_file_->size_) {
    fia.next_available_ref_counters_file_ = nullptr;
  }

  return sent_data;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  if (!CudaIPCHaveRefCounter()) {
    AT_ERROR("CudaIPCSentData() requires initialised RefCounter");
  }

  auto sent_data = new CudaIPCSentData(
      fia.next_available_ref_counters_file_->handle_,
      fia.next_available_ref_counters_file_->next_offset_,
      fia.next_available_ref_counters_file_->counter_ptr());

  *fia.next_available_ref_counters_file_->counter_ptr() = 1;
  fia.next_available_ref_counters_file_->next_offset_++;
  fia.next_available_ref_counters_file_->used_slots_++;
  if (fia.next_available_ref_counters_file_->next_offset_ ==
      fia.next_available_ref_counters_file_->size_) {
    fia.next_available_ref_counters_file_ = nullptr;
  }

  return at::DataPtr(data, sent_data, CudaIPCSentDataDelete, device);
}

CudaIPCRefCountersFile::~CudaIPCRefCountersFile() {
  std::cout << "Deleting ref counter " << handle_ << "\n";
}

} // namespace torch
