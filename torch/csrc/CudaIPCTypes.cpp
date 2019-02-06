#include <torch/csrc/CudaIPCTypes.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <map>

CudaIPCReceivedData::CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
    : shared_ptr_(std::move(shared_ptr)) {}

int64_t CudaIPCSentData::get() {
  return *counter_ptr;
}

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  collect();
  if (shared_blocks_.size() > 0) {
    std::cout <<
        "Producer process has been terminated before all shared CUDA tensors released.\n";
  }
}

void CudaIPCSentDataLimbo::collect() {
  std::vector<CudaIPCSentData*> kept_blocks;
  for (auto const& sd : shared_blocks_) {
    if (sd->get() > 0) {
      kept_blocks.push_back(sd);
    } else {
      delete sd;
    }
  }
  shared_blocks_ = kept_blocks;
}

CudaIPCSentDataLimbo CudaIPCSentDataLimbo;

CudaIPCSentData::~CudaIPCSentData() {
  ReturnRefCounter(handle, offset);
}

void CudaIPCSentDataLimbo::add(CudaIPCSentData* shared_block) {
  shared_blocks_.push_back(shared_block);
}

// CudaIPCSentData::CudaIPCSentData(at::DataPtr shared_storage_ptr)
//     : shared_storage_ptr_(std::move(shared_storage_ptr)) {}

void CudaIPCSentDataDelete(void* ptr) {
  auto rc = (CudaIPCSentData*)ptr;
  if (rc->get() > 0) {
    CudaIPCSentDataLimbo.add(rc);
  } else {
    delete rc;
  }
  CudaIPCSentDataLimbo.collect();
}

std::map<std::string, std::shared_ptr<RefCounterFile>> rc_files;

std::shared_ptr<RefCounterFile> next_available_file;

void ReturnRefCounter(std::string handler, uint64_t offset) {
  rc_files[handler]->used_slots--;
  if (rc_files[handler]->used_slots == 0) {
    rc_files.erase(handler);
  }
}

bool HaveNewRefCounter() {
  return next_available_file != nullptr;
}

void CreateRefCounter(std::string handle, uint64_t size, at::DataPtr data_ptr) {
  if (HaveNewRefCounter()) {
    // ERROR
  }
  auto rc = std::shared_ptr<RefCounterFile>(
      new RefCounterFile(handle, size, std::move(data_ptr)));
  rc_files[handle] = rc;
  next_available_file = rc;
}

IPCShareData GetNewRefCounter() {
  if (!HaveNewRefCounter()) {
    // ERROR
  }
  auto res = IPCShareData(
      next_available_file->handle,
      next_available_file->next_offset,
      next_available_file->counter_ptr());
  *next_available_file->counter_ptr() = 1;
  next_available_file->next_offset++;
  next_available_file->used_slots++;
  if (next_available_file->next_offset == next_available_file->size) {
    next_available_file = nullptr;
  }
  return res;
}

RefCounterFile::~RefCounterFile() {}
