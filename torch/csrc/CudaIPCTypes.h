#pragma once
#include <c10/core/Allocator.h>

struct CudaIPCReceivedData {
  CudaIPCReceivedData(std::shared_ptr<void> shared_ptr);
  std::shared_ptr<void> shared_ptr_;
};

struct CudaIPCSentData {
  CudaIPCSentData(at::DataPtr shared_storage_ptr);
  ~CudaIPCSentData();
  int64_t get();
  at::DataPtr original_ptr_; // Original mem allocation
  at::DataPtr shared_storage_ptr_; // Reference counter shared memory block
};

// All to be deleted data blocks with non zero reference counter goes there
struct CudaIPCSentDataLimbo {
  void collect();
  ~CudaIPCSentDataLimbo();
  void add(CudaIPCSentData* shared_block);
private:
  std::vector<CudaIPCSentData*> shared_blocks_;
};

void CudaIPCSentDataDelete(void* ptr);
