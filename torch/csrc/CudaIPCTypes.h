#pragma once
#include <c10/core/Allocator.h>

#define CUDA_IPC_REF_COUNTER_FILE_SIZE 10000
#define CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO 1000

namespace c10 {
namespace cuda {

struct CudaIPCReceivedData {
  CudaIPCReceivedData(std::shared_ptr<void> shared_ptr);
  std::shared_ptr<void> shared_ptr_;
};

struct CudaIPCSentData {
  CudaIPCSentData(std::string handle, int64_t offset, int64_t* counter_ptr)
      : handle(handle), offset(offset), counter_ptr(counter_ptr){};
  ~CudaIPCSentData();
  int64_t get();
  std::string handle;
  int64_t offset;
  int64_t* counter_ptr; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
};

void CudaIPCSentDataDelete(void* ptr);

// All to be deleted data blocks with non zero reference counter goes there
struct CudaIPCSentDataLimbo {
  void collect();
  ~CudaIPCSentDataLimbo();
  void add(std::unique_ptr<CudaIPCSentData> shared_block);
private:
  std::vector<std::unique_ptr<CudaIPCSentData>> shared_blocks_;
};

CudaIPCSentData* GetNewRefCountedSentData();
void ReturnRefCounter(std::string handle, uint64_t offset);
void CudaIPCCreateRefCounter(std::string handle, uint64_t size, at::DataPtr data_ptr);

bool CudaIPCHaveRefCounter();

struct CudaIPCRefCountersFile {
  uint64_t next_offset;
  uint64_t size;
  uint64_t used_slots;
  std::string handle;
  at::DataPtr refcounted_shared_mem;
  ~CudaIPCRefCountersFile();
  CudaIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr):
  next_offset(0),
  size(size),
  used_slots(0),
  handle(handle),
  refcounted_shared_mem(std::move(data_ptr))
  {};
  int64_t* counter_ptr(){
    return static_cast<int64_t*>(refcounted_shared_mem.get()) + next_offset;
  }
};
} // namespace cuda
} // namespace c10
