#pragma once
#include <c10/core/Allocator.h>

#define REF_COUNTERS_FILE_SIZE 10000

struct CudaIPCReceivedData {
  CudaIPCReceivedData(std::shared_ptr<void> shared_ptr);
  std::shared_ptr<void> shared_ptr_;
};

struct CudaIPCSentData {
  CudaIPCSentData(std::string handle): handle(handle) {};
  ~CudaIPCSentData();
  int64_t get();
  at::DataPtr original_ptr_; // Original mem allocation
  int64_t* counter_ptr; // Reference counter shared memory block
  std::string handle;
  int64_t offset;
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

struct IPCShareData {
  std::string handler;
  uint64_t offset;
  int64_t* counter_ptr;
  IPCShareData(std::string handler,
  uint64_t offset,int64_t* counter_ptr):
   handler(handler),
   offset(offset),
   counter_ptr(counter_ptr){};
};

IPCShareData GetNewRefCounter();
void ReturnRefCounter(std::string handler, uint64_t offset);
void CreateRefCounter(std::string handle, uint64_t size, at::DataPtr data_ptr);

bool HaveNewRefCounter();

struct RefCounterFile {
  uint64_t next_offset;
  uint64_t size;
  uint64_t used_slots;
  std::string handle;
  at::DataPtr refcounted_shared_mem;
  ~RefCounterFile();
  RefCounterFile(std::string handle, uint64_t size, at::DataPtr data_ptr):
  next_offset(0),
  size(size),
  used_slots(0),
  handle(handle),
  refcounted_shared_mem(std::move(data_ptr))
  {};
  int64_t* counter_ptr(){
    return (((int64_t*)(refcounted_shared_mem.get()) + next_offset));
  }
};
