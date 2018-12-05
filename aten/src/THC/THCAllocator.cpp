#include "THCAllocator.h"

THCIpcDeleter::~THCIpcDeleter() {
  int prev_device;
  THCudaCheck(cudaGetDevice(&prev_device));
  THCudaCheck(cudaSetDevice(device_));
  THCudaCheck(cudaSetDevice(prev_device));
}

void deleteTHCIpcDeleter(void* ptr) {
  delete static_cast<THCIpcDeleter*>(ptr);
}

// Refer to NB [CUDA IPC and the caching allocator] for more details
// basePtr - device ptr of allocated CUDA memory region. This memory region
//           might contain storages of different types.
// data    - ptr to where the storage (of a single type) should start.
// device  - device of memory
// Here basePtr should be saved in the struct, while data should be used to
// construct the new storage.
// Every time a storage on the memory region go out of scope, the ref_count
// of basePtr will be decreased 1, until it's closed in its deleter (calling
// cudaIpoCloseMemHandle) when ref_count is 0.
at::DataPtr THCIpcDeleter::makeDataPtr(std::shared_ptr<void> basePtr, void* data, int device) {
  // The dynamic allocation here is a bit unfortunate
  int cur_device;
  THCudaCheck(cudaGetDevice(&cur_device));
  auto* context = new THCIpcDeleter(std::move(basePtr), device);
  return {data, context, &deleteTHCIpcDeleter, at::Device(at::DeviceType::CUDA, cur_device)};
}

THCIpcDeleter::THCIpcDeleter(std::shared_ptr<void> basePtr, int device)
    : basePtr_(std::move(basePtr)), device_(device) {}
