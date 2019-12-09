#include <THC/THCAllocator.h>

THCIpcDeleter::~THCIpcDeleter() {}

void deleteTHCIpcDeleter(void* ptr) {
  delete static_cast<THCIpcDeleter*>(ptr);
}

// Refer to Note [CUDA IPC and the caching allocator] for more details
// basePtr - device ptr of a single cudaMalloc allocation; this may be a large
//           block of memory which is managed by the caching allocator.
// data    - ptr to where the storage (of a single type) should start.
// Invariant: data must lie within the CUDA memory allocation represented by
//   basePtr.
// Here basePtr should be saved in the struct, while data should be used to
// construct the new storage.
// Every time a storage referring to the IPC memory region goes out of scope,
// the reference count on the memory region will be decreased by one, until
// it's zero, at which point IPC memory region is closed (by calling
// cudaIpcCloseMemHandle).
at::DataPtr THCIpcDeleter::makeDataPtr(std::shared_ptr<void> basePtr, void* data) {
  // The dynamic allocation here is a bit unfortunate
  int cur_device;
  THCudaCheck(cudaGetDevice(&cur_device));
  auto* context = new THCIpcDeleter(std::move(basePtr));
  return {data, context, &deleteTHCIpcDeleter, at::Device(at::DeviceType::CUDA, cur_device)};
}

THCIpcDeleter::THCIpcDeleter(std::shared_ptr<void> basePtr)
    : basePtr_(std::move(basePtr)) {}
