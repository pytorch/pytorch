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

at::DataPtr THCIpcDeleter::makeDataPtr(std::shared_ptr<void> basePtr, void* data, int device) {
  // The dynamic allocation here is a bit unfortunate
  int cur_device;
  THCudaCheck(cudaGetDevice(&cur_device));
  auto* context = new THCIpcDeleter(basePtr, device);
  return {data, context, &deleteTHCIpcDeleter, at::Device(at::DeviceType::CUDA, cur_device)};
}

THCIpcDeleter::THCIpcDeleter(std::shared_ptr<void> basePtr, int device)
    : basePtr_(basePtr), device_(device) {}
