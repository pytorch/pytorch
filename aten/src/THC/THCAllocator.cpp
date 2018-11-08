#include "THCAllocator.h"

THCIpcDeleter::~THCIpcDeleter() {
  int prev_device;
  THCudaCheck(cudaGetDevice(&prev_device));
  THCudaCheck(cudaSetDevice(device_));
  THCudaCheck(cudaIpcCloseMemHandle(data_));
  THCudaCheck(cudaSetDevice(prev_device));
}

void deleteTHCIpcDeleter(void* ptr) {
  delete static_cast<THCIpcDeleter*>(ptr);
}

at::DataPtr THCIpcDeleter::makeDataPtr(void* data, int device) {
  // The dynamic allocation here is a bit unfortunate
  int cur_device;
  THCudaCheck(cudaGetDevice(&cur_device));
  auto* context = new THCIpcDeleter(data, device);
  return {data, context, &deleteTHCIpcDeleter, at::Device(at::DeviceType::CUDA, cur_device)};
}
