#include "THCAllocator.h"

static void THCudaHostDeleter(void* ptr) {
  THCudaCheck(cudaFreeHost(ptr));
}

struct THCudaHostAllocator : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void* ptr = nullptr;
    if (size != 0) {
      THCudaCheck(cudaMallocHost(&ptr, size));
    }
    return {ptr, ptr, &THCudaHostDeleter, at::DeviceType::CPU};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &THCudaHostDeleter;
  }
};

static THCudaHostAllocator th_cuda_host_allocator;
at::Allocator* getTHCudaHostAllocator() {
  return &th_cuda_host_allocator;
}

static void THCUVADeleter(void* ptr) {
  THCudaCheck(cudaFree(ptr));
}

struct THCUVAAllocator : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    // See J.1.1 of the CUDA_C_Programming_Guide.pdf for UVA and coherence rules
    // on various compute capabilities.
    void* ptr = nullptr;
    if (size != 0) {
      THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    }
    return {ptr, ptr, &THCUVADeleter, at::DeviceType::CPU};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &THCUVADeleter;
  }
};

static THCUVAAllocator thc_uva_allocator;
at::Allocator* getTHCUVAAllocator() {
  return &thc_uva_allocator;
}


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
