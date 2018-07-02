#include "THCAllocator.h"

struct THCudaHostAllocator : public at::Allocator {
  void* allocate(void* ctx, size_t size) const override {
    void* ptr;

    if (size < 0) THError("Invalid memory size: %ld", size);

    if (size == 0) return NULL;

    THCudaCheck(cudaMallocHost(&ptr, size));

    return ptr;
  }
  void deallocate(void* ctx, void* ptr) const override {
    if (!ptr) return;

    THCudaCheck(cudaFreeHost(ptr));
  }
};

static THCudaHostAllocator th_cuda_host_allocator;
at::Allocator* getTHCudaHostAllocator() {
  return &th_cuda_host_allocator;
}

struct THCIpcAllocator : public at::Allocator {
  void* allocate(void* ctx, size_t size) const override {
    AT_ERROR("THCIpcAllocator.malloc() not supported");
  }
  void deallocate(void* ctx, void* ptr) const override {
    int prev_device;
    int device = (int)(int64_t)ctx;

    THCudaCheck(cudaGetDevice(&prev_device));
    THCudaCheck(cudaSetDevice(device));
    THCudaCheck(cudaIpcCloseMemHandle(ptr));
    THCudaCheck(cudaSetDevice(prev_device));
  }
};

static THCIpcAllocator thc_ipc_allocator;
at::Allocator* getTHCIpcAllocator() {
  return &thc_ipc_allocator;
}

struct THCUVAAllocator : public at::Allocator {
  void* allocate(void* ctx, size_t size) const override {
    if (size < 0) THError("Invalid memory size: %ld", size);

    if (size == 0) return NULL;

    // See J.1.1 of the CUDA_C_Programming_Guide.pdf for UVA and coherence rules
    // on various compute capabilities.
    void* ptr;
    THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    return ptr;
  }
  void deallocate(void* ctx, void* ptr) const override {
    if (!ptr) return;
    THCudaCheck(cudaFree(ptr));
  }
};

static THCUVAAllocator thc_uva_allocator;
at::Allocator* getTHCUVAAllocator() {
  return &thc_uva_allocator;
}
