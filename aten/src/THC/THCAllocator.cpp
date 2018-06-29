#include "THCAllocator.h"

THCudaHostDeleter THCudaHostDeleter::singleton_;

struct THCudaHostAllocator : public at::Allocator {
  std::unique_ptr<void, at::BoundDeleter> allocate(size_t size) const override {
    if (size == 0) return nullptr;
    void* ptr;
    THCudaCheck(cudaMallocHost(&ptr, size));
    return {ptr, THCudaHostDeleter::make()};
  }
  at::BoundDeleter maybeGlobalBoundDeleter() const override {
    return THCudaHostDeleter::make();
  }
};

static THCudaHostAllocator th_cuda_host_allocator;
at::Allocator* getTHCudaHostAllocator() {
  return &th_cuda_host_allocator;
}

THCIpcDeleter THCIpcDeleter::singleton_;
THCUVADeleter THCUVADeleter::singleton_;

struct THCUVAAllocator : public at::Allocator {
  std::unique_ptr<void, at::BoundDeleter> allocate(size_t size) const override {
    if (size == 0) return nullptr;

    // See J.1.1 of the CUDA_C_Programming_Guide.pdf for UVA and coherence rules
    // on various compute capabilities.
    void* ptr;
    THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
    return {ptr, THCUVADeleter::make()};
  }
  at::BoundDeleter maybeGlobalBoundDeleter() const override {
    return THCUVADeleter::make();
  }
};

static THCUVAAllocator thc_uva_allocator;
at::Allocator* getTHCUVAAllocator() {
  return &thc_uva_allocator;
}
