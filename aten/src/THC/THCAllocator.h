#ifndef THC_ALLOCATOR_INC
#define THC_ALLOCATOR_INC

#include "THCGeneral.h"

THC_API THAllocator* getTHCudaHostAllocator();
THC_API THAllocator* getTHCUVAAllocator();
// IPC doesn't support (re)allocation

#ifdef __cplusplus
struct THCudaHostDeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    if (!ptr) return;

    THCudaCheck(cudaFreeHost(ptr));
  }
  static at::BoundDeleter make() {
    return {&singleton_, nullptr};
  }
private:
  static THCudaHostDeleter singleton_;
};

struct THCIpcDeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    int prev_device;
    int device = (int)(int64_t)ctx;

    THCudaCheck(cudaGetDevice(&prev_device));
    THCudaCheck(cudaSetDevice(device));
    THCudaCheck(cudaIpcCloseMemHandle(ptr));
    THCudaCheck(cudaSetDevice(prev_device));
  }

  static at::BoundDeleter make(int device) {
    // TODO: Do this properly with intptr_t (but is it portable enough?)
    return {&singleton_, (void*)(int64_t)device};
  }
private:
  static THCIpcDeleter singleton_;
};

struct THCUVADeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    if (!ptr) return;
    THCudaCheck(cudaFree(ptr));
  }
  static at::BoundDeleter make() {
    return {&singleton_, nullptr};
  }
private:
  static THCUVADeleter singleton_;
};
#endif

#endif
