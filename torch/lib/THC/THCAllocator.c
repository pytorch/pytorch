#include "THCAllocator.h"

static void *THCudaHostAllocator_alloc(void* ctx, ptrdiff_t size) {
  void* ptr;

  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  THCudaCheck(cudaMallocHost(&ptr, size));

  return ptr;
}

static void THCudaHostAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;

  THCudaCheck(cudaFreeHost(ptr));
}

static void *THCudaHostAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  THCudaHostAllocator_free(ctx, ptr);

  if (size == 0) return NULL;

  THCudaCheck(cudaMallocHost(&ptr, size));

  return ptr;
}

void THCAllocator_init(THAllocator *cudaHostAllocator) {
  cudaHostAllocator->malloc = &THCudaHostAllocator_alloc;
  cudaHostAllocator->realloc = &THCudaHostAllocator_realloc;
  cudaHostAllocator->free = &THCudaHostAllocator_free;
}
