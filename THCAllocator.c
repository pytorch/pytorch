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

void THCAllocator_init(THAllocator *cudaHostAllocator) {
  cudaHostAllocator->malloc = &THCudaHostAllocator_alloc;
  cudaHostAllocator->realloc = NULL;
  cudaHostAllocator->free = &THCudaHostAllocator_free;
}
