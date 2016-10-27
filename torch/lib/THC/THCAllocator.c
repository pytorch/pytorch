#include "THCAllocator.h"

static void *THCudaHostAllocator_malloc(void* ctx, ptrdiff_t size) {
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

void THCAllocator_init(THCState *state) {
  state->cudaHostAllocator->malloc = &THCudaHostAllocator_malloc;
  state->cudaHostAllocator->realloc = NULL;
  state->cudaHostAllocator->free = &THCudaHostAllocator_free;
}
