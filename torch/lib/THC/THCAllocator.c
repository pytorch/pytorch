#include "THCAllocator.h"

static void *THCudaHostAllocator_alloc(void* ctx, long size) {
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

static void *THCudaHostAllocator_realloc(void* ctx, void* ptr, long size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  THCudaHostAllocator_free(ctx, ptr);

  if (size == 0) return NULL;

  THCudaCheck(cudaMallocHost(&ptr, size));

  return ptr;
}

void THCAllocator_init(THCState *state) {
  state->cudaHostAllocator = (THAllocator*)malloc(sizeof(THAllocator));
  state->cudaHostAllocator->malloc = &THCudaHostAllocator_alloc;
  state->cudaHostAllocator->realloc = &THCudaHostAllocator_realloc;
  state->cudaHostAllocator->free = &THCudaHostAllocator_free;
}

void THCAllocator_shutdown(THCState *state) {
  free(state->cudaHostAllocator);
}
