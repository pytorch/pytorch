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

static cudaError_t THCIpcAllocator_malloc(void* ctx, void** devPtr, size_t size, cudaStream_t stream)
{
  THError("THCIpcAllocator.malloc() not supported");
  return cudaSuccess;
}

static cudaError_t THCIpcAllocator_free(void* ctx, void* devPtr)
{
  return cudaIpcCloseMemHandle(devPtr);
}

THCDeviceAllocator THCIpcAllocator = {
  &THCIpcAllocator_malloc,
  NULL,
  &THCIpcAllocator_free,
  NULL,
  NULL
};
