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

THAllocator THCudaHostAllocator = {
  &THCudaHostAllocator_malloc,
  NULL,
  &THCudaHostAllocator_free
};

static cudaError_t THCIpcAllocator_malloc(void* ctx, void** devPtr, size_t size, cudaStream_t stream)
{
  THError("THCIpcAllocator.malloc() not supported");
  return cudaSuccess;
}

static cudaError_t THCIpcAllocator_free(void* ctx, void* devPtr)
{
  cudaError_t err;
  int prev_device;
  int device = (int)(int64_t)ctx;

  err = cudaGetDevice(&prev_device);
  if (err != cudaSuccess) { return err; }

  err = cudaSetDevice(device);
  if (err != cudaSuccess) { return err; }

  err = cudaIpcCloseMemHandle(devPtr);

  cudaSetDevice(prev_device);
  return err;
}

THCDeviceAllocator THCIpcAllocator = {
  &THCIpcAllocator_malloc,
  NULL,
  &THCIpcAllocator_free,
  NULL,
  NULL
};

static void *THCUVAAllocator_alloc(void* ctx, ptrdiff_t size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  // See J.1.1 of the CUDA_C_Programming_Guide.pdf for UVA and coherence rules
  // on various compute capabilities.
  void* ptr;
  THCudaCheck(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
  return ptr;
}

static void THCUVAAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;
  THCudaCheck(cudaFree(ptr));
}

THAllocator THCUVAAllocator = {
  &THCUVAAllocator_alloc,
  NULL,
  &THCUVAAllocator_free
};
