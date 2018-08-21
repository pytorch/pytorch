#pragma once

#include "THCGeneral.h"

/* Global state of THC. */
struct THCState {
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  /* Set of all allocated resources. blasHandles and sparseHandles do not have
     a default and must be explicitly initialized. We always initialize 1
     blasHandle and 1 sparseHandle but we can use more.
  */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;
  int numUserBlasHandles;
  int numUserSparseHandles;

  /* Allocator using cudaMallocHost. */
  // NB: These allocators (specifically, cudaHostAllocator) MUST implement
  // maybeGlobalBoundDeleter, because we have a few use-cases where we need to
  // do raw allocations with them (for Thrust).
  // TODO: Make this statically obvious
  at::Allocator* cudaHostAllocator;
  at::Allocator* cudaUVAAllocator;
  at::Allocator* cudaDeviceAllocator;

  /* Index of the current selected BLAS handle. The actual BLAS handle used
     depends on the current device. */
  THCThreadLocal/*<int>*/ currentPerDeviceBlasHandle;
  /* Index of the current selected sparse handle. The actual sparse handle used
     depends on the current device. */
  THCThreadLocal/*<int>*/ currentPerDeviceSparseHandle;

  /* Table of enabled peer-to-peer access between directed pairs of GPUs.
     If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;

  /* Is direct cross-kernel p2p access allowed? Normally, only cross-GPU
     copies are allowed via p2p if p2p access is enabled at all for
     the pair of GPUs in question, but if this flag is true, then
     all cross-GPU access checks are disabled, allowing kernels to
     directly access memory on another GPUs.
     Note that p2p access must exist and be enabled for the pair of
     GPUs in question. */
  int p2pKernelAccessEnabled;

  void (*cutorchGCFunction)(void *data);
  void *cutorchGCData;
  ptrdiff_t heapSoftmax;
  ptrdiff_t heapDelta;
};
