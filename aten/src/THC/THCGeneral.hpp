#pragma once

#include <THC/THCGeneral.h>

/* Global state of THC. */
struct THCState {
  /* Set of all allocated resources. */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;

  /* Allocator using cudaMallocHost. */
  // NB: cudaHostAllocator MUST implement maybeGlobalBoundDeleter, because we have
  // a few use-cases where we need to do raw allocations with them (for Thrust).
  // TODO: Make this statically obvious
  at::Allocator* cudaHostAllocator;

  /* Table of enabled peer-to-peer access between directed pairs of GPUs.
     If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;
};
