#pragma once

#include <THC/THCGeneral.h>

/* Global state of THC. */
struct THCState {
  /* Set of all allocated resources. */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;

  /* Table of enabled peer-to-peer access between directed pairs of GPUs.
     If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;
};
