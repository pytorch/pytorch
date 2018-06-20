#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.cu"
#else

#include "THCThrustAllocator.cuh"
#include "THCTensor.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THCSTensor *THCSTensor_(newCoalesce)(THCState *state, THCSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

// forceClone is intended to use as a boolean, if set, the result will forced to
// be a clone of self.
THCIndexTensor* THCSTensor_(newFlattenedIndices)(THCState *state, THCSTensor *self, int forceClone) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

// In place transpose
void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int d1, int d2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THCSTensor_(getDevice)(THCState* state, const THCSTensor* tensor) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

#endif
