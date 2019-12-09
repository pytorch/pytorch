#pragma once

#include <THC/THCTensorCopy.h>

template <typename ScalarTypeDst, typename ScalarTypeSrc>
void THC_copyTensor(THCState* state, THCTensor* dst, THCTensor* src);

template <typename ScalarType>
THCTensor *THCTensor_newClone(THCState *state, THCTensor *self);

template <typename ScalarType>
THCTensor *THCTensor_newContiguous(THCState *state, THCTensor *self);

template <typename ScalarType>
void THCTensor_freeCopyTo(THCState *state, THCTensor *self, THCTensor *dst);

template <typename ScalarType>
void THCTensor_copyIgnoringOverlaps(THCState* state, THCTensor* dst, THCTensor* src);
