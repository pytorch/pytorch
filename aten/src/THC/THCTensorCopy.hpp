#pragma once

#include "THCTensorCopy.h"

template <typename ScalarTypeDst, typename ScalarTypeSrc>
void THC_copyTensor(THCState* state, _THCTensor* dst, _THCTensor* src);

template <typename ScalarType>
_THCTensor *THCTensor_newClone(THCState *state, _THCTensor *self);

template <typename ScalarType>
_THCTensor *THCTensor_newContiguous(THCState *state, _THCTensor *self);

template <typename ScalarType>
void THCTensor_freeCopyTo(THCState *state, _THCTensor *self, _THCTensor *dst);

template <typename ScalarType>
void THCTensor_copyIgnoringOverlaps(THCState* state, _THCTensor* dst, _THCTensor* src);
