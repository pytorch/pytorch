#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.cu"
#else

THC_API void
THCTensor_(copy)(THCState* state, THCTensor* dst, THCTensor* src) {
  if (dst == src) return;
  THC_copyTensor<real, real>(state, dst, src);
}

template <>
THCTensor *THCTensor_newClone<real>(THCState *state, THCTensor *self) {
  THCTensor *tensor = THCTensor_new(state, THTensor_getStoragePtr(self)->scalar_type());
  THCTensor_resizeAs(state, tensor, self);
  THC_copyTensor<real, real>(state, tensor, self);
  return tensor;
}

template <>
THCTensor *THCTensor_newContiguous<real>(THCState *state, THCTensor *self)
{
  if(!THCTensor_isContiguous(state, self)) {
    return THCTensor_newClone<real>(state, self);
  } else {
    THCTensor_retain(state, self);
    return self;
  }
}


template <>
void THCTensor_freeCopyTo<real>(THCState *state, THCTensor *self, THCTensor *dst) {
  if(self != dst)
    THC_copyTensor<real, real>(state, dst, self);

  THCTensor_free(state, self);
}

template <>
void THCTensor_copyIgnoringOverlaps<real>(THCState* state, THCTensor* dst, THCTensor* src) {
  // Called when we are copying into an overlapping index `dst`, but
  // we don't care which writer wins. Hacky but it works.
  // This is itself invoked by pointwiseApply2 / THCTensor_copy in
  // case that there are write overlaps.
  // FIXME: really, overlapping writes should be illegal/an error in Torch
  THC_pointwiseApply2<real, real>(
    state, dst, src,
    CopyOp<real, real>(),
    ReadOnly, /* ignore overwrites */
    ReadOnly);
}

THC_API void
THCTensor_(copyIgnoringOverlaps)(THCState* state, THCTensor* dst, THCTensor* src) {
  THCTensor_copyIgnoringOverlaps<real>(state, dst, src);
}

#define IMPLEMENT_THC_CUDA_TENSOR_COPY(TYPEC, TYPECUDA, SCALARC)        \
  THC_API void                                                          \
  THCTensor_(copyCuda##TYPEC)(THCState *state,                          \
                              THCTensor *self,                          \
                              THCuda##TYPECUDA##Tensor *src) {          \
    THC_copyTensor<real, SCALARC>(state, self, src); \
  }

IMPLEMENT_THC_CUDA_TENSOR_COPY(Byte, Byte, uint8_t)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Char, Char, int8_t)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Short, Short, int16_t)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Int, Int, int32_t)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Long, Long, int64_t)
// THCudaTensor aka the non-existent THCudaFloatTensor
IMPLEMENT_THC_CUDA_TENSOR_COPY(Float, , float)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Double, Double, double)
IMPLEMENT_THC_CUDA_TENSOR_COPY(Half, Half, half)

#undef IMPLEMENT_THC_CUDA_TENSOR_COPY

#endif
