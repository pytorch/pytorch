#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.c"
#else

/* specific methods */

void THCTensor_(copyCPU)(THCState *state, THCTensor *self, struct THTensor *src)
{
  THArgCheck(THCTensor_(nElement)(state, self) == THTensor_(nElement)(src), 2, "sizes do not match");

  {
    THCTensor *selfc = THCTensor_(newContiguous)(state, self);
    src = THTensor_(newContiguous)(src);

    THCudaCheck(cudaMemcpy(THCTensor_(data)(state,selfc),
                           THTensor_(data)(src),
                           THTensor_(nElement)(src) * sizeof(real),
                           cudaMemcpyHostToDevice));

    THTensor_(free)(src);
    THCTensor_(freeCopyTo)(state, selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                            \
void THCTensor_(copy##TYPEC)(THCState *state, THCTensor *self, struct TH##TYPEC##Tensor *src)                \
{                                                                       \
  THArgCheck(THCTensor_(nElement)(state, self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
  if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {               \
    THCTensor_(copyCPU)(state, self, (THTensor*) src);  /* cast just removes warnings */                     \
  } else {                                                              \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THTensor *srcf = THTensor_(newWithSize)(size, NULL);                \
                                                                        \
    THTensor_(copy##TYPEC)(srcf, src);                                  \
    THCTensor_(copyCPU)(state, self, srcf);                             \
                                                                        \
    THLongStorage_free(size);                                           \
    THTensor_(free)(srcf);                                              \
  }                                                                     \
}

IMPLEMENT_TH_CUDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Float)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Double)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Half)

/* copyCuda */

void THTensor_(copyCuda)(THCState *state, THTensor *self, struct THCTensor *src)
{
  THArgCheck(THTensor_(nElement)(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");

  {
    THTensor *selfc = THTensor_(newContiguous)(self);
    src = THCTensor_(newContiguous)(state, src);

    THCudaCheck(cudaMemcpy(THTensor_(data)(selfc),
                           THCTensor_(data)(state, src),
                           THCTensor_(nElement)(state, src) * sizeof(real),
                           cudaMemcpyDeviceToHost));

    THCTensor_(free)(state, src);
    THTensor_(freeCopyTo)(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                           \
  void TH_CONCAT_4(TH,TYPEC,Tensor_copyCuda,Real)(THCState *state, TH##TYPEC##Tensor *self, struct THCTensor *src) \
  {                                                                       \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");       \
    if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {   \
      THTensor_(copyCuda)(state, (THTensor*) self, src);  /* cast just removes compiler warning */                   \
    } else {                                                              \
      THLongStorage *size = THCTensor_(newSizeOf)(state, src);            \
      THTensor *srcf = THTensor_(newWithSize)(size, NULL);                \
                                                                          \
      THTensor_(copyCuda)(state, srcf, src);                              \
      TH_CONCAT_4(TH,TYPEC,Tensor_copy,Real)(self, srcf);                 \
                                                                          \
      THLongStorage_free(size);                                           \
      THTensor_(free)(srcf);                                              \
    }                                                                     \
  }

IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Float)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Double)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Half)

void THCTensor_(copyCuda)(THCState *state, THCTensor *self, THCTensor *src)
{
  THCTensor_(copy)(state, self, src);
}

void THCTensor_(copyAsyncCPU)(THCState *state, THCTensor *self, struct THTensor *src)
{
  THArgCheck(THCTensor_(nElement)(state, self) == THTensor_(nElement)(src), 2, "sizes do not match");
  THArgCheck(THCTensor_(isContiguous)(state, self), 2, "Target tensor must be contiguous");
  THArgCheck(THTensor_(isContiguous)(src), 3, "Source tensor must be contiguous");

  if (THCTensor_(nElement)(state, self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCTensor_(getDevice)(state, self);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  THCStream *stream  = THCState_getStream(state);
  THCudaCheck(cudaMemcpyAsync(THCTensor_(data)(state, self),
                              THTensor_(data)(src),
                              THTensor_(nElement)(src) * sizeof(real),
                              cudaMemcpyHostToDevice,
                              stream->stream));

  THCudaCheck(THCCachingHostAllocator_recordEvent(src->storage->data, stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

void THTensor_(copyAsyncCuda)(THCState *state, THTensor *self, struct THCTensor *src)
{
  THArgCheck(THTensor_(nElement)(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");
  THArgCheck(THTensor_(isContiguous)(self), 2, "Target tensor must be contiguous");
  THArgCheck(THCTensor_(isContiguous)(state, src), 3, "Source tensor must be contiguous");

  if (THTensor_(nElement)(self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCTensor_(getDevice)(state, src);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  THCStream *stream = THCState_getStream(state);
  THCudaCheck(cudaMemcpyAsync(THTensor_(data)(self),
                              THCTensor_(data)(state, src),
                              THCTensor_(nElement)(state, src) * sizeof(real),
                              cudaMemcpyDeviceToHost,
                              stream->stream));

  THCudaCheck(THCCachingHostAllocator_recordEvent(src->storage->data, stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

#undef IMPLEMENT_TH_CUDA_TENSOR_COPY
#undef IMPLEMENT_TH_CUDA_TENSOR_COPY_TO

#endif
