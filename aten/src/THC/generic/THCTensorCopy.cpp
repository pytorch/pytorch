#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.cpp"
#else

/* specific methods */

void THCTensor_(copyCPU)(THCState *state, THCTensor *self, struct THTensor *src)
{
  THArgCheck(THCTensor_(nElement)(state, self) == THTensor_(nElement)(src), 2, "sizes do not match");

  {
    THCTensor *selfc = THCTensor_(newContiguous)(state, self);
    src = THTensor_(newContiguous)(src);

    cudaStream_t stream = THCState_getCurrentStream(state);
    THCudaCheck(cudaMemcpyAsync(THCTensor_(data)(state,selfc),
                                src->data<scalar_t>(),
                                THTensor_(nElement)(src) * sizeof(scalar_t),
                                cudaMemcpyHostToDevice,
                                stream));
    THCudaCheck(cudaStreamSynchronize(stream));

    c10::raw::intrusive_ptr::decref(src);
    THCTensor_(freeCopyTo)(state, selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                                  \
  void THCTensor_(copy##TYPEC)(                                               \
      THCState * state, THCTensor * self, struct TH##TYPEC##Tensor * src) {   \
    THArgCheck(                                                               \
        THCTensor_(nElement)(state, self) == TH##TYPEC##Tensor_nElement(src), \
        2,                                                                    \
        "sizes do not match");                                                \
    if (THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {                            \
      THCTensor_(copyCPU)(                                                    \
          state, self, (THTensor*)src); /* cast just removes warnings */      \
    } else {                                                                  \
      at::Tensor srcf_wrap =                                                  \
          at::empty(src->sizes(), caffe2::TypeMeta::Make<scalar_t>());        \
      at::Tensor src_wrap = THTensor_wrap(src);                               \
                                                                              \
      at::_copy_(srcf_wrap, src_wrap);                                        \
      THCTensor_(copyCPU)(state, self, srcf_wrap.unsafeGetTensorImpl());      \
    }                                                                         \
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
    int tensorDevice = THCTensor_(getDevice)(state, src);
    int currentDevice;
    THCudaCheck(cudaGetDevice(&currentDevice));

    if (currentDevice != tensorDevice) {
      THCudaCheck(cudaSetDevice(tensorDevice));
    }
    src = THCTensor_(newContiguous)(state, src);

    cudaStream_t stream = THCState_getCurrentStream(state);
    THCudaCheck(cudaMemcpyAsync(selfc->data<scalar_t>(),
                                THCTensor_(data)(state, src),
                                THCTensor_(nElement)(state, src) * sizeof(scalar_t),
                                cudaMemcpyDeviceToHost,
                                stream));
    THCudaCheck(cudaStreamSynchronize(stream));

    if (currentDevice != tensorDevice) {
      THCudaCheck(cudaSetDevice(currentDevice));
    }

    THCTensor_(free)(state, src);
    THTensor_(freeCopyTo)(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                               \
  void TH_CONCAT_4(TH, TYPEC, Tensor_copyCuda, Real)(                         \
      THCState * state, TH##TYPEC##Tensor * self, struct THCTensor * src) {   \
    THArgCheck(                                                               \
        TH##TYPEC##Tensor_nElement(self) == THCTensor_(nElement)(state, src), \
        2,                                                                    \
        "sizes do not match");                                                \
    if (THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {                            \
      THTensor_(copyCuda)(                                                    \
          state,                                                              \
          (THTensor*)self,                                                    \
          src); /* cast just removes compiler warning */                      \
    } else {                                                                  \
      at::Tensor srcf_wrap =                                                  \
          at::empty(src->sizes(), caffe2::TypeMeta::Make<scalar_t>());        \
      at::Tensor self_wrap = THTensor_wrap(self);                             \
                                                                              \
      THTensor_(copyCuda)(state, srcf_wrap.unsafeGetTensorImpl(), src);       \
      at::_copy_(self_wrap, srcf_wrap);                                       \
    }                                                                         \
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
                              src->data<scalar_t>(),
                              THTensor_(nElement)(src) * sizeof(scalar_t),
                              cudaMemcpyHostToDevice,
                              THCStream_stream(stream)));

  THCudaCheck(THCCachingHostAllocator_recordEvent(THStorage_(data)(THTensor_getStoragePtr(src)), stream));

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
  THCudaCheck(cudaMemcpyAsync(self->data<scalar_t>(),
                              THCTensor_(data)(state, src),
                              THCTensor_(nElement)(state, src) * sizeof(scalar_t),
                              cudaMemcpyDeviceToHost,
                              THCStream_stream(stream)));

  THCudaCheck(THCCachingHostAllocator_recordEvent(THCStorage_(data)(state, THTensor_getStoragePtr(src)), stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

#undef IMPLEMENT_TH_CUDA_TENSOR_COPY
#undef IMPLEMENT_TH_CUDA_TENSOR_COPY_TO

#endif
