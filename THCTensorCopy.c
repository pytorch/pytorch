#include "THCTensorCopy.h"
#include "THCGeneral.h"
#include "THCTensor.h"

/* specific methods */

void THCudaTensor_copyFloat(THCState *state, THCudaTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THCudaTensor_nElement(state, self) == THFloatTensor_nElement(src), 2, "sizes do not match");

  {
    THCudaTensor *selfc = THCudaTensor_newContiguous(state, self);
    src = THFloatTensor_newContiguous(src);

    THCudaCheck(cudaMemcpy(THCudaTensor_data(state, selfc),
                           THFloatTensor_data(src),
                           THFloatTensor_nElement(src) * sizeof(float),
                           cudaMemcpyHostToDevice));

    THFloatTensor_free(src);
    THCudaTensor_freeCopyTo(state, selfc, self);
  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                            \
void THCudaTensor_copy##TYPEC(THCState *state, THCudaTensor *self, struct TH##TYPEC##Tensor *src) \
{                                                                       \
  THArgCheck(THCudaTensor_nElement(state, self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
                                                                        \
  {                                                                     \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);        \
                                                                        \
    THFloatTensor_copy##TYPEC(srcf, src);                               \
    THCudaTensor_copyFloat(state, self, srcf);                                 \
                                                                        \
    THLongStorage_free(size);                                           \
    THFloatTensor_free(srcf);                                           \
  }                                                                     \
}

IMPLEMENT_TH_CUDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Double)

/* copyCuda */

void THFloatTensor_copyCuda(THCState *state, THFloatTensor *self, struct THCudaTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THCudaTensor_nElement(state, src), 2, "sizes do not match");

  {
    THFloatTensor *selfc = THFloatTensor_newContiguous(self);
    src = THCudaTensor_newContiguous(state, src);

    THCudaCheck(cudaMemcpy(THFloatTensor_data(selfc),
                           THCudaTensor_data(state, src),
                           THCudaTensor_nElement(state, src) * sizeof(float),
                           cudaMemcpyDeviceToHost));

    THCudaTensor_free(state, src);
    THFloatTensor_freeCopyTo(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                                                          \
  void TH##TYPEC##Tensor_copyCuda(THCState *state, TH##TYPEC##Tensor *self, struct THCudaTensor *src) \
  {                                                                                                      \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THCudaTensor_nElement(state, src), 2, "sizes do not match"); \
                                                                                                         \
    {                                                                                                    \
      THLongStorage *size = THCudaTensor_newSizeOf(state, src);                                          \
      THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);                                       \
                                                                                                         \
      THFloatTensor_copyCuda(state, srcf, src);                                                          \
      TH##TYPEC##Tensor_copyFloat(self, srcf);                                                           \
                                                                                                         \
      THLongStorage_free(size);                                                                          \
      THFloatTensor_free(srcf);                                                                          \
    }                                                                                                    \
  }

IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Double)

void THCudaTensor_copyCuda(THCState *state, THCudaTensor *self, THCudaTensor *src)
{
  THCudaTensor_copy(state, self, src);
}

void THCudaTensor_copyAsyncFloat(THCState *state, THCudaTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THCudaTensor_nElement(state, self) == THFloatTensor_nElement(src), 2, "sizes do not match");
  THArgCheck(THCudaTensor_isContiguous(state, self), 2, "Target tensor must be contiguous");
  THArgCheck(THFloatTensor_isContiguous(src), 3, "Source tensor must be contiguous");

  if (THCudaTensor_nElement(state, self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCudaTensor_getDevice(state, self);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  THCudaCheck(cudaMemcpyAsync(THCudaTensor_data(state, self),
                              THFloatTensor_data(src),
                              THFloatTensor_nElement(src) * sizeof(float),
                              cudaMemcpyHostToDevice,
                              THCState_getDeviceStream(state, tensorDevice,
                                                       THCState_getCurrentStreamIndex(state))));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

void THFloatTensor_copyAsyncCuda(THCState *state, THFloatTensor *self, struct THCudaTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THCudaTensor_nElement(state, src), 2, "sizes do not match");
  THArgCheck(THFloatTensor_isContiguous(self), 2, "Target tensor must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(state, src), 3, "Source tensor must be contiguous");

  if (THFloatTensor_nElement(self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCudaTensor_getDevice(state, src);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  THCudaCheck(cudaMemcpyAsync(THFloatTensor_data(self),
                              THCudaTensor_data(state, src),
                              THCudaTensor_nElement(state, src) * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              THCState_getDeviceStream(state, tensorDevice,
                                                       THCState_getCurrentStreamIndex(state))));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}
