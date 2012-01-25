#include "THCGeneral.h"
#include "THCTensor.h"

/* specific methods */

void THCudaTensor_copyFloat(THCudaTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THFloatTensor_nElement(src), 2, "sizes do not match"); 

  cudaDeviceSynchronize();

  {
    THCudaTensor *selfc = THCudaTensor_newContiguous(self);
    src = THFloatTensor_newContiguous(src);
  
    THCudaCheck(cudaMemcpy(selfc->storage->data + selfc->storageOffset, src->storage->data + src->storageOffset, THFloatTensor_nElement(src) * sizeof(float), cudaMemcpyHostToDevice));

    THFloatTensor_free(src);
    THCudaTensor_freeCopyTo(selfc, self);
  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                            \
void THCudaTensor_copy##TYPEC(THCudaTensor *self, struct TH##TYPEC##Tensor *src) \
{                                                                       \
  THArgCheck(THCudaTensor_nElement(self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
                                                                        \
  cudaDeviceSynchronize();                                              \
                                                                        \
  {                                                                     \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);        \
                                                                        \
    THFloatTensor_copy##TYPEC(srcf, src);                               \
    THCudaTensor_copyFloat(self, srcf);                                 \
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

void THFloatTensor_copyCuda(THFloatTensor *self, struct THCudaTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THCudaTensor_nElement(src), 2, "sizes do not match"); 

  cudaDeviceSynchronize();

  {
    THFloatTensor *selfc = THFloatTensor_newContiguous(self);
    src = THCudaTensor_newContiguous(src);

    THCudaCheck(cudaMemcpy(selfc->storage->data + selfc->storageOffset, src->storage->data + src->storageOffset, THCudaTensor_nElement(src) * sizeof(float), cudaMemcpyDeviceToHost));

    THCudaTensor_free(src);
    THFloatTensor_freeCopyTo(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                         \
  void TH##TYPEC##Tensor_copyCuda(TH##TYPEC##Tensor *self, struct THCudaTensor *src) \
  {                                                                     \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THCudaTensor_nElement(src), 2, "sizes do not match"); \
                                                                        \
    cudaDeviceSynchronize();                                            \
                                                                        \
    {                                                                   \
      THLongStorage *size = THCudaTensor_newSizeOf(src);                \
      THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);      \
                                                                        \
      THFloatTensor_copyCuda(srcf, src);                                \
      TH##TYPEC##Tensor_copyFloat(self, srcf);                          \
                                                                        \
      THLongStorage_free(size);                                         \
      THFloatTensor_free(srcf);                                         \
    }                                                                   \
  }

IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Double)

void THCudaTensor_copyCuda(THCudaTensor *self, THCudaTensor *src)
{
  THCudaTensor_copy(self, src);
}
