#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorageCopy.cpp"
#else

#include <c10/cuda/CUDAFunctions.h>

void THCStorage_(copyCPU)(THCState *state, THCStorage *self, struct THStorage *src)
{
  THArgCheck(self->nbytes() == src->nbytes(), 2, "size does not match");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::memcpy_and_sync(THCStorage_(data)(state, self),
      THStorage_(data)(src),
      self->nbytes(),
      cudaMemcpyHostToDevice,
      stream);

}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                                 \
  void THCStorage_(copy##TYPEC)(                                              \
      THCState * state, THCStorage * self, struct TH##TYPEC##Storage * src) { \
    at::Tensor selfTensor = tensor_reclaim(                                   \
      THCTensor_(newWithStorage1d)(                                           \
          state, self, 0, src->nbytes() / sizeof(scalar_t), 1));              \
    at::Tensor srcTensor = tensor_reclaim(                                    \
        TH##TYPEC##Tensor_newWithStorage1d(                                   \
            src, 0, src->nbytes() / sizeof(scalar_t), 1));                    \
    selfTensor.copy_(srcTensor);                                              \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)

void THStorage_(copyCuda)(THCState *state, THStorage *self, struct THCStorage *src)
{
  THArgCheck(self->nbytes() == src->nbytes(), 2, "size does not match");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::memcpy_and_sync(
      THStorage_(data)(self),
      THCStorage_(data)(state, src),
      self->nbytes(),
      cudaMemcpyDeviceToHost,
      stream);
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                               \
  void TH_CONCAT_4(TH, TYPEC, Storage_copyCuda, Real)(                        \
      THCState * state, TH##TYPEC##Storage * self, struct THCStorage * src) { \
    at::Tensor selfTensor = tensor_reclaim(                                   \
        TH##TYPEC##Tensor_newWithStorage1d(                                   \
            self, 0, self->nbytes() / sizeof(scalar_t), 1));                  \
    at::Tensor srcTensor = tensor_reclaim(                                    \
        THCTensor_(newWithStorage1d)(                                         \
            state, src, 0, src->nbytes() / sizeof(scalar_t), 1));             \
    selfTensor.copy_(srcTensor);                                              \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)

#undef TH_CUDA_STORAGE_IMPLEMENT_COPY
#undef TH_CUDA_STORAGE_IMPLEMENT_COPYTO

#endif
