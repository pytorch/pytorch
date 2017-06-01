#include "torch/csrc/cuda/THCP.h"

// Declare/Define the expansion functions that have THCState.  Note that we
// still need to define the CPU-type verisons because the copy functions that
// copy from GPU to CPU type have a THCState.
#include "torch/csrc/expand_utils.h"
#include "torch/csrc/expand_utils-inl.h"

template <>
THCudaTensor *newForExpand(THCState *s) {
  return THCudaTensor_new(s);
}

template <>
THCudaDoubleTensor *newForExpand(THCState *s) {
  return THCudaDoubleTensor_new(s);
}

#ifdef CUDA_HALF_TENSOR
template <>
THCudaHalfTensor *newForExpand(THCState *s) {
  return THCudaHalfTensor_new(s);
}
#endif // CUDA_HALF_TENSOR

template <>
THCudaByteTensor *newForExpand(THCState *s) {
  return THCudaByteTensor_new(s);
}

template <>
THCudaCharTensor *newForExpand(THCState *s) {
  return THCudaCharTensor_new(s);
}

template <>
THCudaShortTensor *newForExpand(THCState *s) {
  return THCudaShortTensor_new(s);
}

template <>
THCudaIntTensor *newForExpand(THCState *s) {
  return THCudaIntTensor_new(s);
}

template <>
THCudaLongTensor *newForExpand(THCState *s) {
  return THCudaLongTensor_new(s);
}

template<>
int expand(THCState *s, THCudaTensor *r, THCudaTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaDoubleTensor *r, THCudaDoubleTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaDoubleTensor_expand(s, r, tensor, sizes, raiseErrors);
}

#ifdef CUDA_HALF_TENSOR
template<>
int expand(THCState *s, THCudaHalfTensor *r, THCudaHalfTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaHalfTensor_expand(s, r, tensor, sizes, raiseErrors);
}
#endif // CUDA_HALF_TENSOR

template<>
int expand(THCState *s, THCudaByteTensor *r, THCudaByteTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaByteTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaCharTensor *r, THCudaCharTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaCharTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaShortTensor *r, THCudaShortTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaShortTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaIntTensor *r, THCudaIntTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaIntTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template<>
int expand(THCState *s, THCudaLongTensor *r, THCudaLongTensor *tensor, THLongStorage *sizes, int raiseErrors) {
  return THCudaLongTensor_expand(s, r, tensor, sizes, raiseErrors);
}

template <>
int expand2(THCState *s, THCudaTensor *r1, THCudaTensor *r2,
            THCudaTensor *e1, THCudaTensor *e2, int raiseErrors) {
  return THCudaTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(THCState *s, THCudaDoubleTensor *r1, THCudaDoubleTensor *r2,
            THCudaDoubleTensor *e1, THCudaDoubleTensor *e2, int raiseErrors) {
  return THCudaDoubleTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

#ifdef CUDA_HALF_TENSOR
template <>
int expand2(THCState *s, THCudaHalfTensor *r1, THCudaHalfTensor *r2,
            THCudaHalfTensor *e1, THCudaHalfTensor *e2, int raiseErrors) {
  return THCudaHalfTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}
#endif // CUDA_HALF_TENSOR

template <>
int expand2(THCState *s, THCudaByteTensor *r1, THCudaByteTensor *r2,
            THCudaByteTensor *e1, THCudaByteTensor *e2, int raiseErrors) {
  return THCudaByteTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(THCState *s, THCudaCharTensor *r1, THCudaCharTensor *r2,
            THCudaCharTensor *e1, THCudaCharTensor *e2, int raiseErrors) {
  return THCudaCharTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(THCState *s, THCudaShortTensor *r1, THCudaShortTensor *r2,
            THCudaShortTensor *e1, THCudaShortTensor *e2, int raiseErrors) {
  return THCudaShortTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(THCState *s, THCudaIntTensor *r1, THCudaIntTensor *r2,
            THCudaIntTensor *e1, THCudaIntTensor *e2, int raiseErrors) {
  return THCudaIntTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

template <>
int expand2(THCState *s, THCudaLongTensor *r1, THCudaLongTensor *r2,
            THCudaLongTensor *e1, THCudaLongTensor *e2, int raiseErrors) {
  return THCudaLongTensor_expand2(s, r1, r2, e1, e2, raiseErrors);
}

template <>
int expand3(THCState *s, THCudaTensor *r1, THCudaTensor *r2, THCudaTensor *r3,
            THCudaTensor *e1, THCudaTensor *e2, THCudaTensor *e3, int raiseErrors) {
  return THCudaTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(THCState *s, THCudaDoubleTensor *r1, THCudaDoubleTensor *r2, THCudaDoubleTensor *r3,
            THCudaDoubleTensor *e1, THCudaDoubleTensor *e2, THCudaDoubleTensor *e3, int raiseErrors) {
  return THCudaDoubleTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}

#ifdef CUDA_HALF_TENSOR
template <>
int expand3(THCState *s, THCudaHalfTensor *r1, THCudaHalfTensor *r2, THCudaHalfTensor *r3,
            THCudaHalfTensor *e1, THCudaHalfTensor *e2, THCudaHalfTensor *e3, int raiseErrors) {
  return THCudaHalfTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}
#endif // CUDA_HALF_TENSOR

template <>
int expand3(THCState *s, THCudaByteTensor *r1, THCudaByteTensor *r2, THCudaByteTensor *r3,
            THCudaByteTensor *e1, THCudaByteTensor *e2, THCudaByteTensor *e3, int raiseErrors) {
  return THCudaByteTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(THCState *s, THCudaCharTensor *r1, THCudaCharTensor *r2, THCudaCharTensor *r3,
            THCudaCharTensor *e1, THCudaCharTensor *e2, THCudaCharTensor *e3, int raiseErrors) {
  return THCudaCharTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(THCState *s, THCudaShortTensor *r1, THCudaShortTensor *r2, THCudaShortTensor *r3,
            THCudaShortTensor *e1, THCudaShortTensor *e2, THCudaShortTensor *e3, int raiseErrors) {
  return THCudaShortTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(THCState *s, THCudaIntTensor *r1, THCudaIntTensor *r2, THCudaIntTensor *r3,
            THCudaIntTensor *e1, THCudaIntTensor *e2, THCudaIntTensor *e3, int raiseErrors) {
  return THCudaIntTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}

template <>
int expand3(THCState *s, THCudaLongTensor *r1, THCudaLongTensor *r2, THCudaLongTensor *r3,
            THCudaLongTensor *e1, THCudaLongTensor *e2, THCudaLongTensor *e3, int raiseErrors) {
  return THCudaLongTensor_expand3(s, r1, r2, r3, e1, e2, e3, raiseErrors);
}
