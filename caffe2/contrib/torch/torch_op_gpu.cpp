#include "caffe2/core/context_gpu.h"
#include "torch_op.h"

extern "C" {
#include <THCStorage.h>
#include <THCTensor.h>
#include <THCStream.h>
}

namespace caffe2 {

namespace torch {

template <>
struct TyTraits<CUDAContext> {
  static const char* moduleTy;
  static const char* prelude;
  static const char* tensorTy;
  using Tensor = THCudaTensor;
};

const char* TyTraits<CUDAContext>::tensorTy = "torch.CudaTensor";
const char* TyTraits<CUDAContext>::moduleTy = "cuda";
const char* TyTraits<CUDAContext>::prelude = R"(
        require 'torch'
        require 'nn'
        require 'cunn'
)";

THCState* cudaState(Torch<CUDAContext>* t) {
  auto* L = t->L();
  lua_getglobal(L, "cutorch");
  CAFFE_ENFORCE(!lua_isnil(L, -1));
  lua_getfield(L, -1, "_state");
  CAFFE_ENFORCE(!lua_isnil(L, -1));
  THCState* state = reinterpret_cast<THCState*>(lua_touserdata(L, -1));
  lua_pop(L, 2);
  return state;
}

template <>
void Torch<CUDAContext>::setContext(CUDAContext* context) {
  THCState *state = cudaState(this);
  THCStream* stream = THCState_getStream(state);
  THCudaCheck(cudaStreamDestroy(stream->stream));
  stream->stream = context->cuda_stream();
}

template <>
void Torch<CUDAContext>::setTensor(typename Traits::Tensor* t, Blob* blob) {
  CAFFE_ENFORCE_EQ(tensorTy(*blob), Traits::tensorTy);
  auto* cs = cudaState(this);
  auto* tc = blob->template GetMutable<Tensor<CUDAContext>>();
  CAFFE_ENFORCE_EQ(THCudaTensor_nElement(cs, t), tc->size());
  THCudaStorage* storage = THCudaStorage_newWithData(
      cs, tc->template mutable_data<float>(), tc->size());
  THCudaStorage_clearFlag(cs, storage, TH_STORAGE_FREEMEM);
  THCudaStorage* original = t->storage;
  t->storage = storage;
  THCudaStorage_free(cs, original);
}

template <>
typename Torch<CUDAContext>::Traits::Tensor* Torch<CUDAContext>::blobToTensor(
    Blob* blob) {
  CAFFE_ENFORCE_EQ(tensorTy(*blob), Traits::tensorTy);
  auto* cs = cudaState(this);
  auto* tc = blob->template GetMutable<Tensor<CUDAContext>>();

  size_t size = tc->size();
  THLongStorage* thshape = THLongStorage_newWithSize(tc->ndim());
  for (int i = 0; i < tc->ndim(); ++i) {
    THLongStorage_set(thshape, i, tc->dim(i));
  }
  THCudaStorage* storage =
      THCudaStorage_newWithData(cs, tc->template mutable_data<float>(), size);
  THCudaStorage_clearFlag(cs, storage, TH_STORAGE_FREEMEM);
  auto* th = THCudaTensor_newWithStorage(cs, storage, 0, thshape, nullptr);
  THCudaStorage_free(cs, storage);
  THLongStorage_free(thshape);
  CAFFE_ENFORCE_EQ(
      THCudaTensor_storage(cs, th)->data, tc->template mutable_data<float>());
  return th;
}

template <>
std::vector<TIndex> Torch<CUDAContext>::tensorShape(
    typename Traits::Tensor* t) {
  auto* cs = cudaState(this);
  auto* size = t->size;
  return std::vector<TIndex>(size, size + THCudaTensor_nDimension(cs, t));
}

template <>
typename Torch<CUDAContext>::Traits::Tensor* Torch<CUDAContext>::newTensorAs(
    const Tensor<CUDAContext>& tc) {
  auto* cs = cudaState(this);
  THLongStorage* thshape = THLongStorage_newWithSize(tc.ndim());
  for (uint32_t i = 0; i < tc.ndim(); ++i) {
    THLongStorage_set(thshape, i, tc.dim(i));
  }
  THCudaTensor* d = THCudaTensor_newWithSize(cs, thshape, nullptr);
  THLongStorage_free(thshape);
  return d;
}
}

REGISTER_CUDA_OPERATOR(Torch, TorchOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(TorchGradient, TorchGradientOp<CUDAContext>);
}
