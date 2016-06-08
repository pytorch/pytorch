#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

template <typename T>
struct TensorAddConstantOp {
  TensorAddConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in + val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v += val;
  }

  const T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorAddConstantOp<half> {
  TensorAddConstantOp(half v) : val(v) {}
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in, val);
#else
    float fin = __half2float(*in);
    float fval = __half2float(val);
    float fout = fin + fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hadd(*v, val);
#else
    float fv = __half2float(*v);
    float fval = __half2float(val);
    fv += fval;
    *v = __float2half(fv);
#endif
  }

  const half val;
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorMulConstantOp {
  TensorMulConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v *= val;
  }

  const T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorMulConstantOp<half> {
  TensorMulConstantOp(half v) : val(v) {}
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in, val);
#else
    float fin = __half2float(*in);
    float fval = __half2float(val);
    float fout = fin * fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hmul(*v, val);
#else
    float fv = __half2float(*v);
    float fval = __half2float(val);
    fv *= fval;
    *v = __float2half(fv);
#endif
  }

  const half val;
};
#endif // CUDA_HALF_TENSOR

template <int Upper>
struct TensorTriOp {
  TensorTriOp(float *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(float *in) {
    ptrdiff_t n = in - start;
    long row, col;
    if (stride0 > stride1)
    {
      row = (long) (n / stride0);
      col = (long) ((n % stride0) / stride1);
    }
    else
    {
      row = (long) ((n % stride1) / stride0);
      col = (long) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = mask(in) ? *in : 0;
  }

  __device__ __forceinline__ void operator()(float* v) {
    if (!mask(v))
      *v = 0;
  }

  const float *start;
  const long stride0, stride1, k;
};

void THCudaTensor_tril(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_triu(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(cudaGetLastError());
}

#include "generic/THCTensorMathPairwise.cu"
#include "THCGenerateAllTypes.h"
