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

// Copy the kth diagonal of a matrix B to a vector A.
__global__ void THCudaTensor_copyFromDiagonal(float* a, float* b, long start, long size, long strideSum, long strideA) {
  for (long linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const long bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
__global__ void THCudaTensor_copyToDiagonal(float* a, float* b, long start, long size, long strideSum, long strideB) {
  for (long linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const long aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

void THCudaTensor_diag(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k){
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  int nDimension = THCudaTensor_nDimension(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    long stride0 = THCudaTensor_stride(state, src_, 0);
    long stride1 = THCudaTensor_stride(state, src_, 1);
    long size0 = THCudaTensor_size(state, src_, 0);
    long size1 = THCudaTensor_size(state, src_, 1);
    long size = (k > 0) ? min((long long)size0, (long long)size1 - k) : min((long long)size0 + k, (long long)size1);
    THCudaTensor_resize1d(state, self_, size);
    long strideSelf = THCudaTensor_stride(state, self_, 0);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (long)threads.x)));
    long start = (k >= 0 ? k * stride1 : -k * stride0);
    THCudaTensor_copyFromDiagonal<<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCudaTensor_data(state, self_), THCudaTensor_data(state, src_), start, size, stride0 + stride1, strideSelf);
  } else {
    long totalElements = THCudaTensor_nElement(state, src_);
    long size = (k > 0) ? totalElements + k : totalElements - k;
    long strideSrc = THCudaTensor_stride(state, src_, 0);
    THCudaTensor_resize2d(state, self_, size, size);
    THCudaTensor_zero(state, self_);
    long stride0 = THCudaTensor_stride(state, self_, 0);
    long stride1 = THCudaTensor_stride(state, self_, 1);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (long)threads.x)));
    long start = (k >= 0 ? k * stride1 : -k * stride0);
    THCudaTensor_copyToDiagonal<<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCudaTensor_data(state, self_), THCudaTensor_data(state, src_), start, totalElements, stride0 + stride1, strideSrc);
  }
  THCudaCheck(cudaGetLastError());
}

float THCudaTensor_trace(THCState *state, THCudaTensor *src_) {
  THAssert(THCudaTensor_checkGPU(state, 1, src_));
  THArgCheck((src_->nDimension == 2), 1, "expected a matrix");
  THCudaTensor *diag = THCudaTensor_new(state);
  THCudaTensor_diag(state, diag, src_, 0);
  float trace = THCudaTensor_sumall(state, diag);
  THCudaTensor_free(state, diag);
  return trace;
}
