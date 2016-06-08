#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(float* out, float* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(float* v) const {        \
      *v = CFUNC(*v);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCudaTensor_##NAME(THCState* state, THCudaTensor* self_, THCudaTensor* src) { \
    THAssert(THCudaTensor_checkGPU(state, 2, self_, src));                \
    if (self_ == src) {                                                 \
      if (!THC_pointwiseApply1(state, self_, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); \
      }                                                                 \
    } else {                                                            \
      THCudaTensor_resizeAs(state, self_, src);                         \
                                                                        \
      if (!THC_pointwiseApply2(state, self_, src, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); \
      }                                                                 \
    }                                                                   \
                                                                        \
    THCudaCheck(cudaGetLastError());                                    \
  }

__device__ __forceinline__ double frac(double x)
{
  return x - trunc(x);
}

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(rsqrt, rsqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(trunc, trunc)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(frac, frac)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(neg, -)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cinv, 1.0f / )

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

struct TensorSigmoidOp {
  __device__ __forceinline__ void operator()(float* out, float* in) const {
    *out = 1.0f / (1.0f + expf(- *in));
  }

  __device__ __forceinline__ void operator()(float* v) const {
    *v = 1.0f / (1.0f + expf(- *v));
  }
};

void THCudaTensor_sigmoid(THCState* state, THCudaTensor* self_, THCudaTensor* src) {
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorSigmoidOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorSigmoidOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMaxOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = max(*in1, *in2);
  }
};

void THCudaTensor_cmax(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMaxOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMaxOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMinOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = min(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = min(*in1, *in2);
  }
};

void THCudaTensor_cmin(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMinOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMinOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMaxValueOp {
  TensorMaxValueOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out) {
    *out = max(*out, val);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(*in, val);
  }

  float val;
};

void THCudaTensor_cmaxValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMaxValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMaxValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMinValueOp {
  TensorMinValueOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out) {
    *out = min(*out, val);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = min(*in, val);
  }

  float val;
};

void THCudaTensor_cminValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMinValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMinValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

template <typename T>
struct TensorAddOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out += *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 + *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorAddOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*out, *in);
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout += fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in1, *in2);
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 + fin2;
    *out = __float2half(fout);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out += val * *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 + val * *in2;
  }

  T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCAddOp<half> {
  TensorCAddOp(half v) : val(v) {}

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*out, __hmul(val, *in));
#else
    float fout = __half2float(*out);
    float fval = __half2float(val);
    float fin = __half2float(*in);

    fout += fval * fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in1, __hmul(val, *in2));
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fval = __half2float(val);

    float fout = fin1 + fval * fin2;
    *out = __float2half(fout);
#endif
  }

  half val;
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorSubOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out -= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 - *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSubOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*out, *in);
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout -= fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in1, *in2);
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 - fin2;
    *out = __float2half(fout);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorMulOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 * *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorMulOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*out, *in);
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout *= fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in1, *in2);
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 * fin2;
    *out = __float2half(fout);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCPowOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = powf((float) *out, (float) *in);
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = powf((float) *in1, (float) *in2);
  }
};

template <>
struct TensorCPowOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = pow(*out, *in);
  }

  __device__ __forceinline__ void operator()(double* out, double* in1, double* in2) {
    *out = pow(*in1, *in2);
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCPowOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    // No fp16 pow function yet
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout = powf(fout, fin);
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    // No fp16 pow function yet
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = powf(fin1, fin2);
    *out = __float2half(fout);
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorDivOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out /= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 / *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorDivOp<half> {
  __device__ __forceinline__ void
  operator()(half* out, half* in) {
    // No fp16 div instruction yet
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout /= fin;
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void
  operator()(half* out, half* in1, half* in2) {
    // No fp16 div instruction yet
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 / fin2;
    *out = __float2half(fout);
  }
};
#endif // CUDA_HALF_TENSOR

#include "generic/THCTensorMathPointwise.cu"
#include "THCGenerateAllTypes.h"
