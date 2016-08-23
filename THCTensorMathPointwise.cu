#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"

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
struct TensorSigmoidOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    T one = (T) 1.0;
    *out = one / (one + THCNumerics<T>::exp(- *in));
  }

  __device__ __forceinline__ void operator()(T* v) const {
    T one = (T) 1.0;
    *v = one / (one + THCNumerics<T>::exp(- *v));
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSigmoidOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    half one = ScalarConvert<int, half>::to(1);
    *out = hdiv(one, __hadd(one, hexp(__hneg(*in))));
#else
    float fin = __half2float(*in);
    *out = __float2half(1.0f / (1.0f + expf(- fin)));
#endif
  }

  __device__ __forceinline__ void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    half one = ScalarConvert<int, half>::to(1);
    *v = hdiv(one, __hadd(one, hexp(__hneg(*v))));
#else
    float fv = __half2float(*v);
    *v = __float2half(1.0f / (1.0f + expf(- fv)));
#endif
  }
};
#endif

template <typename T>
struct TensorSignOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T orig = *in;
    *out = (orig > 0) - (orig < 0);
  }

  __device__ __forceinline__ void operator()(T* v) {
    T orig = *v;
    *v = (orig > 0) - (orig < 0);
  }
};

template <>
struct TensorSignOp<unsigned char> {
  __device__ __forceinline__ void operator()(unsigned char* out, unsigned char* in) {
    *out = 1;
  }

  __device__ __forceinline__ void operator()(unsigned char* v) {
    *v = 1;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSignOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    half zero = ScalarConvert<int, half>::to(0);
    half orig = *in;
    *out = __float2half((float) __hgt(orig, zero) - (float) __hlt(orig, zero));
#else
    float orig = __half2float(*in);
    *out = __float2half((orig > 0) - (orig < 0));
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    half zero = ScalarConvert<int, half>::to(0);
    half orig = *v;
    *v = __float2half((float) __hgt(orig, zero) -  (float) __hlt(orig, zero));
#else
    float orig = __half2float(*v);
    *v = __float2half((orig > 0) - (orig < 0));
#endif
  }
};
#endif

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

__device__ __forceinline__ double frac(double x)
{
  return x - trunc(x);
}

#include "generic/THCTensorMathPointwise.cu"
#include "THCGenerateAllTypes.h"
