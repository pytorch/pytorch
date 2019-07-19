#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPointwise.cu"
#else

#include <ATen/MemoryOverlap.h>
#ifdef BUILD_NAMEDTENSOR
#include <ATen/NamedTensorUtils.h>
#endif

void THCTensor_(cbitand)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  return THError("cbitand is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src2, TensorBitAndOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorBitAndOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(cbitor)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src2, TensorBitOrOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorBitOrOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(cbitxor)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src2, TensorBitXorOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorBitXorOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(sign)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorSignOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorSignOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(self_, src);
#endif
}

void THCTensor_(cmax)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self, src2, TensorMaxOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self, src1, src2, TensorMaxOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

void THCTensor_(cmin)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self, src2, TensorMinOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self, src1, src2, TensorMinOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

void THCTensor_(cmaxValue)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1<scalar_t>(state, self, TensorMaxValueOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src);
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self, src, TensorMaxValueOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

void THCTensor_(cminValue)(THCState *state, THCTensor *self, THCTensor *src, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1<scalar_t>(state, self, TensorMinValueOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src);
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self, src, TensorMinValueOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

#if !defined(THC_REAL_IS_BOOL)

static void propagate_names_if_named_tensor_enabled(THCTensor* result, THCTensor* src) {
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(result, src);
#endif
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)             \
  struct Tensor_##NAME##_##REAL##_Op {                                  \
    __device__ __forceinline__ void operator()(scalar_t* out, scalar_t* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(scalar_t* v) const {         \
      *v = CFUNC(*v);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCTensor_(NAME)(THCState* state, THCTensor* self_, THCTensor* src) { \
    THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));               \
    at::assert_no_internal_overlap(self_, #NAME);                       \
    if (self_ == src) {                                                 \
      if (!THC_pointwiseApply1<scalar_t>(state, self_, Tensor_##NAME##_##REAL##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    } else {                                                            \
      THCTensor_(resizeAs)(state, self_, src);                          \
                                                                        \
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, Tensor_##NAME##_##REAL##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    }                                                                   \
                                                                        \
    THCudaCheck(cudaGetLastError());                                    \
    propagate_names_if_named_tensor_enabled(self_, src);                \
  }

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC, REAL) \
  IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  log, THCNumerics<scalar_t>::log,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(lgamma, THCNumerics<scalar_t>::lgamma, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log10, THCNumerics<scalar_t>::log10, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, THCNumerics<scalar_t>::log1p, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( log2, THCNumerics<scalar_t>::log2,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  exp, THCNumerics<scalar_t>::exp,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(expm1, THCNumerics<scalar_t>::expm1, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cos, THCNumerics<scalar_t>::cos,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  sin, THCNumerics<scalar_t>::sin,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( sqrt, THCNumerics<scalar_t>::sqrt,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(rsqrt, THCNumerics<scalar_t>::rsqrt, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( ceil, THCNumerics<scalar_t>::ceil,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, THCNumerics<scalar_t>::floor, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(trunc, THCNumerics<scalar_t>::trunc, Real)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  acos, THCNumerics<scalar_t>::acos,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cosh, THCNumerics<scalar_t>::cosh,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  asin, THCNumerics<scalar_t>::asin,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  sinh, THCNumerics<scalar_t>::sinh,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(   tan, THCNumerics<scalar_t>::tan,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  atan, THCNumerics<scalar_t>::atan,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  tanh, THCNumerics<scalar_t>::tanh,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(   erf, THCNumerics<scalar_t>::erf,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  erfc, THCNumerics<scalar_t>::erfc,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(erfinv, THCNumerics<scalar_t>::erfinv,Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( round, THCNumerics<scalar_t>::round, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  frac, THCNumerics<scalar_t>::frac,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cinv, THCNumerics<scalar_t>::cinv,  Real)

#endif

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  neg, THCNumerics<scalar_t>::neg,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  abs, THCNumerics<scalar_t>::abs,   Real)

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_
#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

void THCTensor_(clamp)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t min_value,
  scalar_t max_value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorClampOp<scalar_t>(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorClampOp<scalar_t>(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(crossKernel)(THCState *state, THCTensor *self, THCTensor *x, THCTensor *y, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, x, y));

  int64_t sx = THCTensor_(stride)(state, x, dimension);
  int64_t sy = THCTensor_(stride)(state, y, dimension);
  int64_t so = THCTensor_(stride)(state, self, dimension);
  THCTensor *nx = THCTensor_(newNarrow)(state, x, dimension, 0, 1);
  THCTensor *ny = THCTensor_(newNarrow)(state, y, dimension, 0, 1);
  THCTensor *nself = THCTensor_(newNarrow)(state, self, dimension, 0, 1);
  if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, nself, nx, ny, TensorCrossOp<scalar_t>(sx, sy, so))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
  THCTensor_(free)(state, nx);
  THCTensor_(free)(state, ny);
  THCTensor_(free)(state, nself);
}

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

void THCTensor_(atan2)(THCState *state, THCTensor *self_, THCTensor *tx, THCTensor *ty)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, tx, ty));
  THArgCheck(THCTensor_(nElement)(state, tx) ==
             THCTensor_(nElement)(state, ty), 3, "sizes do not match");
  THCTensor_(resizeAs)(state, self_, tx);

  if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, tx, ty, TensorATan2Op<scalar_t>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(sigmoid)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorSigmoidOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorSigmoidOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(self_, src);
#endif
}

void THCTensor_(digamma)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ != src) {
    THCTensor_(resizeAs)(state, self_, src);
  }
  if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorDigammaOp<scalar_t, accreal>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(self_, src);
#endif
}

void THCTensor_(polygamma)(THCState* state, THCTensor* self_, int64_t n, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ != src) {
    THCTensor_(resizeAs)(state, self_, src);
  }
  switch (n) {
    case 0:
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorDigammaOp<scalar_t, accreal>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
      break;
    case 1:
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorTrigammaOp<scalar_t, accreal>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
      break;
    default:
      THError("polygamma(n,x) is not implemented for n>=2");
  }

  THCudaCheck(cudaGetLastError());
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(self_, src);
#endif
}

#endif

namespace {
c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl> retainTensorImpl(THCTensor* self) {
  c10::raw::intrusive_ptr::incref(self);
  return c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(self);
}
}

void THCTensor_(cadd)(THCState *state, THCTensor *self_, THCTensor* src1, scalar_t value, THCTensor *src2)
{
  auto out = at::Tensor(retainTensorImpl(self_));
#ifdef THC_REAL_IS_HALF
  auto alpha = at::Half(value);
#else
  auto alpha = value;
#endif
  at::add_out(out, at::Tensor(retainTensorImpl(src1)), at::Tensor(retainTensorImpl(src2)), alpha);
}

void THCTensor_(csub)(THCState *state, THCTensor *self_, THCTensor* src1, scalar_t value, THCTensor *src2)
{
  auto out = at::Tensor(retainTensorImpl(self_));
#ifdef THC_REAL_IS_HALF
  auto alpha = at::Half(value);
#else
  auto alpha = value;
#endif
  at::sub_out(out, at::Tensor(retainTensorImpl(src1)), at::Tensor(retainTensorImpl(src2)), alpha);
}

void THCTensor_(cmul)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  auto out = at::Tensor(retainTensorImpl(self_));
  at::mul_out(out, at::Tensor(retainTensorImpl(src1)), at::Tensor(retainTensorImpl(src2)));
}

void THCTensor_(cpow)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src2, TensorCPowOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = pow(src1, src2)
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorCPowOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(pow)(THCState *state, THCTensor *self_, THCTensor *src, scalar_t value) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
      if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorPowOp<scalar_t, 1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
      if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorPowOp<scalar_t, 2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
      if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorPowOp<scalar_t, 3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-1))) {
      if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorPowOp<scalar_t, -1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-2))) {
      if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorPowOp<scalar_t, -2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
#endif
    } else {
      // fallback implementation using pow
      if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorPowOp<scalar_t, -3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(1))) {
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorPowOp<scalar_t, 1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(2))) {
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorPowOp<scalar_t, 2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(3))) {
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorPowOp<scalar_t, 3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-1))) {
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorPowOp<scalar_t, -1>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else if (THCNumerics<scalar_t>::eq(value, ScalarConvert<int, scalar_t>::to(-2))) {
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorPowOp<scalar_t, -2>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
#endif
    } else {
      // fallback implementation using pow
      if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorPowOp<scalar_t, -3>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(tpow)(THCState *state, THCTensor *self_, scalar_t value, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorTPowOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src, TensorTPowOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}
void THCTensor_(cdiv)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  auto out = at::Tensor(retainTensorImpl(self_));
  at::div_out(out, at::Tensor(retainTensorImpl(src1)), at::Tensor(retainTensorImpl(src2)));
}

void THCTensor_(clshift)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF)
  return THError("clshift not supported for torch.CudaHalfTensor");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src2, TensorLShiftOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorLShiftOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(crshift)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF)
  return THError("crshift not supported for torch.CudaHalfTensor");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src2, TensorRShiftOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorRShiftOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

void THCTensor_(cremainder)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self, src2, TensorCRemainderOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self, src1, src2, TensorCRemainderOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

void THCTensor_(cfmod)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self, src2, TensorCFmodOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self, src1, src2, TensorCFmodOp<scalar_t>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

void THCTensor_(addcmul)(THCState *state, THCTensor *self_, THCTensor *t, scalar_t value, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THCTensor_(resizeAs)(state, self_, t);
    THCTensor_(copy)(state, self_, t);
  }
  else
  {
    THArgCheck(THCTensor_(nElement)(state, self_) == THCTensor_(nElement)(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THCTensor_(nElement)(state, src1) == THCTensor_(nElement)(state, src2),
             3, "sizes do not match");

  if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorAddCMulOp<scalar_t>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(addcdiv)(THCState *state, THCTensor *self_, THCTensor *t, scalar_t value, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THCTensor_(resizeAs)(state, self_, t);
    THCTensor_(copy)(state, self_, t);
  }
  else
  {
    THArgCheck(THCTensor_(nElement)(state, self_) == THCTensor_(nElement)(state, src1),
               1, "sizes do not match");
  }
  THArgCheck(THCTensor_(nElement)(state, src1) == THCTensor_(nElement)(state, src2),
             3, "sizes do not match");

  if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, self_, src1, src2, TensorAddCDivOp<scalar_t>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif
#endif
