#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathPointwise.cu"
#else

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)             \
  struct Tensor_##NAME##_##REAL##_Op {                                  \
    __device__ __forceinline__ void operator()(real* out, real* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(real* v) const {         \
      *v = CFUNC(*v);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCTensor_(NAME)(THCState* state, THCTensor* self_, THCTensor* src) { \
    THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));               \
    if (self_ == src) {                                                 \
      if (!THC_pointwiseApply1(state, self_, Tensor_##NAME##_##REAL##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    } else {                                                            \
      THCTensor_(resizeAs)(state, self_, src);                          \
                                                                        \
      if (!THC_pointwiseApply2(state, self_, src, Tensor_##NAME##_##REAL##_Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);                      \
      }                                                                 \
    }                                                                   \
                                                                        \
    THCudaCheck(cudaGetLastError());                                    \
  }

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC, REAL) \
  IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_(NAME, CFUNC, REAL)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  log, THCNumerics<real>::log,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(lgamma, THCNumerics<real>::lgamma, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, THCNumerics<real>::log1p, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  exp, THCNumerics<real>::exp,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  cos, THCNumerics<real>::cos,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  sin, THCNumerics<real>::sin,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( sqrt, THCNumerics<real>::sqrt,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(rsqrt, THCNumerics<real>::rsqrt, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( ceil, THCNumerics<real>::ceil,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, THCNumerics<real>::floor, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(trunc, THCNumerics<real>::trunc, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  neg, THCNumerics<real>::neg,   Real)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( acos, THCNumerics<real>::acos,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( cosh, THCNumerics<real>::cosh,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( asin, THCNumerics<real>::asin,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( sinh, THCNumerics<real>::sinh,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  tan, THCNumerics<real>::tan,   Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( atan, THCNumerics<real>::atan,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( tanh, THCNumerics<real>::tanh,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, THCNumerics<real>::round, Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( frac, THCNumerics<real>::frac,  Real)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC( cinv, THCNumerics<real>::cinv,  Real)

#endif

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(  abs, THCNumerics<real>::abs,   Real)

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_
#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

void THCTensor_(sign)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorSignOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorSignOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(clamp)(THCState *state, THCTensor *self_, THCTensor *src, real min_value,
  real max_value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorClampOp<real>(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorClampOp<real>(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cross)(THCState *state, THCTensor *self, THCTensor *x, THCTensor *y, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, x, y));

  int i;
  long nd = THCTensor_(nDimension)(state, x);
  ptrdiff_t nelem = THCTensor_(nElement)(state, x);
  THArgCheck(nd == THCTensor_(nDimension)(state, y), 1, "tensors must have same number of dimensions");
  for (i = 0; i < nd; i++) {
    THArgCheck(THCTensor_(size)(state, x, i) == THCTensor_(size)(state, y, i), 1, "dimension %i of x and y does not match", i);
    if (dimension < 0 && THCTensor_(size)(state, x, i) == 3) {
      dimension = i;
    }
  }

  THArgCheck(dimension >= 0 && dimension < nd, 3, "dimension %d out of range", dimension+1);
  THArgCheck(THCTensor_(size)(state, x, dimension) == 3, 3,
      "dimension %d does not have size 3", dimension+1);
  THCTensor_(resizeAs)(state, self, x);

  long sx = THCTensor_(stride)(state, x, dimension);
  long sy = THCTensor_(stride)(state, y, dimension);
  long so = THCTensor_(stride)(state, self, dimension);
  THCTensor *nx = THCTensor_(newNarrow)(state, x, dimension, 0, 1);
  THCTensor *ny = THCTensor_(newNarrow)(state, y, dimension, 0, 1);
  THCTensor *nself = THCTensor_(newNarrow)(state, self, dimension, 0, 1);
  if (!THC_pointwiseApply3(state, nself, nx, ny, TensorCrossOp<real>(sx, sy, so))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
  THCTensor_(free)(state, nx);
  THCTensor_(free)(state, ny);
  THCTensor_(free)(state, nself);
}


#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

void THCTensor_(sigmoid)(THCState* state, THCTensor* self_, THCTensor* src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorSigmoidOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorSigmoidOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(pow)(THCState *state, THCTensor *self_, THCTensor *src, real value) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorPowOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorPowOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(tpow)(THCState *state, THCTensor *self_, real value, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorTPowOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorTPowOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(lerp)(THCState *state, THCTensor *result, THCTensor *a, THCTensor *b, real w)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, result, a, b));
  THArgCheck(THCTensor_(nElement)(state, a) ==
             THCTensor_(nElement)(state, b), 3, "sizes do not match");
  THCTensor_(resizeAs)(state, result, a);

  if (!THC_pointwiseApply3(state, result, a, b, TensorLerpOp<real>(w))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif

THC_API void
THCTensor_(cadd)(THCState *state, THCTensor *self_, THCTensor* src1, real value, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == ScalarConvert<int, real>::to(1)) {
      // self += src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorAddOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorCAddOp<real>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    if (value == ScalarConvert<int, real>::to(1)) {
      // self = src1 + src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorCAddOp<real>(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(csub)(THCState *state, THCTensor *self_, THCTensor* src1, real value, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == ScalarConvert<int, real>::to(1)) {
      // self -= src2
      if (!THC_pointwiseApply2(state, self_, src2, TensorSubOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += -value * src2
      if (!THC_pointwiseApply2(state, self_, src2,
                                   TensorCAddOp<real>(
                                     ScalarNegate<real>::to(value)))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    if (value == ScalarConvert<int, real>::to(1)) {
      // self = src1 - src2
      if (!THC_pointwiseApply3(state, self_, src1, src2, TensorSubOp<real>())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 - value * src2
      if (!THC_pointwiseApply3(state, self_, src1, src2,
                                   TensorCAddOp<real>(
                                     ScalarNegate<real>::to(value)))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cmul)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorMulOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 * src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorMulOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cpow)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THC_pointwiseApply2(state, self_, src2, TensorCPowOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = pow(src1, src2)
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorCPowOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cdiv)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorDivOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorDivOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(clshift)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF)
  return THError("clshift not supported for torch.CudaHalfTensor");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorLShiftOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorLShiftOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(crshift)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF)
  return THError("crshift not supported for torch.CudaHalfTensor");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorRShiftOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorRShiftOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(cmax)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMaxOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMaxOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cmin)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorMinOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorMinOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cremainder)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorCRemainderOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorCRemainderOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cfmod)(THCState *state, THCTensor *self, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THC_pointwiseApply2(state, self, src2, TensorCFmodOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src1);
    if (!THC_pointwiseApply3(state, self, src1, src2, TensorCFmodOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cmaxValue)(THCState *state, THCTensor *self, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMaxValueOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMaxValueOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(cminValue)(THCState *state, THCTensor *self, THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));

  if (self == src) {
    if (!THC_pointwiseApply1(state, self, TensorMinValueOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self, src);
    if (!THC_pointwiseApply2(state, self, src, TensorMinValueOp<real>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

THC_API void
THCTensor_(addcmul)(THCState *state, THCTensor *self_, THCTensor *t, real value, THCTensor *src1, THCTensor *src2)
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

  if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp<real>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(addcdiv)(THCState *state, THCTensor *self_, THCTensor *t, real value, THCTensor *src1, THCTensor *src2)
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

  if (!THC_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp<real>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(cbitand)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  return THError("cbitand is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorBitAndOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorBitAndOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(cbitor)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorBitOrOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorBitOrOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}

THC_API void
THCTensor_(cbitxor)(THCState* state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
#if defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
  return THError("cbitor is only supported for integer type tensors");
#else
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self /= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorBitXorOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 / src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorBitXorOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
#endif
}
#endif
