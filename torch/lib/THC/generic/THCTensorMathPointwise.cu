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
    THAssert(THCTensor_(checkGPU)(state, 2, self_, src));               \
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
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src));
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

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

void THCTensor_(sigmoid)(THCState* state, THCTensor* self_, THCTensor* src) {
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src));
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

#endif

THC_API void
THCTensor_(cadd)(THCState *state, THCTensor *self_, THCTensor* src1, real value, THCTensor *src2)
{
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
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
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
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
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
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
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
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
  THAssert(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THArgCheck(THCTensor_(nElement)(state, src1) ==
             THCTensor_(nElement)(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THC_pointwiseApply2(state, self_, src2, TensorDivOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src1);

    // self = src1 * src2
    if (!THC_pointwiseApply3(state, self_, src1, src2, TensorDivOp<real>())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif
