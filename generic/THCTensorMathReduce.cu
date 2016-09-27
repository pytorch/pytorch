#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathReduce.cu"
#else

THC_API void
THCTensor_(sum)(THCState* state, THCTensor *self, THCTensor *src, long dimension) {
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  if (!THC_reduceDim(state, self, src,
                     thrust::identity<real>(),
                     ReduceAdd<real, real>(),
                     ScalarConvert<int, real>::to(0),
                     dimension)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(prod)(THCState* state, THCTensor *self, THCTensor *src, long dimension) {
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  if (!THC_reduceDim(state, self, src,
                     thrust::identity<real>(),
                     ReduceMultiply<real, real>(),
                     ScalarConvert<int, real>::to(1),
                     dimension)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(mean)(THCState *state, THCTensor *self, THCTensor *src, long dim)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  THCTensor_(sum)(state, self, src, dim);
  THCTensor_(div)(state, self, self, ScalarConvert<long, real>::to(THCTensor_(size)(state, src, dim)));
}

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

void THCTensor_(renorm)(THCState *state, THCTensor* self, THCTensor* src, real value, long dimension, real maxnorm)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  THCTensor *self_;
  THCTensor *src_ = THCTensor_(newTranspose)(state, src, dimension, 0);
  THCTensor *data = THCTensor_(newClone)(state, src_);
  long size = THCTensor_(nElement)(state, data)/data->size[0];

  THArgCheck(dimension >= 0 && dimension < THCTensor_(nDimension)(state, src), 3, "invalid dimension");
  THArgCheck(THCNumerics<real>::gt(value, ScalarConvert<int, real>::to(0)), 2, "non-positive-norm not supported");
  THArgCheck(THCTensor_(nDimension)(state, src) > 1, 1, "need at least 2 dimensions");

  dim3 grid(data->size[0]);
  dim3 threads(32);

  THCTensor_kernel_renorm<real><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(THCTensor_(data)(state, data), value, size, maxnorm);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCTensor_(free)(state, src_);
  self_ = THCTensor_(newTranspose)(state, data, dimension, 0);
  THCTensor_(resizeAs)(state, self, self_);
  THCTensor_(freeCopyTo)(state, self_, self);
  THCTensor_(free)(state, data);
}

void THCTensor_(std)(THCState *state, THCTensor *self_, THCTensor *src, long dimension, int flag)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src));
  THLongStorage *dim = THCTensor_(newSizeOf)(state, src);
  THLongStorage_set(dim, dimension, 1);
  THCTensor_(resize)(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  src = THCTensor_(newContiguous)(state, src);

  if (dimension == THCTensor_(nDimension)(state, src) - 1) {
    THCTensor_varInnermostDim<THCTensor, real, true>(state, self, src, flag);
  } else {
    THCTensor_varOuterDim<THCTensor, real, true>(state, self, src, dimension, flag);
  }

  THCTensor_(free)(state, src);
  THCTensor_(freeCopyTo)(state, self, self_);
}

#endif

THC_API accreal
THCTensor_(sumall)(THCState *state, THCTensor *self) {
  THAssert(THCTensor_(checkGPU)(state, 1, self));
  accreal val;
  if (!THC_reduceAll(state, self,
                     thrust::identity<real>(),
                     ReduceAdd<real, accreal>(),
                     ReduceAdd<accreal, accreal>(),
                     ScalarConvert<int, accreal>::to(0),
                     &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
  return val;
}

THC_API accreal
THCTensor_(prodall)(THCState *state, THCTensor *self) {
  THAssert(THCTensor_(checkGPU)(state, 1, self));
  accreal val;
  if (!THC_reduceAll(state, self,
                     thrust::identity<real>(),
                     ReduceMultiply<real, accreal>(),
                     ReduceMultiply<accreal, accreal>(),
                     ScalarConvert<int, accreal>::to(1),
                     &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
  return val;
}

THC_API accreal
THCTensor_(meanall)(THCState *state, THCTensor *self)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self));
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THCTensor_(sumall)(state, self)/THCTensor_(nElement)(state, self);
}

THC_API real
THCTensor_(minall)(THCState *state, THCTensor *self) {
  THAssert(THCTensor_(checkGPU)(state, 1, self));
  real val;
  if (!THC_reduceAll(state, self,
                     thrust::identity<real>(),
                     ReduceMin<real>(),
                     ReduceMin<real>(),
                     THCNumerics<real>::max(), &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
  return val;
}

THC_API real
THCTensor_(maxall)(THCState *state, THCTensor *self) {
  THAssert(THCTensor_(checkGPU)(state, 1, self));
  real val;
  if (!THC_reduceAll(state, self,
                     thrust::identity<real>(),
                     ReduceMax<real>(),
                     ReduceMax<real>(),
                     THCNumerics<real>::min(), &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
  return val;
}

THC_API void
THCTensor_(max)(THCState *state,
                THCTensor *values,
                THCudaLongTensor *indices,
                THCTensor *src,
                long dimension) {
  THAssert(THCTensor_(checkGPU)(state, 3, values, indices, src));

  thrust::pair<typename TensorUtils<THCTensor>::DataType, long>
    init =
    thrust::make_pair<typename TensorUtils<THCTensor>::DataType, long>(
      THCNumerics<typename TensorUtils<THCTensor>::DataType>::min(), 1);

  return THC_reduceDimIndex(
    state, values, indices, src, dimension, init,
    MaxValuePair<typename TensorUtils<THCTensor>::DataType, long>());
}

THC_API void
THCTensor_(min)(THCState *state,
                THCTensor *values,
                THCudaLongTensor *indices,
                THCTensor *src,
                long dimension) {
  THAssert(THCTensor_(checkGPU)(state, 3, values, indices, src));

  thrust::pair<typename TensorUtils<THCTensor>::DataType, long>
    init =
    thrust::make_pair<typename TensorUtils<THCTensor>::DataType, long>(
      THCNumerics<typename TensorUtils<THCTensor>::DataType>::max(), 1);

  return THC_reduceDimIndex(
    state, values, indices, src, dimension, init,
    MinValuePair<typename TensorUtils<THCTensor>::DataType, long>());
}

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void THCTensor_(norm)(THCState *state, THCTensor* self, THCTensor* src, real value, long dimension)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self, src));
  if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(0.0))) {
    THC_reduceDim(state, self, src,
                  TensorNonZeroOp<real>(), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension);
  } else if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(1.0))) {
    THC_reduceDim(state, self, src,
                  TensorNormOp<real, 1>(value), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension);

  } else if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(2.0))) {
    THC_reduceDim(state, self, src,
                  TensorNormOp<real, 2>(value), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension);
    THCTensor_(pow)(state, self, self, ScalarConvert<float, real>::to(0.5));

  } else {
    THC_reduceDim(state, self, src,
                  TensorNormOp<real, -1>(value), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension);
    THCTensor_(pow)(state, self, self, THCNumerics<real>::cinv(value));
  }

  THCudaCheck(cudaGetLastError());
}

#endif

#endif
