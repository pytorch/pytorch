#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathReduce.cu"
#else

THC_API void
THCTensor_(sum)(THCState* state, THCTensor *self, THCTensor *src, long dimension, int keepdim) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  if (!THC_reduceDim(state, self, src,
                     thrust::identity<real>(),
                     ReduceAdd<real, real>(),
                     ScalarConvert<int, real>::to(0),
                     dimension,
                     keepdim)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(prod)(THCState* state, THCTensor *self, THCTensor *src, long dimension, int keepdim) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  if (!THC_reduceDim(state, self, src,
                     thrust::identity<real>(),
                     ReduceMultiply<real, real>(),
                     ScalarConvert<int, real>::to(1),
                     dimension,
                     keepdim)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(mean)(THCState *state, THCTensor *self, THCTensor *src, long dim, int keepdim)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  THCTensor_(sum)(state, self, src, dim, keepdim);
  THCTensor_(div)(state, self, self, ScalarConvert<long, real>::to(THCTensor_(size)(state, src, dim)));
}

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

THC_API void
THCTensor_(renorm)(THCState *state, THCTensor* self, THCTensor* src, real value, long dimension, real maxnorm)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  THCTensor *self_;
  THCTensor *src_ = THCTensor_(newTranspose)(state, src, dimension, 0);
  THCTensor *data = THCTensor_(newClone)(state, src_);
  ptrdiff_t size = THCTensor_(nElement)(state, data)/data->size[0];

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

THC_API void
THCTensor_(std)(THCState *state, THCTensor *self_, THCTensor *src, long dimension, int flag, int keepdim)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
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

  if (!keepdim) {
    THCTensor_(squeeze1d)(state, self_, self_, dimension);
  }
}

THC_API void
THCTensor_(var)(THCState *state, THCTensor *self_, THCTensor *src, long dimension, int flag, int keepdim)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src));
  THLongStorage *dim = THCTensor_(newSizeOf)(state, src);
  THLongStorage_set(dim, dimension, 1);
  THCTensor_(resize)(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  src = THCTensor_(newContiguous)(state, src);

  if (dimension == THCTensor_(nDimension)(state, src) - 1) {
    THCTensor_varInnermostDim<THCTensor, real, false>(state, self, src, flag);
  } else {
    THCTensor_varOuterDim<THCTensor, real, false>(state, self, src, dimension, flag);
  }

  THCTensor_(free)(state, src);
  THCTensor_(freeCopyTo)(state, self, self_);

  if (!keepdim) {
    THCTensor_(squeeze1d)(state, self_, self_, dimension);
  }
}

THC_API accreal
THCTensor_(stdall)(THCState *state, THCTensor *self)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
  return THCNumerics<accreal>::sqrt((THCTensor_(varall)(state, self)));
}

THC_API accreal
THCTensor_(varall)(THCState *state, THCTensor *self)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
  accreal mean = THCTensor_(meanall)(state, self);

  accreal val;
  if (!THC_reduceAll(state, self,
                     SquareFunctor<accreal, real>(mean),
                     ReduceAdd<accreal, accreal>(),
                     ReduceAdd<accreal, accreal>(),
                     ScalarConvert<int, accreal>::to(0),
                     &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  val = THCNumerics<accreal>::div(
    val,
    ScalarConvert<ptrdiff_t, accreal>::to(THCTensor_(nElement)(state, self) - 1)
  );

  THCudaCheck(cudaGetLastError());
  return val;
}

THC_API void
THCTensor_(norm)(THCState *state, THCTensor* self, THCTensor* src, real value, long dimension, int keepdim)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(0.0))) {
    THC_reduceDim(state, self, src,
                  TensorNonZeroOp<real>(), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension, keepdim);
  } else if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(1.0))) {
    THC_reduceDim(state, self, src,
                  TensorNormOp<real, 1>(value), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension, keepdim);

  } else if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(2.0))) {
    THC_reduceDim(state, self, src,
                  TensorNormOp<real, 2>(value), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension, keepdim);
    THCTensor_(pow)(state, self, self, ScalarConvert<float, real>::to(0.5));

  } else {
    THC_reduceDim(state, self, src,
                  TensorNormOp<real, -1>(value), ReduceAdd<real, real>(),
                  ScalarConvert<float, real>::to(0.0), dimension, keepdim);
    THCTensor_(pow)(state, self, self, THCNumerics<real>::cinv(value));
  }

  THCudaCheck(cudaGetLastError());
}

THC_API accreal
THCTensor_(normall)(THCState *state, THCTensor *self, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
  accreal result;

  if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(0.0))) {
    THC_reduceAll(state, self,
                  TensorNonZeroOp<real>(),
                  ReduceAdd<real, accreal>(),
                  ReduceAdd<accreal, accreal>(),
                  ScalarConvert<float, accreal>::to(0.0f),
                  &result, 0);
  } else if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(1.0))) {
    THC_reduceAll(state, self,
                  TensorNormOp<real, 1>(value),
                  ReduceAdd<real, accreal>(),
                  ReduceAdd<accreal, accreal>(),
                  ScalarConvert<float, accreal>::to(0.0f),
                  &result, 0);
  } else if (THCNumerics<real>::eq(value, ScalarConvert<float, real>::to(2.0))) {
    THC_reduceAll(state, self,
                  TensorNormOp<real, 2>(value),
                  ReduceAdd<real, accreal>(),
                  ReduceAdd<accreal, accreal>(),
                  ScalarConvert<float, accreal>::to(0.0f),
                  &result, 0);
    result = THCNumerics<accreal>::sqrt(result);
  } else {
    THC_reduceAll(state, self,
                  TensorNormOp<real, -1>(value),
                  ReduceAdd<real, accreal>(),
                  ReduceAdd<accreal, accreal>(),
                  ScalarConvert<float, accreal>::to(0.0f),
                  &result, 0);
    result = THCNumerics<accreal>::pow(
      result,
      ScalarConvert<real, accreal>::to(THCNumerics<real>::cinv(value))
    );
  }

  THCudaCheck(cudaGetLastError());
  return result;
}

accreal THCTensor_(dist)(THCState *state, THCTensor *self,
                         THCTensor *src, real value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, src));
  self = THCTensor_(newContiguous)(state, self);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  src = THCTensor_(newContiguous)(state, src);
  thrust::device_ptr<real> self_data(THCTensor_(data)(state, self));
  thrust::device_ptr<real> src_data(THCTensor_(data)(state, src));

  THCThrustAllocator thrustAlloc(state);
  accreal result = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+size, src_data, ScalarConvert<int, accreal>::to(0),
    thrust::plus<accreal>(),
    TensorDistOp<accreal, real>(ScalarConvert<real, accreal>::to(value)));

  THCTensor_(free)(state, src);
  THCTensor_(free)(state, self);

  return THCNumerics<accreal>::pow(result, 1.0 / ScalarConvert<real, accreal>::to(value));
}

#endif

THC_API accreal
THCTensor_(sumall)(THCState *state, THCTensor *self) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THCTensor_(sumall)(state, self)/THCTensor_(nElement)(state, self);
}

THC_API real
THCTensor_(minall)(THCState *state, THCTensor *self) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
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
                long dimension,
                int keepdim) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, values, indices, src));

  thrust::pair<typename TensorUtils<THCTensor>::DataType, long>
    init =
    thrust::make_pair<typename TensorUtils<THCTensor>::DataType, long>(
      THCNumerics<typename TensorUtils<THCTensor>::DataType>::min(), 1);

  return THC_reduceDimIndex(
    state, values, indices, src, dimension, keepdim, init,
    MaxValuePair<typename TensorUtils<THCTensor>::DataType, long>());
}

THC_API void
THCTensor_(min)(THCState *state,
                THCTensor *values,
                THCudaLongTensor *indices,
                THCTensor *src,
                long dimension,
                int keepdim) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, values, indices, src));

  thrust::pair<typename TensorUtils<THCTensor>::DataType, long>
    init =
    thrust::make_pair<typename TensorUtils<THCTensor>::DataType, long>(
      THCNumerics<typename TensorUtils<THCTensor>::DataType>::max(), 1);

  return THC_reduceDimIndex(
    state, values, indices, src, dimension, keepdim, init,
    MinValuePair<typename TensorUtils<THCTensor>::DataType, long>());
}

#endif
