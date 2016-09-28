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

#endif
