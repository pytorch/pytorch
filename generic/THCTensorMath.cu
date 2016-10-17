#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMath.cu"
#else

THC_API void
THCTensor_(fill)(THCState* state, THCTensor *self_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));

  if (!THC_pointwiseApply1(
        state, self_, TensorFillOp<real>(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(zero)(THCState *state, THCTensor *self_)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  if (THCTensor_(isContiguous)(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCTensor_(data)(state, self_),
                                0,
                                sizeof(real) * THCTensor_(nElement)(state, self_),
                                THCState_getCurrentStream(state)));
  } else {
    if (!THC_pointwiseApply1(
          state, self_,
          TensorFillOp<real>(ScalarConvert<int, real>::to(0)))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(zeros)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(zero)(state, r_);
}

THC_API void
THCTensor_(ones)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(fill)(state, r_, ScalarConvert<int, real>::to(1));
}

THC_API void
THCTensor_(reshape)(THCState *state, THCTensor *r_, THCTensor *t, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 2, r_, t));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(copy)(state, r_, t);
}

ptrdiff_t
THCTensor_(numel)(THCState *state, THCTensor *t)
{
  return THCTensor_(nElement)(state, t);
}

void THCTensor_(cat)(THCState *state, THCTensor *result,
		     THCTensor *ta, THCTensor *tb, int dimension)
{
  THCTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THCTensor_(catArray)(state, result, inputs, 2, dimension);
}

void THCTensor_(catArray)(THCState *state, THCTensor *result,
			  THCTensor **inputs, int numInputs, int dimension)
{
  THLongStorage *size;
  int i, j;
  long offset;
  int ndim = dimension + 1;
  for (i = 0; i < numInputs; i++)
  {
    ndim = THMax(ndim, THCTensor_(nDimension)(state, inputs[i]));
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(dimension >= 0, 4, "invalid dimension %d", dimension+1);

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    long dimSize = i < THCTensor_(nDimension)(state, inputs[0])
                       ? THCTensor_(size)(state, inputs[0], i)
                       : 1;
    if (i == dimension)
    {
      for (j = 1; j < numInputs; j++)
      {
        dimSize += i < THCTensor_(nDimension)(state, inputs[j])
                       ? THCTensor_(size)(state, inputs[j], i)
                       : 1;
      }
    }
    else
    {
      for (j = 1; j < numInputs; j++)
      {
        if (dimSize != (i < THCTensor_(nDimension)(state, inputs[j])
			? THCTensor_(size)(state, inputs[j], i)
			: 1)) {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        }
      }
    }
    size->data[i] = dimSize;
  }

  THCTensor_(resize)(state, result, size, NULL);
  THLongStorage_free(size);

  offset = 0;
  for (j = 0; j < numInputs; j++)
  {
    long dimSize = dimension < THCTensor_(nDimension)(state, inputs[j])
			       ? THCTensor_(size)(state, inputs[j], dimension)
			       : 1;
    THCTensor *nt = THCTensor_(newWithTensor)(state, result);
    THCTensor_(narrow)(state, nt, NULL, dimension, offset, dimSize);
    THCTensor_(copy)(state, nt, inputs[j]);
    THCTensor_(free)(state, nt);
    offset += dimSize;
  }
}

#endif
