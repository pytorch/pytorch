#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorRandom.cu"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)
THC_API void THCTensor_(uniform)(THCState* state, THCTensor *self_, double a, double b)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_uniform<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, a, b);

  THCTensor_(freeCopyTo)(state, self, self_);
};

GENERATE_KERNEL2(generate_normal, real, double mean, double stdv, float, curand_normal, (ScalarConvert<float, real>::to((x * stdv) + mean)))

THC_API void THCTensor_(normal)(THCState* state, THCTensor *self_, double mean, double stdv)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_normal<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, mean, stdv);

  THCTensor_(freeCopyTo)(state, self, self_);
};

GENERATE_KERNEL1(generate_exponential, real, double lambda, float, curand_uniform, (ScalarConvert<float, real>::to((float)(-1. / lambda * log(1-x)))))

THC_API void THCTensor_(exponential)(THCState* state, THCTensor *self_, double lambda)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_exponential<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, lambda);

  THCTensor_(freeCopyTo)(state, self, self_);
};

#undef NUM_BLOCKS

THC_API void THCTensor_(rand)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(uniform)(state, r_, 0, 1);
}

#endif

GENERATE_KERNEL1(generate_bernoulli, real, double p, float, curand_uniform, (ScalarConvert<bool, real>::to(x <= p)))

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)
THC_API void THCTensor_(bernoulli)(THCState* state, THCTensor *self_, double p)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_bernoulli<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, p);

  THCTensor_(freeCopyTo)(state, self, self_);
};

#undef NUM_BLOCKS

#endif
