#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorRandom.cu"
#else

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

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

THC_API void THCTensor_(logNormal)(THCState* state, THCTensor *self_, double mean, double stdv)
{

  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generateLogNormal<real><<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
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

GENERATE_KERNEL2(generate_cauchy, real, double median, double sigma, float, curand_uniform, (ScalarConvert<float, real>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))

THC_API void THCTensor_(cauchy)(THCState* state, THCTensor *self_, double median, double sigma)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_cauchy<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, median, sigma);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(renormRows)(struct THCState* state,
                             THCTensor* t) {
  THAssert(THCTensor_(nDimension)(state, t) == 2);
  long rows = THCTensor_(size)(state, t, 0);
  long cols = THCTensor_(size)(state, t, 1);

  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);

  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

  renormRowsL1<real>
    <<<grid, block, block.x * sizeof(float),
    THCState_getCurrentStream(state)>>>(THCTensor_(data)(state, t),
                                        rows, cols);
}

THC_API void THCTensor_(multinomial)(struct THCState *state,
                                      THCTensor *self,
                                      THCTensor *prob_dist,
                                      int n_sample,
                                      int with_replacement)
{
  THAssert(THCTensor_(checkGPU)(state, 2, self, prob_dist));
  Generator* gen = THCRandom_getGenerator(state);

  int inputSize = THCTensor_(nDimension)(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");

  // Categories are in the innermost dimension
  long numDist =
    inputSize == 1 ? 1 : THCTensor_(size)(state, prob_dist, 0);
  long numCategoriesLong =
    inputSize == 1 ? THCTensor_(size)(state, prob_dist, 0) :
    THCTensor_(size)(state, prob_dist, 1);

  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  THArgCheck(numCategoriesLong <= FLOAT32_MAX_CONSECUTIVE_INT, 2,
             "number of categories cannot exceed 2^24");
  int numCategories = (int) numCategoriesLong;

  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");

  if (!with_replacement) {
    THArgCheck(n_sample <= numCategories, 2,
               "cannot sample n_sample > prob_dist:size(1) samples without "
               "replacement");
  }

  // It is possible that prob_dist is non-contiguous
  THCTensor* probDistContig =
    THCTensor_(newContiguous)(state, prob_dist);

  // Restructure data for 2d
  if (inputSize == 1) {
    THCTensor_(resize2d)(state, probDistContig, 1, numCategories);
  }

  THCTensor_(resize2d)(state, self, numDist, n_sample);

  if (n_sample == 1) {
    // Optimized allocation-free implementation

    // To exploit greater parallelism for the sampling, generate the
    // Uniform random samples in a separate kernel launch, into the
    // result memory. The device RNG is thread-limited
    THCTensor_(uniform)(state, self, 0.0, 1.0);

    cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
    THAssert(props != NULL);

    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;

    dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
    dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

    sampleMultinomialOnce
      <<<grid, block, block.x * sizeof(float),
         THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, self),
      numDist,
      numCategories,
      THCTensor_(data)(state, probDistContig));
  } else {
    // Generic, slow implementation with memory allocations

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    THCTensor* origDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, origDist, probDistContig);
    THCTensor_(copy)(state, origDist, probDistContig);

    THCTensor* normDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, normDist, probDistContig);

    THCTensor* prefixSum = THCTensor_(new)(state);

    // Renorm along rows
    THCTensor_(copy)(state, normDist, origDist);
    THCTensor_(renormRows)(state, normDist);

    // Prefix sum along rows
    THCTensor_(cumsum)(state, prefixSum, normDist, 1);

    if (with_replacement) {
      // Sample with replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from one
      // distribution concurrently.
      dim3 grid(numDist < MAX_NUM_BLOCKS ? numDist : MAX_NUM_BLOCKS);

      sampleMultinomialWithReplacement
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          gen->gen_states,
          n_sample,
          THCTensor_(data)(state, self),
          numDist, numCategories,
          THCTensor_(data)(state, prefixSum));
    } else {
      // Sample without replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from a different
      // distribution concurrently.
      ptrdiff_t numBlocks = THCCeilDiv(numDist, 4L);
      dim3 grid(numBlocks < MAX_NUM_BLOCKS ? numBlocks : MAX_NUM_BLOCKS);

      for (int sample = 0; sample < n_sample; ++sample) {
        if (sample > 0) {
          // Update probabilities
          // Renorm along rows
          THCTensor_(copy)(state, normDist, origDist);
          THCTensor_(renormRows)(state, normDist);

          // Prefix sum along rows
          THCTensor_(cumsum)(state, prefixSum, normDist, 1);
        }

        // The kernel can only draw one sample before we have to
        // recalculate our distribution
        sampleMultinomialWithoutReplacement
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
            gen->gen_states,
            n_sample,
            sample,
            THCTensor_(data)(state, self),
            numDist, numCategories,
            THCTensor_(data)(state, origDist),
            THCTensor_(data)(state, prefixSum));
      }
    }

    THCTensor_(free)(state, prefixSum);
    THCTensor_(free)(state, normDist);
    THCTensor_(free)(state, origDist);
  }

  // Revert data restructuring based on input sizes
  if (inputSize == 1) {
    THCTensor_(resize1d)(state, self, n_sample);

    // Unfortunately, if prob_dist is contiguous already,
    // newContiguous is not a private copy, so we have to restructure
    // this too, so as to not affect prob_dist
    THCTensor_(resize1d)(state, probDistContig, numCategories);
  }

  THCTensor_(free)(state, probDistContig);
}

THC_API void THCTensor_(rand)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(uniform)(state, r_, 0, 1);
}

void THCTensor_(randn)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(normal)(state, r_, 0, 1);
}

#endif

GENERATE_KERNEL1(generate_bernoulli, real, double p, float, curand_uniform, (ScalarConvert<bool, real>::to(x <= p)))

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

GENERATE_KERNEL1(generate_geometric, real, double p, float, curand_uniform, (ScalarConvert<float, real>::to((log(1-x) / log(p)) + 1)))

THC_API void THCTensor_(geometric)(THCState* state, THCTensor *self_, double p)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  ptrdiff_t size = THCTensor_(nElement)(state, self);
  real *data = THCTensor_(data)(state, self);

  generate_geometric<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, size, data, p);

  THCTensor_(freeCopyTo)(state, self, self_);
};
#undef NUM_BLOCKS

#endif
