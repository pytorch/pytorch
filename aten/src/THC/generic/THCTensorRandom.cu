#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorRandom.cu"
#else

#include "ATen/cuda/CUDAContext.h"

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

void THCTensor_(uniform)(THCState* state, THCTensor *self_, double a, double b)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  generate_uniform<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, size, data, a, b);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(normal)(THCState* state, THCTensor *self_, double mean, double stdv)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  generate_normal<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, size, data, mean, stdv);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(normal_means)(THCState *state, THCTensor *self, THCTensor *means, double stddev) {
  THCTensor_(resizeAs)(state, self, means);
  THCTensor_(normal)(state, self, 0, stddev);
  THCTensor_(cadd)(state, self, self, ScalarConvert<int, scalar_t>::to(1), means);
}

void THCTensor_(normal_stddevs)(THCState *state, THCTensor *self, double mean, THCTensor *stddevs)
{
  THCTensor_(resizeAs)(state, self, stddevs);
  THCTensor_(normal)(state, self, 0, 1);
  THCTensor_(cmul)(state, self, self, stddevs);
  THCTensor_(add)(state, self, self, ScalarConvert<double, scalar_t>::to(mean));
}

void THCTensor_(normal_means_stddevs)(THCState *state, THCTensor *self, THCTensor *means, THCTensor *stddevs)
{
  THCTensor_(resizeAs)(state, self, means);
  THCTensor_(normal)(state, self, 0, 1);
  THCTensor_(cmul)(state, self, self, stddevs);
  THCTensor_(cadd)(state, self, self, ScalarConvert<int, scalar_t>::to(1), means);
}

void THCTensor_(logNormal)(THCState* state, THCTensor *self_, double mean, double stdv)
{

  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  generateLogNormal<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, size, data, mean, stdv);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(exponential)(THCState* state, THCTensor *self_, double lambda)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  generate_exponential<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, size, data, lambda);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(cauchy)(THCState* state, THCTensor *self_, double median, double sigma)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  generate_cauchy<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, size, data, median, sigma);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(renormRows)(struct THCState* state,
                             THCTensor* t) {
  THAssert(THCTensor_(nDimensionLegacyAll)(state, t) == 2);
  int64_t rows = THCTensor_(size)(state, t, 0);
  int64_t cols = THCTensor_(size)(state, t, 1);

  cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
  THAssert(props != NULL);

  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

  renormRowsL1<scalar_t>
    <<<grid, block, block.x * sizeof(scalar_t),
    THCState_getCurrentStream(state)>>>(THCTensor_(data)(state, t),
                                        rows, cols);
}

void THCTensor_(multinomial)(struct THCState *state,
                              THCudaLongTensor *self,
                              THCTensor *prob_dist,
                              int n_sample,
                              int with_replacement)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self, prob_dist));
  THCGenerator* gen = THCRandom_getGenerator(state);

  int inputSize = THCTensor_(nDimensionLegacyAll)(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");

  // Categories are in the innermost dimension
  int64_t numDist =
    inputSize == 1 ? 1 : THCTensor_(sizeLegacyNoScalars)(state, prob_dist, 0);
  int64_t numCategoriesLong =
    inputSize == 1 ? THCTensor_(sizeLegacyNoScalars)(state, prob_dist, 0) :
    THCTensor_(sizeLegacyNoScalars)(state, prob_dist, 1);

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

  int free_prob_dist = 0;

  // Restructure data for 2d
  if (inputSize == 1) {
    THCTensor *temp = THCTensor_(new)(state);
    THCTensor_(unsqueeze1d)(state, temp, prob_dist, 0);
    prob_dist = temp;
    free_prob_dist = 1;
  }

  THCudaLongTensor_resize2d(state, self, numDist, n_sample);

  // get current device properties
  cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
  THAssert(props != NULL);
  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;
  int maxShared = props->sharedMemPerBlock;
  int requiredShared = (numCategories < maxThreads ? numCategories : maxThreads)
                                * (sizeof(scalar_t) + sizeof(accreal));

  if (n_sample == 1 && maxShared >= requiredShared) {
    // Optimized allocation-free implementation
    // To exploit greater parallelism for the sampling, generate the
    // Uniform random samples in a separate kernel launch, into
    // temporarily allocated memory. The device RNG is thread-limited
    THCTensor *sampled = THCTensor_(newWithSize2d)(state, numDist, n_sample);
    THCTensor_(uniform)(state, sampled, 0.0, 1.0);

    dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
    dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

    sampleMultinomialOnce<scalar_t, accreal>
      <<<grid, block,
         requiredShared,
         THCState_getCurrentStream(state)>>>(
      THCudaLongTensor_data(state, self),
      numDist,
      numCategories,
      THCTensor_(data)(state, sampled),
      THCTensor_(data)(state, prob_dist),
      THCTensor_(stride)(state, prob_dist, 0),
      THCTensor_(stride)(state, prob_dist, 1)
      );
    THCTensor_(free)(state, sampled);
  } else {
    // Generic, slow implementation with memory allocations

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    THCTensor* origDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, origDist, prob_dist);
    THCTensor_(copy)(state, origDist, prob_dist);

    THCTensor* normDist = THCTensor_(new)(state);
    THCTensor_(resizeAs)(state, normDist, prob_dist);

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
          gen->state.gen_states,
          n_sample,
          THCudaLongTensor_data(state, self),
          numDist, numCategories,
          THCTensor_(data)(state, prefixSum),
          THCTensor_(data)(state, normDist));
    } else {
      // Sample without replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from a different
      // distribution concurrently.
      ptrdiff_t numBlocks = THCCeilDiv(numDist, (int64_t) 4);
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
            gen->state.gen_states,
            n_sample,
            sample,
            THCudaLongTensor_data(state, self),
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
    THCudaLongTensor_resize1d(state, self, n_sample);
  }
  if (free_prob_dist) {
    THCTensor_(free)(state, prob_dist);
  }
}

void THCTensor_(multinomialAliasSetup)(THCState *state, THCTensor *_probs, THCudaLongTensor *_J, THCTensor *_q){
  THArgCheck(_probs->dim() == 1, 1,
             "expected 1-D probability tensor, got %d-D probability tensor instead",
             _probs->dim());
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  THCTensor *probs = THCTensor_(newContiguous)(state, _probs);
  THAssert(THCTensor_(isContiguous)(state, probs));
  int64_t inputsize = THCTensor_(nElement)(state, probs);
  THCudaLongTensor *smaller = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *smaller_short = THCudaLongTensor_newWithSize1d(state, inputsize);
  THCudaLongTensor *larger_short = THCudaLongTensor_newWithSize1d(state, inputsize);

  THCudaLongTensor_resize1d(state, _J, inputsize);
  THCTensor_(resize1d)(state, _q, inputsize);

  scalar_t one = ScalarConvert<int64_t, scalar_t>::to(1);
  int inputBlockDim = THCCeilDiv((int)inputsize + BLOCK_SIZE - 1, BLOCK_SIZE);
  aliasMultinomialFilter
    <<<inputBlockDim, BLOCK_SIZE, 0, THCState_getCurrentStream(state) >>>(
                     THCTensor_(data)(state, _q),
                     THCTensor_(data)(state, probs),
                     THCudaLongTensor_data(state, smaller),
                     THCudaLongTensor_data(state, larger),
                     THCudaLongTensor_data(state, _J),
                     THCudaLongTensor_data(state, smaller_short),
                     THCudaLongTensor_data(state, larger_short),
                     one, inputsize
                     );

  THCudaLongTensor_nonzero(state, smaller_short, smaller);
  THCudaLongTensor_nonzero(state, larger_short, larger);
  int h_large_c = THCudaLongTensor_nElement(state, larger_short);
  THCudaLongTensor_resize1d(state, smaller_short, inputsize);
  THCudaLongTensor_resize1d(state, larger_short, inputsize);
  aliasMultinomialSetup
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
                THCudaLongTensor_data(state, _J),
                THCTensor_(data)(state, _q),
                inputsize,
                THCudaLongTensor_data(state, smaller_short),
                THCudaLongTensor_data(state, larger_short),
                inputsize - h_large_c, h_large_c
                );
  scalar_t q_max = THCTensor_(maxall)(state, _q);
  condDiv<<<
    inputBlockDim, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
                      THCTensor_(data)(state, _q),
                      THCudaLongTensor_data(state, _J),
                      inputsize, q_max
                      );

  THCudaLongTensor_free(state, smaller);
  THCudaLongTensor_free(state, larger);
  THCudaLongTensor_free(state, smaller_short);
  THCudaLongTensor_free(state, larger_short);
  THCTensor_free(state, probs);
}

void THCTensor_(multinomialAliasDraw)(THCState *state, THCudaLongTensor *self, THCTensor *_q, THCudaLongTensor *_J, int n_sample){
  THArgCheck(_q->dim() == 1, 1,
             "expected 1-D probability table, got %d-D probability table instead",
             _q->dim());
  THArgCheck(_J->dim() == 1, 2,
             "expected 1-D alias table, got %d-D alias table instead",
             _J->dim());
  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");
  THAssert(THCTensor_(isContiguous)(state, _q));
  THAssert(THCudaLongTensor_isContiguous(state, _J));
  THCGenerator* gen = THCRandom_getGenerator(state);
  int64_t K = THCudaLongTensor_nElement(state, _J);
  THCudaLongTensor_resize1d(state, self, n_sample);
  ptrdiff_t size = THCudaLongTensor_nElement(state, self);

  THCTensor *uniform = THCTensor_(newWithSize1d)(state, n_sample);
  THCTensor *bernoulli = THCTensor_(newWithSize1d)(state, n_sample);

  THCTensor_(uniform)(state, uniform, 0, K);
  THCTensor_(uniform)(state, bernoulli, 0, 1);

  multinomialAliasDrawKernel
    <<<THCCeilDiv((int)n_sample+BLOCK_SIZE-1, BLOCK_SIZE), BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
          size,
          THCudaLongTensor_data(state, self),
          THCudaLongTensor_data(state, _J),
          THCTensor_(data)(state, _q),
          K,
          THCTensor_(data)(state, uniform),
          THCTensor_(data)(state, bernoulli)
          );
  THCTensor_(free)(state, uniform);
  THCTensor_(free)(state, bernoulli);
}

#endif

#if defined(THC_REAL_IS_DOUBLE)
GENERATE_KERNEL1(generate_geometric, double, double p, double, curand_uniform_double, ceil(log(x) / log(1-p)))
#else
GENERATE_KERNEL1(generate_geometric, scalar_t, double p, float, curand_uniform, (ScalarConvert<float, scalar_t>::to(ceilf(logf(x) / log(1-p)))))
#endif

#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
#define CURAND64(STATE) (((uint64_t)curand(STATE)) << 32) | (uint64_t)curand(STATE)
GENERATE_KERNEL2(generate_random, scalar_t, int32_t base, uint32_t range, uint32_t, curand, \
    static_cast<scalar_t>(static_cast<int32_t>((x % range) + base)))
GENERATE_KERNEL2(generate_random_64, scalar_t, int64_t base, uint64_t range, uint64_t, CURAND64, \
    static_cast<scalar_t>(static_cast<int64_t>((x % range) + base)))
#elif defined(THC_REAL_IS_HALF)
GENERATE_KERNEL2(generate_random, scalar_t, int32_t base, uint32_t range, uint32_t, curand,
    (ScalarConvert<int32_t, scalar_t>::to(static_cast<int32_t>(x % range + base))))
#else
GENERATE_KERNEL2(generate_random, scalar_t, int32_t base, uint32_t range, uint32_t, curand,
    static_cast<scalar_t>(static_cast<int32_t>(x % range + base)))
#endif

void THCTensor_(geometric)(THCState* state, THCTensor *self_, double p)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);

  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  generate_geometric<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, size, data, p);

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(clampedRandom)(THCState* state, THCTensor *self_, int64_t min_val, int64_t max_val)
{
  THArgCheck(min_val < max_val, 2,
             "max must be greater than min, but got: min = %lld, max = %lld", min_val, max_val);
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

  uint64_t range = max_val - min_val;

#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
  if (range > 1ULL << 32) {
    generate_random_64<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        gen->state.gen_states, static_cast<int>(size), data, min_val, range);
  } else {
#endif
    generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
        gen->state.gen_states, static_cast<int>(size), data, static_cast<int32_t>(min_val), static_cast<uint32_t>(range));
#if defined(THC_REAL_IS_LONG) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_FLOAT)
  }
#endif

  THCTensor_(freeCopyTo)(state, self, self_);
};

void THCTensor_(cappedRandom)(THCState* state, THCTensor *self_, int64_t max_val)
{
  THCTensor_(clampedRandom)(state, self_, 0LL, max_val);
};

#define HLF_MANT_DIG 11

void THCTensor_(random)(THCState* state, THCTensor *self_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  ptrdiff_t size = THCTensor_(nElement)(state, self_);
  if (size == 0) return;
  THCGenerator* gen = THCRandom_getGenerator(state);
  THCTensor *self = THCTensor_(newContiguous)(state, self_);
  scalar_t *data = THCTensor_(data)(state, self);

#if defined(THC_REAL_IS_HALF)
  generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, static_cast<int>(size), data, static_cast<int32_t>(0UL), static_cast<uint32_t>((1UL << HLF_MANT_DIG) + 1));
#elif defined(THC_REAL_IS_FLOAT)
  generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, static_cast<int>(size), data, static_cast<int32_t>(0UL), static_cast<uint32_t>((1UL << FLT_MANT_DIG) + 1));
#elif defined(THC_REAL_IS_DOUBLE)
  generate_random_64<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, static_cast<int>(size), data, static_cast<int64_t>(0ULL), static_cast<uint64_t>((1ULL << DBL_MANT_DIG) + 1));
#elif defined(THC_REAL_IS_LONG)
  generate_random_64<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, static_cast<int>(size), data, static_cast<int64_t>(0ULL), static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1);
#else
  generate_random<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, static_cast<int>(size), data, static_cast<int32_t>(0UL), static_cast<uint32_t>(std::numeric_limits<scalar_t>::max()) + 1);
#endif

  THCTensor_(freeCopyTo)(state, self, self_);
};

#undef HLF_MANT_DIG
#undef CURAND64
#undef NUM_BLOCKS

#endif
