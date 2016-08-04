#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"

#include <thrust/functional.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256

/* Sets up generator. Allocates but does not create the generator states. */
__host__ void initializeGenerator(THCState *state, Generator* gen)
{
  THCudaCheck(THCudaMalloc(state, (void**)&gen->gen_states, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  THCudaCheck(THCudaMalloc(state, (void**)&gen->kernel_params, sizeof(mtgp32_kernel_params)));
}

/* Frees memory allocated during setup. */
__host__ void destroyGenerator(THCState *state, Generator* gen)
{
  if (gen->gen_states)
  {
    THCudaCheck(THCudaFree(state, gen->gen_states));
    gen->gen_states = NULL;
  }
  if (gen->kernel_params)
  {
    THCudaCheck(THCudaFree(state, gen->kernel_params));
    gen->kernel_params = NULL;
  }
}

/* Creates a new generator state given the seed. */
__host__ void createGeneratorState(Generator* gen, unsigned long seed)
{
  if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, gen->kernel_params) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP constants failed.");
  }
  if (curandMakeMTGP32KernelState(gen->gen_states, mtgp32dc_params_fast_11213,
                                  gen->kernel_params, MAX_NUM_BLOCKS, seed) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP kernel state failed.");
  }
}

/* Initialize generator array (must be called before any other function) */
__host__ void THCRandom_init(THCState* state, int devices, int current_device)
{
  THCRNGState* rng_state = state->rngState;
  rng_state->num_devices = devices;
  rng_state->gen = (Generator*)malloc(rng_state->num_devices * sizeof(Generator));
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    rng_state->gen[i].initf = 0;
    rng_state->gen[i].initial_seed = 0;
    rng_state->gen[i].gen_states = NULL;
    rng_state->gen[i].kernel_params = NULL;
  }
  rng_state->current_gen = &rng_state->gen[current_device];
  // Initialize the generator for the current device. Other generators will be
  // initialized on-demand in THCRandom_setGenerator.
  initializeGenerator(state, rng_state->current_gen);
  THCRandom_seed(state);
}

/* Destroy generators and free memory */
__host__ void THCRandom_shutdown(THCState* state)
{
  THCRNGState* rng_state = state->rngState;
  if (rng_state->gen == NULL) return;
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    destroyGenerator(state, &rng_state->gen[i]);
  }
  free(rng_state->gen);
  rng_state->gen = NULL;
  rng_state->current_gen = NULL;
}

/* Set the generator for the current device */
__host__ void THCRandom_setGenerator(THCState* state, int device)
{
  THCRNGState* rng_state = state->rngState;
  if (device >= rng_state->num_devices) THError("Invalid device index.");
  rng_state->current_gen = &rng_state->gen[device];
  if (rng_state->current_gen->initf == 0)
  {
    initializeGenerator(state, rng_state->current_gen);
    THCRandom_seed(state);
  }
}

/* Random seed */
__host__ unsigned long THCRandom_seed(THCState* state)
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeed(state, s);
  return s;
}

__host__ unsigned long THCRandom_seedAll(THCState* state)
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeedAll(state, s);
  return s;
}

/* Manually set the seed */
__host__ void THCRandom_manualSeed(THCState* state, unsigned long seed)
{
  THCRNGState* rng_state = state->rngState;
  if (rng_state->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  rng_state->current_gen->initial_seed = seed;
  createGeneratorState(rng_state->current_gen, seed);
  rng_state->current_gen->initf = 1;
}

__host__ void THCRandom_manualSeedAll(THCState* state, unsigned long seed)
{
  THCRNGState* rng_state = state->rngState;
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));
  for (int i = 0; i < rng_state->num_devices; ++i) {
    THCudaCheck(cudaSetDevice(i));
    THCRandom_setGenerator(state, i);
    THCRandom_manualSeed(state, seed);
  }
  THCudaCheck(cudaSetDevice(currentDevice));
  THCRandom_setGenerator(state, currentDevice);
}

/* Get the initial seed */
__host__ unsigned long THCRandom_initialSeed(THCState* state)
{
  return state->rngState->current_gen->initial_seed;
}

__host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }

  // The RNG state comprises the MTPG32 states and the seed.
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THCudaCheck(cudaMemcpy(THByteTensor_data(rng_state), state->rngState->current_gen->gen_states,
                         states_size, cudaMemcpyDeviceToHost));
  memcpy(THByteTensor_data(rng_state) + states_size, &state->rngState->current_gen->initial_seed, seed_size);
}

__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
  state[threadIdx.x].k = kernel;
}

__host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }

  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  THCudaCheck(cudaMemcpy(state->rngState->current_gen->gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, state->rngState->current_gen->kernel_params);
  memcpy(&state->rngState->current_gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
}

#define GENERATE_KERNEL1(NAME, ARG1, CURAND_FUNC, TRANSFORM)                   \
__global__ void NAME(curandStateMtgp32 *state, int size, float *result, ARG1)  \
{                                                                              \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                             \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \
    float x = CURAND_FUNC(&state[blockIdx.x]);                                 \
    if (i < size) {                                                            \
      x = TRANSFORM;                                                           \
      result[i] = x;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, ARG1, ARG2, CURAND_FUNC, TRANSFORM)                   \
__global__ void NAME(curandStateMtgp32 *state, int size, float *result, ARG1, ARG2)  \
{                                                                                    \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                   \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                           \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    float x = CURAND_FUNC(&state[blockIdx.x]);                                       \
    if (i < size) {                                                                  \
      x = TRANSFORM;                                                                 \
      result[i] = x;                                                                 \
    }                                                                                \
  }                                                                                  \
}

GENERATE_KERNEL2(generate_uniform, double a, double b, curand_uniform, x * (b-a) + a)
GENERATE_KERNEL1(generate_bernoulli, double p, curand_uniform, (float)x <= p)
GENERATE_KERNEL2(generate_normal, double mean, double stdv, curand_normal, (x * stdv) + mean)
GENERATE_KERNEL1(generate_geometric, double p, curand_uniform, (log(1-x) / log(p)) + 1)
GENERATE_KERNEL1(generate_exponential, double lambda, curand_uniform, (float)(-1. / lambda * log(1-x)))
GENERATE_KERNEL2(generate_cauchy, double median, double sigma, curand_uniform, (float)(median + sigma * tan(M_PI*(x-0.5))))

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2

/* Separate kernel because curand_log_normal gets extra parameters. */
__global__ void generate_log_normal(curandStateMtgp32 *state, int size, float *result, float mean, float stddev)
{
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float x = curand_log_normal(&state[blockIdx.x], mean, stddev);
    if (i < size) {
      result[i] = x;
    }
  }
}

#define NUM_BLOCKS min((int)THCCeilDiv(size, (long) BLOCK_SIZE), MAX_NUM_BLOCKS)
THC_API void THCudaTensor_uniform(THCState* state, THCudaTensor *self_, double a, double b)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_uniform<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, a, b);

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_bernoulli(THCState* state, THCudaTensor *self_, double p)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_bernoulli<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, p);

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_normal(THCState* state, THCudaTensor *self_, double mean, double stdv)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_normal<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, mean, stdv);

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_logNormal(THCState* state, THCudaTensor *self_, double mean, double stdv)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_log_normal<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, mean, stdv);

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_geometric(THCState* state, THCudaTensor *self_, double p)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_geometric<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, p);

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_exponential(THCState* state, THCudaTensor *self_, double lambda)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_exponential<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, lambda);

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_cauchy(THCState* state, THCudaTensor *self_, double median, double sigma)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  if (state->rngState->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  long size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);

  generate_cauchy<<<NUM_BLOCKS, BLOCK_SIZE, 0, THCState_getCurrentStream(state)>>>(
      state->rngState->current_gen->gen_states, size, data, median, sigma);

  THCudaTensor_freeCopyTo(state, self, self_);
};

__device__ int binarySearchForMultinomial(float* dist,
                                          int size,
                                          float val) {
  int start = 0;
  int end = size;

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    float midVal = dist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first element
    start = 0;
  }

  return start;
}

// Normalizes the L1 norm of every row to 1; used by multinomial
__global__ void renormRowsL1(float* dist, long rows, long cols) {
  extern __shared__ float smem[];

  for (long row = blockIdx.x; row < rows; row += gridDim.x) {
    float sum = 0.0f;
    for (long col = threadIdx.x; col < cols; col += blockDim.x) {
      sum += dist[row * cols + col];
    }

    sum = reduceBlock(smem, blockDim.x, sum, thrust::plus<float>(), 0.0f);
    if (threadIdx.x == 0) {
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (sum > 0.0f) {
      for (long col = threadIdx.x; col < cols; col += blockDim.x) {
        dist[row * cols + col] /= sum;
      }
    }
  }
}

void THCudaTensor_renormRows(struct THCState* state,
                             THCudaTensor* t) {
  THAssert(THCudaTensor_nDimension(state, t) == 2);
  long rows = THCudaTensor_size(state, t, 0);
  long cols = THCudaTensor_size(state, t, 1);

  cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);

  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

  renormRowsL1
    <<<grid, block, block.x * sizeof(float),
    THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, t),
                                        rows, cols);
}

__global__ void
sampleMultinomialOnce(float* dest,
                      long distributions,
                      int categories,
                      float* dist) {
  extern __shared__ float smem[];

  for (long curDist = blockIdx.x;
       curDist < distributions; curDist += gridDim.x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    float sum = 0.0f;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      sum += dist[curDist * categories + cat];
    }

    // threadIdx.x == 0 has the sum value from this
    sum = reduceBlock(smem, blockDim.x, sum, thrust::plus<float>(), 0.0f);

    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      smem[0] = sum;
      smem[1] = dest[curDist];
    }
    __syncthreads();

    sum = smem[0];
    float sample = smem[1];
    __syncthreads();

    if (sum == 0.0f || sample == 0.0f) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = 1;
      }

      continue;
    }

    int chunks = THCCeilDiv(categories, (int) blockDim.x);
    float prevHighProb = 0.0f;

    for (int chunk = 0; chunk < chunks; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;

      float val =
        cat < categories ? dist[curDist * categories + cat] / sum : 0.0f;
      smem[threadIdx.x] = val;
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0.0f;

        if (threadIdx.x >= offset) {
          val = smem[threadIdx.x - offset] + smem[threadIdx.x];
        }

        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      float curBucket =
        smem[threadIdx.x] + prevHighProb;
      float prevBucket =
        threadIdx.x == 0 ? prevHighProb : smem[threadIdx.x - 1] + prevHighProb;
      bool inBucket =
        (cat < categories) && (sample <= curBucket) && (sample > prevBucket);

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        // FIXME: broadcast exit flag?
        dest[curDist] = cat + 1;
      }

      // Store the previous scan's high value for future use
      prevHighProb += smem[blockDim.x - 1];

      __syncthreads();
    }
  }
}

__global__ void
sampleMultinomialWithReplacement(curandStateMtgp32* state,
                                 int totalSamples,
                                 float* dest,
                                 long distributions,
                                 int categories,
                                 float* normDistPrefixSum) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block determines the distribution for which we generate a point
  for (long curDist = blockIdx.x;
       curDist < distributions;
       curDist += gridDim.x) {
    for (int sampleBase = 0;
         sampleBase < totalSamples; sampleBase += blockDim.y) {
      // The warp determines the sample
      int sample = sampleBase + threadIdx.y;

      // All threads participate in this
      float r = curand_uniform(&state[blockIdx.x]);

      if (threadIdx.x == 0 && sample < totalSamples) {
        // Find the bucket that a uniform sample lies in
        int choice = binarySearchForMultinomial(
          normDistPrefixSum + curDist * categories,
          categories,
          r);

        // Torch indices are 1-based
        dest[curDist * totalSamples + sample] = (float) choice + 1.0f;
      }
    }
  }
}

__global__ void
sampleMultinomialWithoutReplacement(curandStateMtgp32* state,
                                    int totalSamples,
                                    int sample,
                                    float* dest,
                                    long distributions,
                                    int categories,
                                    float* origDist,
                                    float* normDistPrefixSum) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block and warp determines the distribution for which we
  // generate a point
  for (long curDistBase = blockIdx.x * blockDim.y;
       curDistBase < distributions;
       curDistBase += gridDim.x * blockDim.y) {
    // The warp determines the distribution
    long curDist = curDistBase + threadIdx.y;

    // All threads must participate in this
    float r = curand_uniform(&state[blockIdx.x]);

    if (threadIdx.x == 0 && curDist < distributions) {
      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial(
        normDistPrefixSum + curDist * categories,
        categories,
        r);

      // Torch indices are 1-based
      dest[curDist * totalSamples + sample] = (float) choice + 1.0f;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      origDist[curDist * categories + choice] = 0.0f;
    }
  }
}

THC_API void THCudaTensor_multinomial(struct THCState *state,
                                      THCudaTensor *self,
                                      THCudaTensor *prob_dist,
                                      int n_sample,
                                      int with_replacement)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, prob_dist));
  if (state->rngState->current_gen == NULL) {
    THError("Random number generators have not been initialized.");
  }

  int inputSize = THCudaTensor_nDimension(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");

  // Categories are in the innermost dimension
  long numDist =
    inputSize == 1 ? 1 : THCudaTensor_size(state, prob_dist, 0);
  long numCategoriesLong =
    inputSize == 1 ? THCudaTensor_size(state, prob_dist, 0) :
    THCudaTensor_size(state, prob_dist, 1);

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
  THCudaTensor* probDistContig =
    THCudaTensor_newContiguous(state, prob_dist);

  // Restructure data for 2d
  if (inputSize == 1) {
    THCudaTensor_resize2d(state, probDistContig, 1, numCategories);
  }

  THCudaTensor_resize2d(state, self, numDist, n_sample);

  if (n_sample == 1) {
    // Optimized allocation-free implementation

    // To exploit greater parallelism for the sampling, generate the
    // Uniform random samples in a separate kernel launch, into the
    // result memory. The device RNG is thread-limited
    THCudaTensor_uniform(state, self, 0.0, 1.0);

    cudaDeviceProp* props = THCState_getCurrentDeviceProperties(state);
    THAssert(props != NULL);

    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;

    dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
    dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

    sampleMultinomialOnce
      <<<grid, block, block.x * sizeof(float),
         THCState_getCurrentStream(state)>>>(
      THCudaTensor_data(state, self),
      numDist,
      numCategories,
      THCudaTensor_data(state, probDistContig));
  } else {
    // Generic, slow implementation with memory allocations

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    THCudaTensor* origDist = THCudaTensor_new(state);
    THCudaTensor_resizeAs(state, origDist, probDistContig);
    THCudaTensor_copy(state, origDist, probDistContig);

    THCudaTensor* normDist = THCudaTensor_new(state);
    THCudaTensor_resizeAs(state, normDist, probDistContig);

    THCudaTensor* prefixSum = THCudaTensor_new(state);

    // Renorm along rows
    THCudaTensor_copy(state, normDist, origDist);
    THCudaTensor_renormRows(state, normDist);

    // Prefix sum along rows
    THCudaTensor_cumsum(state, prefixSum, normDist, 1);

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
          state->rngState->current_gen->gen_states,
          n_sample,
          THCudaTensor_data(state, self),
          numDist, numCategories,
          THCudaTensor_data(state, prefixSum));
    } else {
      // Sample without replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from a different
      // distribution concurrently.
      long numBlocks = THCCeilDiv(numDist, 4L);
      dim3 grid(numBlocks < MAX_NUM_BLOCKS ? numBlocks : MAX_NUM_BLOCKS);

      for (int sample = 0; sample < n_sample; ++sample) {
        if (sample > 0) {
          // Update probabilities
          // Renorm along rows
          THCudaTensor_copy(state, normDist, origDist);
          THCudaTensor_renormRows(state, normDist);

          // Prefix sum along rows
          THCudaTensor_cumsum(state, prefixSum, normDist, 1);
        }

        // The kernel can only draw one sample before we have to
        // recalculate our distribution
        sampleMultinomialWithoutReplacement
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
            state->rngState->current_gen->gen_states,
            n_sample,
            sample,
            THCudaTensor_data(state, self),
            numDist, numCategories,
            THCudaTensor_data(state, origDist),
            THCudaTensor_data(state, prefixSum));
      }
    }

    THCudaTensor_free(state, prefixSum);
    THCudaTensor_free(state, normDist);
    THCudaTensor_free(state, origDist);
  }

  // Revert data restructuring based on input sizes
  if (inputSize == 1) {
    THCudaTensor_resize1d(state, self, n_sample);

    // Unfortunately, if prob_dist is contiguous already,
    // newContiguous is not a private copy, so we have to restructure
    // this too, so as to not affect prob_dist
    THCudaTensor_resize1d(state, probDistContig, numCategories);
  }

  THCudaTensor_free(state, probDistContig);
}

#undef NUM_BLOCKS
