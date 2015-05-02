#include "THCTensorRandom.h"
#include "THCGeneral.h"

#include <thrust/functional.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

/* Sets up generator. Allocates but does not create the generator states. */
__host__ void initializeGenerator(Generator* gen)
{
  THCudaCheck(cudaMalloc((void**)&gen->gen_states, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  THCudaCheck(cudaMalloc((void**)&gen->kernel_params, sizeof(mtgp32_kernel_params)));
}

/* Frees memory allocated during setup. */
__host__ void destroyGenerator(Generator* gen)
{
  if (gen->gen_states)
  {
    THCudaCheck(cudaFree(gen->gen_states));
    gen->gen_states = NULL;
  }
  if (gen->kernel_params)
  {
    THCudaCheck(cudaFree(gen->kernel_params));
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
  initializeGenerator(rng_state->current_gen);
  THCRandom_seed(state);
}

/* Destroy generators and free memory */
__host__ void THCRandom_shutdown(THCState* state)
{
  THCRNGState* rng_state = state->rngState;
  if (rng_state->gen == NULL) return;
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    destroyGenerator(&rng_state->gen[i]);
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
    initializeGenerator(rng_state->current_gen);
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

__host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THCudaCheck(cudaMemcpy(state->rngState->current_gen->gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  memcpy(&state->rngState->current_gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
}

#define GENERATE_KERNEL1(NAME, ARG1, CURAND_FUNC, TRANSFORM)                   \
__global__ void NAME(curandStateMtgp32 *state, int size, float *result, ARG1)  \
{                                                                              \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                             \
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
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
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;                           \
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
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float x = curand_log_normal(&state[blockIdx.x], mean, stddev);
    if (i < size) {
      result[i] = x;
    }
  }
}

#define NUM_BLOCKS min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS)
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
#undef NUM_BLOCKS
