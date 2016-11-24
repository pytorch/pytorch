#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorRandom.cuh"

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
  THCRNGState* rng_state = THCState_getRngState(state);
  rng_state->num_devices = devices;
  rng_state->gen = (Generator*)malloc(rng_state->num_devices * sizeof(Generator));
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    rng_state->gen[i].initf = 0;
    rng_state->gen[i].initial_seed = 0;
    rng_state->gen[i].gen_states = NULL;
    rng_state->gen[i].kernel_params = NULL;
  }
}

/* Destroy generators and free memory */
__host__ void THCRandom_shutdown(THCState* state)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  if (rng_state->gen == NULL) return;
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    destroyGenerator(state, &rng_state->gen[i]);
  }
  free(rng_state->gen);
  rng_state->gen = NULL;
}

/* Manually set the generator seed */
__host__ static void THCRandom_manualSeedGen(Generator* gen, unsigned long seed)
{
  gen->initial_seed = seed;
  createGeneratorState(gen, seed);
  gen->initf = 1;
}

/* Get the generator for the current device */
__host__ Generator* THCRandom_getGenerator(THCState* state)
{
  THCRNGState* rng_state = THCState_getRngState(state);

  int device;
  THCudaCheck(cudaGetDevice(&device));
  if (device >= rng_state->num_devices) THError("Invalid device index.");

  Generator* gen = &rng_state->gen[device];
  if (gen->initf == 0)
  {
    initializeGenerator(state, gen);
    THCRandom_manualSeedGen(gen, (unsigned long)time(0));
  }
  return gen;
}

__host__ struct curandStateMtgp32* THCRandom_generatorStates(struct THCState* state)
{
  return THCRandom_getGenerator(state)->gen_states;
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
  Generator* gen = THCRandom_getGenerator(state);
  THCRandom_manualSeedGen(gen, seed);
}

__host__ void THCRandom_manualSeedAll(THCState* state, unsigned long seed)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));
  for (int i = 0; i < rng_state->num_devices; ++i) {
    THCudaCheck(cudaSetDevice(i));
    THCRandom_manualSeed(state, seed);
  }
  THCudaCheck(cudaSetDevice(currentDevice));
}

/* Get the initial seed */
__host__ unsigned long THCRandom_initialSeed(THCState* state)
{
  return THCRandom_getGenerator(state)->initial_seed;
}

__host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

  // The RNG state comprises the MTPG32 states and the seed.
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THCudaCheck(cudaMemcpy(THByteTensor_data(rng_state), gen->gen_states,
                         states_size, cudaMemcpyDeviceToHost));
  memcpy(THByteTensor_data(rng_state) + states_size, &gen->initial_seed, seed_size);
}

__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
  state[threadIdx.x].k = kernel;
}

__host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  THCudaCheck(cudaMemcpy(gen->gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, gen->kernel_params);
  memcpy(&gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
}

#define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)               \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)  \
{                                                                              \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                             \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                                 \
    if (i < size) {                                                            \
      T y = TRANSFORM;                                                           \
      result[i] = y;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, CURAND_T, CURAND_FUNC, TRANSFORM)                \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1, ARG2)  \
{                                                                                    \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                   \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                           \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                                       \
    if (i < size) {                                                                  \
      T y = TRANSFORM;                                                                 \
      result[i] = y;                                                                 \
    }                                                                                \
  }                                                                                  \
}

GENERATE_KERNEL2(generate_uniform, float, double a, double b, float, curand_uniform, x * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, double, double a, double b, double, curand_uniform_double, x * (b-a) + a)

GENERATE_KERNEL2(generate_normal, float, double mean, double stdv, float, curand_normal, (x * stdv) + mean)
GENERATE_KERNEL2(generate_normal, double, double mean, double stdv, double, curand_normal_double, (x * stdv) + mean)

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, curand_uniform, (float)(-1. / lambda * log(1-x)))
GENERATE_KERNEL1(generate_exponential, double, double lambda, double, curand_uniform_double, (double)(-1. / lambda * log(1-x)))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, curand_uniform, (float)(median + sigma * tan(M_PI*(x-0.5))))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, double, curand_uniform_double, (double)(median + sigma * tan(M_PI*(x-0.5))))

#ifdef CUDA_HALF_TENSOR
GENERATE_KERNEL2(generate_uniform, half, double a, double b, float, curand_uniform, (ScalarConvert<float, half>::to(x * (b-a) + a)))
GENERATE_KERNEL2(generate_normal, half, double mean, double stdv, float, curand_normal, (ScalarConvert<float, half>::to((x * stdv) + mean)))
GENERATE_KERNEL1(generate_exponential, half, double lambda, float, curand_uniform, (ScalarConvert<float, half>::to((float)(-1. / lambda * log(1-x)))))
GENERATE_KERNEL2(generate_cauchy, half, double median, double sigma, float, curand_uniform, (ScalarConvert<float, half>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))
#endif // CUDA_HALF_TENSOR

#include "generic/THCTensorRandom.cu"
#include "THCGenerateAllTypes.h"

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2

