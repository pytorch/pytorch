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


Generator* THCRandom_getGenerator(THCState* state);

/* Sets up generator. Allocates but does not create the generator states. */
__host__ void initializeGenerator(THCState *state, Generator* gen)
{
  THCudaCheck(THCudaMalloc(state, (void**)&gen->gen_states, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  THCudaCheck(THCudaMalloc(state, (void**)&gen->kernel_params, sizeof(mtgp32_kernel_params)));
}

/* Creates a new generator state given the seed. */
__host__ void createGeneratorState(Generator* gen, unsigned long long seed)
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

__host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

  // The RNG state comprises the MTPG32 states and the seed.
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->initial_seed);
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
  static const size_t seed_size = sizeof(gen->initial_seed);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  THCudaCheck(cudaMemcpy(gen->gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, gen->kernel_params);
  memcpy(&gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
}

#define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)      \
{                                                                              \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                             \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                              \
    if (i < size) {                                                            \
      T y = TRANSFORM;                                                         \
      result[i] = y;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1, ARG2)      \
{                                                                                    \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                   \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                      \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                                    \
    if (i < size) {                                                                  \
      T y = TRANSFORM;                                                               \
      result[i] = y;                                                                 \
    }                                                                                \
  }                                                                                  \
}

template<typename T, typename U>
struct is_same { static const bool value = false; };

template<typename T>
struct is_same<T, T> { static const bool value = true; };

template<typename real, typename prob_type>
__global__ void generate_bernoulli_tensor(curandStateMtgp32 *state, int size,
        real *result, prob_type *probs)
{
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    if (is_same<prob_type, double>::value) {
      double x = curand_uniform_double(&state[blockIdx.x]);
      if (i < size)
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
    } else {
      float x = curand_uniform(&state[blockIdx.x]);
      if (i < size)
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
    }
  }
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
