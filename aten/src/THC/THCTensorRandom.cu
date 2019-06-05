#include <THC/THCTensorRandom.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorRandom.cuh>
#include <THC/THCGenerator.hpp>
#include <ATen/Config.h>

#include <ATen/cuda/_curand_mtgp32_host.h>

#include <thrust/functional.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256


THCGenerator* THCRandom_getGenerator(THCState* state);

/* Sets up generator. Allocates but does not create the generator states. Not thread-safe. */
__host__ void initializeGenerator(THCState *state, THCGenerator* gen)
{
  gen->state.gen_states = static_cast<curandStateMtgp32*>(THCudaMalloc(state, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  gen->state.kernel_params = static_cast<mtgp32_kernel_params*>(THCudaMalloc(state, sizeof(mtgp32_kernel_params)));
}

/* Creates a new generator state given the seed. Not thread-safe. */
__host__ void createGeneratorState(THCGenerator* gen, uint64_t seed)
{
  if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, gen->state.kernel_params) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP constants failed.");
  }
  if (curandMakeMTGP32KernelState(gen->state.gen_states, mtgp32dc_params_fast_11213,
                                  gen->state.kernel_params, MAX_NUM_BLOCKS, seed) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP kernel state failed.");
  }
  // seed and offset for philox
  gen->state.initial_seed = seed;
  gen->state.philox_seed_offset = 0;
}

THC_API __host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  THCGenerator* gen = THCRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  // The RNG state comprises the MTPG32 states, the seed, and an offset used for Philox
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->state.initial_seed);
  static const size_t offset_size = sizeof(gen->state.philox_seed_offset);
  static const size_t total_size = states_size + seed_size + offset_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THCudaCheck(cudaMemcpy(THByteTensor_data(rng_state), gen->state.gen_states,
                         states_size, cudaMemcpyDeviceToHost));
  memcpy(THByteTensor_data(rng_state) + states_size, &gen->state.initial_seed, seed_size);
  memcpy(THByteTensor_data(rng_state) + states_size + seed_size, &gen->state.philox_seed_offset, offset_size);
}

__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
#ifndef __HIP_PLATFORM_HCC__
  state[threadIdx.x].k = kernel;
#else
  state[threadIdx.x].set_params(kernel);
#endif
}

THC_API __host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  THCGenerator* gen = THCRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->state.initial_seed);
  static const size_t offset_size = sizeof(gen->state.philox_seed_offset);
  static const size_t total_size = states_size + seed_size + offset_size;
  bool no_philox_seed = false;
  if (THByteTensor_nElement(rng_state) == total_size - offset_size) {
    no_philox_seed = true;
  }
  else {
    THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  }
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  THCudaCheck(cudaMemcpy(gen->state.gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, gen->state.kernel_params);
  memcpy(&gen->state.initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
  if (!no_philox_seed) {
    memcpy(&gen->state.philox_seed_offset, THByteTensor_data(rng_state) + states_size + seed_size, offset_size);
  }
  else {
    gen->state.philox_seed_offset = 0;
  }
}

#define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)    \
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
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1, ARG2)    \
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

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateBoolType.h>

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2
