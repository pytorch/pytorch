#include <THC/THCTensorRandom.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorRandom.cuh>
#include <ATen/Config.h>

#include <thrust/functional.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256

// NB: ROCm compiler seems to have a bug where __host__ functions must be
// explicitly specified extern "C" otherwise ROCm compiler doesn't respect it.
// See https://github.com/RadeonOpenCompute/hcc/issues/839
extern "C" __host__ void THCRandom_getRNGState(at::Generator gen_, THByteTensor *rng_state)
{
  auto gen = at::check_generator<at::CUDAGenerator>(gen_);
  std::lock_guard<std::mutex> lock(gen->mutex_);
  // The RNG state comprises the seed, and an offset used for Philox.
  // The following line is just here for BC reason. sizeof curandStateMtgp32 is 4120.
  // It used to be static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  // MAX_NUM_BLOCKS was 200 and sizeof(curandStateMtgp32) is 4120. Hardcoding these numbers here
  // because this is just host side code and we don't want to worry about linking with cuda
  static const size_t states_size = 200 * sizeof(4120);
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = states_size + seed_size + offset_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  // since curandStateMTGP is not used anymore, fill gen_states of THCGenerator with deterministic garbage value of -1
  // gen_states in THCGenerator struct was an array of curandStateMtgp32s.
  memset(THByteTensor_data(rng_state), -1, states_size);
  auto current_seed = gen->current_seed();
  auto offset = static_cast<int64_t>(gen->philox_offset_per_thread()); // Note that old THCGeneratorState had offset as std::atomic<int64_t>
  memcpy(THByteTensor_data(rng_state) + states_size, &current_seed, seed_size);
  memcpy(THByteTensor_data(rng_state) + states_size + seed_size, &offset, offset_size);
}

extern "C" __host__ void THCRandom_setRNGState(at::Generator gen_, THByteTensor *rng_state)
{
  auto gen = at::check_generator<at::CUDAGenerator>(gen_);
  std::lock_guard<std::mutex> lock(gen->mutex_);
  static const size_t states_size = 200 * sizeof(4120); // this line is just here for BC reason
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = states_size + seed_size + offset_size;
  bool no_philox_seed = false;
  if (THByteTensor_nElement(rng_state) == total_size - offset_size) {
    no_philox_seed = true;
  }
  else {
    THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  }
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  uint64_t input_seed;
  memcpy(&input_seed, THByteTensor_data(rng_state) + states_size, seed_size);
  gen->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, THByteTensor_data(rng_state) + states_size + seed_size, offset_size);
  }
  gen->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateBoolType.h>
