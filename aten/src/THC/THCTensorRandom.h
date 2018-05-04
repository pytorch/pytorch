#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THCTensor.h"

#include "generic/THCTensorRandom.h"
#include "THCGenerateAllTypes.h"

typedef struct THCGenerator THCGenerator;

typedef struct THCRNGState {
  /* One generator per GPU */
  THCGenerator* gen;
  int num_devices;
} THCRNGState;

struct THCState;

THC_API void THCRandom_init(struct THCState *state, int num_devices, int current_device);
THC_API void THCRandom_shutdown(struct THCState *state);
THC_API uint64_t THCRandom_seed(struct THCState *state);
THC_API uint64_t THCRandom_seedAll(struct THCState *state);
THC_API void THCRandom_manualSeed(struct THCState *state, uint64_t the_seed_);
THC_API void THCRandom_manualSeedAll(struct THCState *state, uint64_t the_seed_);
THC_API uint64_t THCRandom_initialSeed(struct THCState *state);
THC_API void THCRandom_getRNGState(struct THCState *state, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(struct THCState *state, THByteTensor *rng_state);

THC_API struct curandStateMtgp32* THCRandom_generatorStates(struct THCState* state);

#endif
