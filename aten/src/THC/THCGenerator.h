#ifndef THC_GENERATOR_INC
#define THC_GENERATOR_INC

#include <mutex>

typedef struct THCGeneratorState {
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params *kernel_params;
  int initf;
  uint64_t initial_seed;
  int64_t philox_seed_offset;
} THCGeneratorState;

struct THCGenerator {
  std::mutex mutex; /* mutex for using this generator */
  THCGeneratorState state;
};

#endif
