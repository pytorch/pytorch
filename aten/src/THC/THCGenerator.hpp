#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <atomic>
#include <mutex>

typedef struct THCGeneratorState {
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params *kernel_params;
  int initf;
  uint64_t initial_seed;
  std::atomic<int64_t> philox_seed_offset;
} THCGeneratorState;

struct THCGenerator {
  std::mutex mutex; /* mutex for using this generator */
  THCGeneratorState state;
};
