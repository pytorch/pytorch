#pragma once

#include <ATen/core/MT19937RNGEngine.h>

/**
 * THGeneratorState is a POD class needed for memcpys
 * in torch.get_rng_state() and torch.set_rng_state().
 * It is a legacy class and even though it is replaced with
 * at::CPUGenerator, we need this class and some of its fields
 * to support backward compatibility on loading checkpoints.
 */
struct THGeneratorState {
  /* The initial seed. */
  uint64_t the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  uint64_t next;
  uint64_t state[at::MERSENNE_STATE_N]; /* the array for the state vector  */

  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
};

/**
 * THGeneratorStateNew is a POD class containing
 * new data introduced in at::CPUGenerator. It is used
 * as a helper for torch.get_rng_state() and torch.set_rng_state()
 * functions.
 */ 
struct THGeneratorStateNew {
  at::mt19937 engine;
  c10::optional<float> next_float_normal_sample;
  c10::optional<double> next_double_normal_sample;
};