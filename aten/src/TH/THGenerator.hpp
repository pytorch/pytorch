#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <mutex>

struct THGeneratorState {
  /* The initial seed. */
  uint64_t the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  uint64_t next;
  uint64_t state[_MERSENNE_STATE_N]; /* the array for the state vector  */

  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
};

/* A THGenerator contains all the state required for a single random number stream */
struct THGenerator {
  std::mutex mutex; /* mutex for using this generator */
  THGeneratorState gen_state;
};
