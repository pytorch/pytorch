#include "THCTensorRandom.h"
#include "THCGenerator.h"

#include <random>
#include <curand.h>


void initializeGenerator(THCState *state, THCGenerator* gen);
void createGeneratorState(THCGenerator* gen, uint64_t seed);


/* Frees memory allocated during setup. */
void destroyGenerator(THCState *state, THCGenerator* gen)
{
  std::lock_guard<std::mutex> lock(gen->mutex);
  if (gen->state.gen_states)
  {
    THCudaCheck(THCudaFree(state, gen->state.gen_states));
    gen->state.gen_states = NULL;
  }
  if (gen->state.kernel_params)
  {
    THCudaCheck(THCudaFree(state, gen->state.kernel_params));
    gen->state.kernel_params = NULL;
  }
}

static uint64_t createSeed(std::random_device& rd)
{
  // limit to 53 bits to ensure unique representation in double
  uint64_t seed = (((uint64_t)rd()) << 32) + rd();
  return seed & 0x1FFFFFFFFFFFFF;
}

/* Initialize generator array (must be called before any other function) */
void THCRandom_init(THCState* state, int devices, int current_device)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  rng_state->num_devices = devices;
  rng_state->gen = (THCGenerator*)malloc(rng_state->num_devices * sizeof(THCGenerator));
  std::random_device rd;
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    new (&rng_state->gen[i].mutex) std::mutex();
    rng_state->gen[i].state.initf = 0;
    rng_state->gen[i].state.initial_seed = createSeed(rd);
    rng_state->gen[i].state.philox_seed_offset = 0;
    rng_state->gen[i].state.gen_states = NULL;
    rng_state->gen[i].state.kernel_params = NULL;
  }
}

/* Destroy generators and free memory */
void THCRandom_shutdown(THCState* state)
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

/* Get the generator for the current device, but does not initialize the state */
static THCGenerator* THCRandom_rawGenerator(THCState* state)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  int device;
  THCudaCheck(cudaGetDevice(&device));
  if (device >= rng_state->num_devices) THError("Invalid device index.");
  return &rng_state->gen[device];
}

/* Get the generator for the current device and initializes it if necessary */
THCGenerator* THCRandom_getGenerator(THCState* state)
{
  THCGenerator* gen = THCRandom_rawGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);
  if (gen->state.initf == 0)
  {
    initializeGenerator(state, gen);
    createGeneratorState(gen, gen->state.initial_seed);
    gen->state.initf = 1;
  }
  return gen;
}

struct curandStateMtgp32* THCRandom_generatorStates(struct THCState* state)
{
  THCGenerator* gen = THCRandom_getGenerator(state);
  return gen->state.gen_states;
}

/* Random seed */
uint64_t THCRandom_seed(THCState* state)
{
  std::random_device rd;
  uint64_t s = createSeed(rd);
  THCRandom_manualSeed(state, s);
  return s;
}

uint64_t THCRandom_seedAll(THCState* state)
{
  std::random_device rd;
  uint64_t s = createSeed(rd);
  THCRandom_manualSeedAll(state, s);
  return s;
}

/* Manually set the seed */
void THCRandom_manualSeed(THCState* state, uint64_t seed)
{
  THCGenerator* gen = THCRandom_rawGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);
  gen->state.initial_seed = seed;
  if (gen->state.initf) {
    createGeneratorState(gen, seed);
  }
}

void THCRandom_manualSeedAll(THCState* state, uint64_t seed)
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
uint64_t THCRandom_initialSeed(THCState* state)
{
  THCGenerator* gen = THCRandom_getGenerator(state);
  return gen->state.initial_seed;
}
