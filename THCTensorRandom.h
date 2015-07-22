#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THCTensor.h"

/* Generator */
typedef struct _Generator {
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params *kernel_params;
  int initf;
  unsigned long initial_seed;
} Generator;

typedef struct THCRNGState {
  /* One generator per GPU */
  Generator* gen;
  Generator* current_gen;
  int num_devices;
} THCRNGState;

struct THCState;

THC_API void THCRandom_init(struct THCState *state, int num_devices, int current_device);
THC_API void THCRandom_shutdown(struct THCState *state);
THC_API void THCRandom_setGenerator(struct THCState *state, int device);
THC_API unsigned long THCRandom_seed(struct THCState *state);
THC_API unsigned long THCRandom_seedAll(struct THCState *state);
THC_API void THCRandom_manualSeed(struct THCState *state, unsigned long the_seed_);
THC_API void THCRandom_manualSeedAll(struct THCState *state, unsigned long the_seed_);
THC_API unsigned long THCRandom_initialSeed(struct THCState *state);
THC_API void THCRandom_getRNGState(struct THCState *state, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(struct THCState *state, THByteTensor *rng_state);
THC_API void THCudaTensor_geometric(struct THCState *state, THCudaTensor *self, double p);
THC_API void THCudaTensor_bernoulli(struct THCState *state, THCudaTensor *self, double p);
THC_API void THCudaTensor_uniform(struct THCState *state, THCudaTensor *self, double a, double b);
THC_API void THCudaTensor_normal(struct THCState *state, THCudaTensor *self, double mean, double stdv);
THC_API void THCudaTensor_exponential(struct THCState *state, THCudaTensor *self, double lambda);
THC_API void THCudaTensor_cauchy(struct THCState *state, THCudaTensor *self, double median, double sigma);
THC_API void THCudaTensor_logNormal(struct THCState *state, THCudaTensor *self, double mean, double stdv);

THC_API void THCudaTensor_multinomial(struct THCState *state, THCudaTensor *self, THCudaTensor *prob_dist, int n_sample, int with_replacement);

#endif
