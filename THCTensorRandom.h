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

typedef struct THCudaRNGState {
  /* One generator per GPU */
  Generator* gen;
  Generator* current_gen;
  int num_devices;
} THCudaRNGState;

THC_API void THCRandom_init(THCudaRNGState* state, int num_devices, int current_device);
THC_API void THCRandom_shutdown(THCudaRNGState* state);
THC_API void THCRandom_setGenerator(THCudaRNGState* state, int device);
THC_API unsigned long THCRandom_seed(THCudaRNGState* state);
THC_API unsigned long THCRandom_seedAll(THCudaRNGState* state);
THC_API void THCRandom_manualSeed(THCudaRNGState* state, unsigned long the_seed_);
THC_API void THCRandom_manualSeedAll(THCudaRNGState* state, unsigned long the_seed_);
THC_API unsigned long THCRandom_initialSeed(THCudaRNGState* state);
THC_API void THCRandom_getRNGState(THCudaRNGState* state, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(THCudaRNGState* state, THByteTensor *rng_state);
THC_API void THCudaTensor_geometric(THCudaRNGState* state, THCudaTensor *self, double p);
THC_API void THCudaTensor_bernoulli(THCudaRNGState* state, THCudaTensor *self, double p);
THC_API void THCudaTensor_uniform(THCudaRNGState* state, THCudaTensor *self, double a, double b);
THC_API void THCudaTensor_normal(THCudaRNGState* state, THCudaTensor *self, double mean, double stdv);
THC_API void THCudaTensor_exponential(THCudaRNGState* state, THCudaTensor *self, double lambda);
THC_API void THCudaTensor_cauchy(THCudaRNGState* state, THCudaTensor *self, double median, double sigma);
THC_API void THCudaTensor_logNormal(THCudaRNGState* state, THCudaTensor *self, double mean, double stdv);

#endif
