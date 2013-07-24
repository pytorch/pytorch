#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THCTensor.h"

TH_API unsigned long THCRandom_seed();
TH_API void THCRandom_manualSeed(unsigned long the_seed_);
TH_API unsigned long THCRandom_initialSeed();

// TH_API void THCudaTensor_random(THCudaTensor *self);
// TH_API void THCudaTensor_random1(THCudaTensor *self, long b);
// TH_API void THCudaTensor_random2(THCudaTensor *self, long a, long b);
TH_API void THCudaTensor_geometric(THCudaTensor *self, double p);
TH_API void THCudaTensor_bernoulli(THCudaTensor *self, double p);
TH_API void THCudaTensor_uniform(THCudaTensor *self, double a, double b);
TH_API void THCudaTensor_normal(THCudaTensor *self, double mean, double stdv);
TH_API void THCudaTensor_exponential(THCudaTensor *self, double lambda);
TH_API void THCudaTensor_cauchy(THCudaTensor *self, double median, double sigma);
TH_API void THCudaTensor_logNormal(THCudaTensor *self, double mean, double stdv);

#endif
