#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THCTensor.h"

TH_API unsigned long THCRandom_seed();
TH_API void THCRandom_manualSeed(unsigned long the_seed_);
TH_API unsigned long THCRandom_initialSeed();
TH_API unsigned long THCRandom_random();
TH_API unsigned long THCRandom_random1(long b);
TH_API unsigned long THCRandom_random2(long a, long b);
TH_API double THCRandom_uniform(double a, double b);
TH_API double THCRandom_normal(double mean, double stdv);
TH_API double THCRandom_exponential(double lambda);
TH_API double THCRandom_cauchy(double median, double sigma);
TH_API double THCRandom_logNormal(double mean, double stdv);
TH_API int THCRandom_geometric(double p);
TH_API int THCRandom_bernoulli(double p);

TH_API void THCudaTensor_random(THCudaTensor *self);
TH_API void THCudaTensor_random1(THCudaTensor *self, long b);
TH_API void THCudaTensor_random2(THCudaTensor *self, long a, long b);
TH_API void THCudaTensor_geometric(THCudaTensor *self, double p);
TH_API void THCudaTensor_bernoulli(THCudaTensor *self, double p);
TH_API void THCudaTensor_uniform(THCudaTensor *self, double a, double b);
TH_API void THCudaTensor_normal(THCudaTensor *self, double mean, double stdv);
TH_API void THCudaTensor_exponential(THCudaTensor *self, double lambda);
TH_API void THCudaTensor_cauchy(THCudaTensor *self, double median, double sigma);
TH_API void THCudaTensor_logNormal(THCudaTensor *self, double mean, double stdv);

#endif
