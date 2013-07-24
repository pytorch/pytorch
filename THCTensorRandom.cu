#include "THCTensorRandom.h"
#include "THCGeneral.h"

#include <thrust/functional.h>
#include <curand.h>

/* Generator */
static curandGenerator_t gen;

/* Initial seed */
static int initf = 0;
static unsigned long initial_seed = 0;

/* Random seed (this must be called once) */
__host__ unsigned long THCRandom_seed()
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeed(s);
  return s;
}

/* Manually set the seed */
__host__ void THCRandom_manualSeed(unsigned long seed)
{
  initial_seed = seed;
  if (initf == 1) curandDestroyGenerator(gen);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(gen, initial_seed);
  initf = 1;
}

/* Get the initial seed */
__host__ unsigned long THCRandom_initialSeed()
{
  return initial_seed;
}

/* The following functors are use to modify uniform distributions  */
struct bernoulli_functor
{
  const double p;
  bernoulli_functor(double p_) : p(p_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)(x <= p);
  }
};

struct geometric_functor
{
  const double p;
  geometric_functor(double p_) : p(p_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)((log(1-x) / log(p)) + 1);
  }
};

struct exponential_functor
{
  const double lambda;
  exponential_functor(double lambda_) : lambda(lambda_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)(-1. / lambda * log(1-x));
  }
};

struct cauchy_functor
{
  const double median,sigma;
  cauchy_functor(double median_, double sigma_) : median(median_),sigma(sigma_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    return (float)(median + sigma * tan(M_PI*(x-0.5)));
  }
};

TH_API void THCudaTensor_uniform(THCudaTensor *self_, double a, double b) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);

  curandGenerateUniform(gen, data, size);

  if ((a != 0) || (b != 1)) {
      THCudaTensor_mul(self, b-a);
      THCudaTensor_add(self, a);
  }

  THCudaTensor_freeCopyTo(self, self_);
};

TH_API void THCudaTensor_bernoulli(THCudaTensor *self_, double p) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, bernoulli_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

TH_API void THCudaTensor_normal(THCudaTensor *self_, double mean, double stdv) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);

  curandGenerateNormal(gen, data, size, mean, stdv);

  THCudaTensor_freeCopyTo(self, self_);
};

TH_API void THCudaTensor_logNormal(THCudaTensor *self_, double mean, double stdv) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  
  curandGenerateLogNormal(gen, data, size, mean, stdv);

  THCudaTensor_freeCopyTo(self, self_);
};

TH_API void THCudaTensor_geometric(THCudaTensor *self_, double p) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, geometric_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

TH_API void THCudaTensor_exponential(THCudaTensor *self_, double lambda) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, exponential_functor(lambda));

  THCudaTensor_freeCopyTo(self, self_);
};

TH_API void THCudaTensor_cauchy(THCudaTensor *self_, double median, double sigma) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, cauchy_functor(median, sigma));

  THCudaTensor_freeCopyTo(self, self_);
};
