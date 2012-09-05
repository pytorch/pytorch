#include "THCTensorRandom.h"
#include "THCGeneral.h"

#include <thrust/random.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>

/* The initial seed. */
__device__ static int initf = 0;
__device__ static unsigned long the_initial_seed = 0;
__device__ static unsigned long step = 0;

/* Seeds */
__host__ unsigned long THCRandom_seed()
{
  unsigned long s = (unsigned long)1; // TODO: this should be random
  THCRandom_manualSeed(s);
  return s;
}

__host__ void THCRandom_manualSeed(unsigned long the_seed_)
{
  the_initial_seed = the_seed_;
  initf = 1;
}

__host__ unsigned long THCRandom_initialSeed()
{
  if(initf == 0) THCRandom_seed();
  return the_initial_seed;
}

__host__ __device__ unsigned long THCRandom_random()
{
  thrust::default_random_engine rng(the_initial_seed); rng.discard(step++);
  thrust::uniform_int_distribution<unsigned long> ufm(0,(((unsigned long)1)<<31)-1);
  return ufm(rng);
}

/* generates a random number on [0,1)-double-interval */
__host__ __device__ static double __uniform__()
{
  thrust::default_random_engine rng(the_initial_seed); rng.discard(step++);
  thrust::uniform_real_distribution<double> ufm(0,1);
  return ufm(rng);
}

__host__ __device__ unsigned long THCRandom_random1(long b)
{
  //THArgCheck(b > 0, 1, "upper bound must be strictly positive");
  return(THCRandom_random() % b + 1);
}

__host__ __device__ unsigned long THCRandom_random2(long a, long b)
{
  //THArgCheck(b >= a, 2, "upper bound must be larger than lower bound");
  return((THCRandom_random() % (b+1-a)) + a);
}

__host__ __device__ double THCRandom_uniform(double a, double b)
{
  return(__uniform__() * (b - a) + a);
}

__host__ __device__ double THCRandom_normal(double mean, double stdv)
{
  //THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");
  thrust::default_random_engine rng(the_initial_seed); rng.discard(step++);
  thrust::random::experimental::normal_distribution<double> normal(mean,stdv);
  return normal(rng);
}

__host__ __device__ double THCRandom_exponential(double lambda)
{
  return(-1. / lambda * log(1-__uniform__()));
}

__host__ __device__ double THCRandom_cauchy(double median, double sigma)
{
  return(median + sigma * tan(M_PI*(__uniform__()-0.5)));
}

__host__ __device__ double THCRandom_logNormal(double mean, double stdv)
{
  //THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");
  double zm = mean*mean;
  double zs = stdv*stdv;
  thrust::default_random_engine rng(the_initial_seed); rng.discard(step++);
  thrust::random::experimental::normal_distribution<double> normal(log(zm/sqrt(zs + zm)), sqrt(log(zs/zm+1)));
  return exp(normal(rng));
}

__host__ __device__ int THCRandom_geometric(double p)
{
  //THArgCheck(p > 0 && p < 1, 1, "must be > 0 and < 1");
  return((int)(log(1-__uniform__()) / log(p)) + 1);
}

__host__ __device__ int THCRandom_bernoulli(double p)
{
  //THArgCheck(p > 0 && p < 1, 1, "must be > 0 and < 1");
  return(__uniform__() <= p);
}

struct random_functor
{
  random_functor() {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_int_distribution<unsigned long> ufm(0,(((unsigned long)1)<<31)-1);
    unsigned long r = ufm(rng);
    return (float)(r % ((1UL << FLT_MANT_DIG)+1));
  }
};

TH_API void THCudaTensor_random(THCudaTensor *self_) { 
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, random_functor());

  THCudaTensor_freeCopyTo(self, self_);
};

struct random1_functor
{
  const long b;

  random1_functor(long b_) : b(b_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_int_distribution<unsigned long> ufm(0,(((unsigned long)1)<<31)-1);
    unsigned long r = ufm(rng);
    return (float)(r % b + 1);
  }
};

TH_API void THCudaTensor_random1(THCudaTensor *self_, long b) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, random1_functor(b));

  THCudaTensor_freeCopyTo(self, self_);
};

struct random2_functor
{
  const long a,b;

  random2_functor(long a_, long b_) : a(a_), b(b_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_int_distribution<unsigned long> ufm(0,(((unsigned long)1)<<31)-1);
    unsigned long r = ufm(rng);
    return (float)((r % (b+1-a)) + a);
  }
};

TH_API void THCudaTensor_random2(THCudaTensor *self_, long a, long b) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, random2_functor(a,b));

  THCudaTensor_freeCopyTo(self, self_);
};

struct bernoulli_functor
{
  const double p;

  bernoulli_functor(double p_) : p(p_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_real_distribution<float> uniform(0,1);
    return (float)(uniform(rng) <= p);
  }
};

TH_API void THCudaTensor_bernoulli(THCudaTensor *self_, double p) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, bernoulli_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

struct uniform_functor
{
  const double a,b;

  uniform_functor(double a_, double b_) : a(a_),b(b_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_real_distribution<float> uniform(a,b);
    return uniform(rng);
  }
};

TH_API void THCudaTensor_uniform(THCudaTensor *self_, double a, double b) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, uniform_functor(a,b));

  THCudaTensor_freeCopyTo(self, self_);
};

struct normal_functor
{
  const double mean,stdv;

  normal_functor(double mean_, double stdv_)
    : mean(mean_),stdv(stdv_) {}

  __host__ __device__ 
  float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::random::experimental::normal_distribution<float> normal(mean,stdv);
    return normal(rng);
  }
};

TH_API void THCudaTensor_normal(THCudaTensor *self_, double mean, double stdv) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, normal_functor(mean,stdv));

  THCudaTensor_freeCopyTo(self, self_);
};

struct geometric_functor
{
  const double p;

  geometric_functor(double p_) : p(p_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_real_distribution<float> uniform(0,1);
    float u = uniform(rng);
    return (float)((log(1-u) / log(p)) + 1);
  }
};

TH_API void THCudaTensor_geometric(THCudaTensor *self_, double p) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, geometric_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

struct exponential_functor
{
  const double lambda;

  exponential_functor(double lambda_) : lambda(lambda_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_real_distribution<float> uniform(0,1);
    float u = uniform(rng);
    return (float)(-1. / lambda * log(1-u));
  }
};

TH_API void THCudaTensor_exponential(THCudaTensor *self_, double lambda) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, exponential_functor(lambda));

  THCudaTensor_freeCopyTo(self, self_);
};

struct cauchy_functor
{
  const double median,sigma;

  cauchy_functor(double median_, double sigma_) : median(median_),sigma(sigma_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::uniform_real_distribution<float> uniform(0,1);
    float u = uniform(rng);
    return (float)(median + sigma * tan(M_PI*(u-0.5)));
  }
};

TH_API void THCudaTensor_cauchy(THCudaTensor *self_, double median, double sigma) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, cauchy_functor(median, sigma));

  THCudaTensor_freeCopyTo(self, self_);
};

struct logNormal_functor
{
  const double mean,stdv;

  logNormal_functor(double mean_, double stdv_) : mean(mean_),stdv(stdv_) {}

  __host__ __device__ float operator()(const float& x) const
  {
    double zm = mean*mean;
    double zs = stdv*stdv;
    thrust::default_random_engine rng(the_initial_seed); rng.discard(x);
    thrust::random::experimental::normal_distribution<double> normal(log(zm/sqrt(zs + zm)), sqrt(log(zs/zm+1)));
    return exp(normal(rng));
  }
};

TH_API void THCudaTensor_logNormal(THCudaTensor *self_, double mean, double stdv) {
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::sequence(self_data, self_data+size, step+=size);
  thrust::transform(self_data, self_data+size, self_data, logNormal_functor(mean, stdv));

  THCudaTensor_freeCopyTo(self, self_);
};
