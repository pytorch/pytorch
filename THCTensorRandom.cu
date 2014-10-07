#include "THCTensorRandom.h"
#include "THCGeneral.h"

#include <thrust/functional.h>
#include <curand.h>

/* Generator */
typedef struct _Generator {
  curandGenerator_t gen;
  int initf;
  unsigned long initial_seed;
} Generator;

/* One generator per GPU */
static Generator* gen = NULL;
static Generator* current_gen = NULL;
static int num_devices = -1;

/* Initialize generator array (must be called before any other function) */
__host__ void THCRandom_init(int devices, int current_device)
{
  num_devices = devices;
  if (gen) free(gen);
  gen = (Generator*)malloc(num_devices * sizeof(Generator));
  for (int i = 0; i < num_devices; ++i)
  {
    gen[i].initf = 0;
    gen[i].initial_seed = 0;
  }
  current_gen = &gen[current_device];
  // Initialize the generator for the current device. Other generators will be
  // initialized on-demand in THCRandom_setGenerator.
  THCRandom_seed();
}

/* Destroy generators and free memory */
__host__ void THCRandom_shutdown()
{
  if (gen == NULL) return;
  for (int i = 0; i < num_devices; ++i)
  {
    curandDestroyGenerator(gen[i].gen);
  }
  free(gen);
  gen = NULL;
  current_gen = NULL;
}

/* Set the generator for the current device */
__host__ void THCRandom_setGenerator(int device)
{
  if (device >= num_devices) THError("Invalid device index.");
  current_gen = &gen[device];
  if (current_gen->initf == 0)
  {
    THCRandom_seed();
  }
}

/* Random seed */
__host__ unsigned long THCRandom_seed()
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeed(s);
  return s;
}

/* Manually set the seed */
__host__ void THCRandom_manualSeed(unsigned long seed)
{
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  current_gen->initial_seed = seed;
  if (current_gen->initf == 1) curandDestroyGenerator(current_gen->gen);
  curandCreateGenerator(&current_gen->gen, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(current_gen->gen, seed);
  current_gen->initf = 1;
}

/* Get the initial seed */
__host__ unsigned long THCRandom_initialSeed()
{
  return current_gen->initial_seed;
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

THC_API void THCudaTensor_uniform(THCudaTensor *self_, double a, double b) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);

  curandGenerateUniform(current_gen->gen, data, size);

  if ((a != 0) || (b != 1)) {
      THCudaTensor_mul(self, b-a);
      THCudaTensor_add(self, a);
  }

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_bernoulli(THCudaTensor *self_, double p) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(current_gen->gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, bernoulli_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_normal(THCudaTensor *self_, double mean, double stdv) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);

  curandGenerateNormal(current_gen->gen, data, size, mean, stdv);

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_logNormal(THCudaTensor *self_, double mean, double stdv) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  
  curandGenerateLogNormal(current_gen->gen, data, size, mean, stdv);

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_geometric(THCudaTensor *self_, double p) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(current_gen->gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, geometric_functor(p));

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_exponential(THCudaTensor *self_, double lambda) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(current_gen->gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, exponential_functor(lambda));

  THCudaTensor_freeCopyTo(self, self_);
};

THC_API void THCudaTensor_cauchy(THCudaTensor *self_, double median, double sigma) {
  if (current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  float *data = THCudaTensor_data(self);
  thrust::device_ptr<float> tdata(data);
  
  curandGenerateUniform(current_gen->gen, data, size);
  
  thrust::transform(tdata, tdata+size, tdata, cauchy_functor(median, sigma));

  THCudaTensor_freeCopyTo(self, self_);
};
