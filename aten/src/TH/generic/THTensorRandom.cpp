#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.cpp"
#else

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cpuinfo.h>

void THTensor_(random)(THTensor *self, at::Generator *_generator)
{
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (uint8_t)(_generator->random64() % (UINT8_MAX + 1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int8_t)(_generator->random64() % (INT8_MAX + 1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int16_t)(_generator->random64() % (INT16_MAX + 1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int32_t)(_generator->random64() % (INT32_MAX + 1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (uint64_t)(_generator->random64() % (LONG_MAX + 1ULL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (float)(((uint32_t)(_generator->random64())) % ((1ULL << FLT_MANT_DIG) + 1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (double)(_generator->random64() % ((1ULL << DBL_MANT_DIG) + 1)););
#else
#error "Unknown type"
#endif

}

void THTensor_(clampedRandom)(THTensor *self, at::Generator *_generator, int64_t min, int64_t max) {
  THArgCheck(max > min, 2, "max must be greater than min, but got: min = %lld, max = %lld", min, max);
  uint64_t range = max - min;
#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    if (range >= 1ULL << 32) {
      TH_TENSOR_APPLY(scalar_t, self, *self_data = static_cast<scalar_t>(static_cast<int64_t>((_generator->random64() % range) + min));)
      return;
    }
#endif
    TH_TENSOR_APPLY(scalar_t, self, *self_data = static_cast<scalar_t>(static_cast<int64_t>((((uint32_t)(_generator->random64())) % range) + min));)
}

void THTensor_(cappedRandom)(THTensor *self, at::Generator *_generator, int64_t max) {
  THArgCheck(max > 0, 1, "max must be positive, but got: max = %lld", max);
  THTensor_(clampedRandom)(self, _generator, 0, max);
}

void THTensor_(geometric)(THTensor *self, at::Generator *_generator, double p)
{
  std::geometric_distribution<> geometric(p);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)geometric(_generator->getCPUEngine()););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined(TH_REAL_IS_FLOAT)
#define TH_REAL_MIN FLT_MIN
#elif defined(TH_REAL_IS_DOUBLE)
#define TH_REAL_MIN DBL_MIN
#endif

void THTensor_(uniform)(THTensor *self, at::Generator *_generator, double a, double b)
{
  #if defined(TH_REAL_IS_FLOAT)
  std::uniform_real_distribution<float> uniform((float)a, (float)b);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)uniform(_generator->getCPUEngine()););
  #else
  std::uniform_real_distribution<double> uniform(a, b);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)uniform(_generator->getCPUEngine()););
  #endif
}

void THTensor_(normal)(THTensor *self, at::Generator *_generator, double mean, double stddev)
{
  const int64_t size = THTensor_(numel)(self);
  if (size >= 16 && THTensor_(isContiguous)(self)) {
    THVector_(normal_fill)(THStorage_(data)(THTensor_getStoragePtr(self)) + self->storage_offset(), size, _generator, mean, stddev);
  } else {
    std::normal_distribution<double> normal{mean, stddev};
    TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)normal(_generator->getCPUEngine()););
  }
}

void THTensor_(normal_means)(THTensor *self, at::Generator *gen, THTensor *means, double stddev)
{
  THTensor_(resizeAs)(self, means);
  THTensor_(normal)(self, gen, 0, stddev);
  THTensor_(cadd)(self, self, 1, means);
}

void THTensor_(normal_stddevs)(THTensor *self, at::Generator *gen, double mean, THTensor *stddevs)
{
  THTensor_(resizeAs)(self, stddevs);
  THTensor_(normal)(self, gen, 0, 1);
  THTensor_(cmul)(self, self, stddevs);
  THTensor_(add)(self, self, mean);
}

void THTensor_(normal_means_stddevs)(THTensor *self, at::Generator *gen, THTensor *means, THTensor *stddevs)
{
  THTensor_(resizeAs)(self, means);
  THTensor_(normal)(self, gen, 0, 1);
  THTensor_(cmul)(self, self, stddevs);
  THTensor_(cadd)(self, self, 1, means);
}

void THTensor_(exponential)(THTensor *self, at::Generator *_generator, double lambda)
{
  std::exponential_distribution<double> exponential(lambda);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)exponential(_generator->getCPUEngine()););
}

#undef TH_REAL_MIN

void THTensor_(cauchy)(THTensor *self, at::Generator *_generator, double median, double sigma)
{
  std::cauchy_distribution<double> cauchy(median, sigma);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)cauchy(_generator->getCPUEngine()););
}

void THTensor_(logNormal)(THTensor *self, at::Generator *_generator, double mean, double stdv)
{
  std::lognormal_distribution<double> logNormal(mean, stdv);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)logNormal(_generator->getCPUEngine()););
}

void THTensor_(multinomialAliasSetup)(THTensor *probs, THLongTensor *J, THTensor *q)
{
  int64_t inputsize = THTensor_(nElement)(probs);
  int64_t i = 0;
  THLongTensor *smaller = THLongTensor_newWithSize1d(inputsize);
  THLongTensor *larger = THLongTensor_newWithSize1d(inputsize);
  int64_t small_c = 0;
  int64_t large_c = 0;
  THLongTensor_resize1d(J, inputsize);
  THTensor_(resize1d)(q, inputsize);
  scalar_t *q_data = q->data<scalar_t>();
  int64_t *J_data = THLongTensor_data(J);

  for (i = 0; i < inputsize; i++)
    {
      THLongTensor_fastSet1d(J, i, 0L);
      scalar_t val = THTensor_(fastGet1d)(probs, i);
      THTensor_(fastSet1d)(q, i, inputsize*val);

      if (inputsize * val < 1.0)
        {
          THLongTensor_fastSet1d(smaller, small_c, i);
          small_c += 1;
        }
      else
        {
          THLongTensor_fastSet1d(larger, large_c, i);
          large_c += 1;
        }
    }

  // Loop through and create little binary mixtures that
  // appropriately allocate the larger outcomes over the
  // overall uniform mixture.
  int64_t large, small;
  while (small_c > 0 && large_c > 0)
    {
      large = THLongTensor_fastGet1d(larger, large_c-1);
      small = THLongTensor_fastGet1d(smaller, small_c-1);

      THLongTensor_fastSet1d(J, small, large);
      q_data[large * q->stride(0)] -= 1.0 - THTensor_(fastGet1d)(q, small);

      if(q_data[large * q->stride(0)] < 1.0)
        {
          THLongTensor_fastSet1d(smaller, small_c-1, large);
          large_c -= 1;
        }
      else
        {
          THLongTensor_fastSet1d(larger, large_c-1, large);
          small_c -= 1;
        }
    }

  scalar_t q_min = THTensor_(fastGet1d)(q, inputsize-1);
  scalar_t q_max = q_min;
  scalar_t q_temp;
  for (i=0; i < inputsize; i++)
    {
      q_temp = THTensor_(fastGet1d)(q, i);
      if (q_temp < q_min)
        q_min = q_temp;
      else if (q_temp > q_max)
        q_max = q_temp;
    }
  THArgCheckWithCleanup((q_min > 0),
                        THCleanup(THLongTensor_free(smaller); THLongTensor_free(larger);), 2,
                        "q_min is less than 0");

  if (q_max > 1)
    {
      for (i=0; i < inputsize; i++)
        {
          q_data[i*q->stride(0)] /= q_max;
        }
    }
  for (i=0; i < inputsize; i++)
    {
      // sometimes an large index isn't added to J.
      // fix it by making the probability 1 so that J isn't indexed.
      if(J_data[i] <= 0)
        q_data[i] = 1.0;
    }
  THLongTensor_free(smaller);
  THLongTensor_free(larger);
}
void THTensor_(multinomialAliasDraw)(THLongTensor *self, at::Generator *_generator, THLongTensor *J, THTensor *q)
{
  int64_t K = THLongTensor_nElement(J);
  int64_t output_nelem = THLongTensor_nElement(self);
  int64_t i = 0, _mask=0;
  scalar_t _q;
  int64_t rand_ind, sample_idx, J_sample;

  for (i=0; i < output_nelem; i++)
    {
      std::uniform_real_distribution<double> uniform(0, K);
      rand_ind = uniform(_generator->getCPUEngine());

      _q = THTensor_(fastGet1d)(q, rand_ind);
      std::bernoulli_distribution bernoulli(_q);
      _mask = static_cast<int64_t>(bernoulli(_generator->getCPUEngine()));

      J_sample = THLongTensor_fastGet1d(J, rand_ind);

      sample_idx = J_sample*(1 -_mask) + (rand_ind+1L) * _mask;

      THLongTensor_fastSet1d(self, i, sample_idx-1L);
    }
}
void THTensor_(multinomial)(THLongTensor *self, at::Generator *_generator, THTensor *prob_dist, int n_sample, int with_replacement)
{
  int64_t start_dim = THTensor_(nDimensionLegacyAll)(prob_dist);
  int64_t n_dist;
  int64_t n_categories;
  THDoubleTensor* cum_dist;
  int64_t i,j,k;

  if (start_dim == 1)
  {
    THTensor_(unsqueeze1d)(prob_dist, prob_dist, 0);
  }

  n_dist = THTensor_(size)(prob_dist, 0);
  n_categories = THTensor_(size)(prob_dist, 1);

  THArgCheckWithCleanup(n_sample > 0,
    THCleanup(if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
    2,
    "cannot sample n_sample <= 0 samples");

  if (!with_replacement)
  {
    THArgCheckWithCleanup((!with_replacement) && (n_sample <= n_categories),
      THCleanup(if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
      2,
      "cannot sample n_sample > prob_dist.size(1) samples without replacement");
  }

  /* cumulative probability distribution vector */
  cum_dist = THDoubleTensor_newWithSize1d(n_categories);

  /* will contain multinomial samples (category indices to be returned) */
  THLongTensor_resize2d(self, n_dist , n_sample);

  for (i=0; i<n_dist; i++)
  {
    /* Get normalized cumulative distribution from prob distribution */
    double sum = 0;
    double val;
    int n_zeros = 0;
    for (j=0; j<n_categories; j++)
    {
      val = THStorage_(get)( \
        THTensor_getStoragePtr(prob_dist), \
        prob_dist->storage_offset()+i*prob_dist->stride(0)+j*prob_dist->stride(1) \
      );
      THArgCheckWithCleanup((val >= 0),
                            THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                            2,
                            "invalid multinomial distribution (encountering probability entry < 0)");
      THArgCheckWithCleanup((std::isfinite(val)),
                            THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                            2,
                            "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
      sum += val;
      if (val == 0) {
        n_zeros += 1;
      }
      THDoubleStorage_set(
        THTensor_getStoragePtr(cum_dist), \
        cum_dist->storage_offset()+j*cum_dist->stride(0), \
        sum \
      );
    }
    THArgCheckWithCleanup((sum > 0),
                          THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                          2,
                          "invalid multinomial distribution (sum of probabilities <= 0)");
    THArgCheckWithCleanup((with_replacement || (n_categories - n_zeros >= n_sample)),
                          THCleanup(THDoubleTensor_free(cum_dist); if (start_dim == 1) THTensor_(squeeze1d)(prob_dist, prob_dist, 0);),
                          2,
                          "invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)");
    /* normalize cumulative probability distribution so that last val is 1
    i.e. doesn't assume original prob_dist row sums to one */
    if ( (sum > 0) || ( ( sum < 1.00001) && (sum > 0.99999) ) )
    {
      for (j=0; j<n_categories; j++)
      {
        THDoubleTensor_data(cum_dist)[j*cum_dist->stride(0)] /= sum;
      }
    }

    for (j=0; j<n_sample; j++)
    {
      /* sample a probability mass from a uniform distribution */
      std::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(_generator->getCPUEngine());
      /* Do a binary search for the slot in which the prob falls
      ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
      int left_pointer = 0;
      int right_pointer = n_categories;
      int mid_pointer;
      double cum_prob;
      int sample_idx;
      /* Make sure the last cumulative distribution bucket sums to 1 */
      THDoubleTensor_data(cum_dist)[(n_categories-1)*cum_dist->stride(0)] = 1;

      while(right_pointer - left_pointer > 0)
      {
          mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
          cum_prob = THDoubleStorage_get( \
            THTensor_getStoragePtr(cum_dist), \
            cum_dist->storage_offset()+mid_pointer*cum_dist->stride(0) \
          );
          if (cum_prob < uniform_sample)
          {
            left_pointer = mid_pointer + 1;
          }
          else
          {
            right_pointer = mid_pointer;
          }
      }
      sample_idx = left_pointer;

       /* store in result tensor (will be incremented for lua compat by wrapper) */
      THLongStorage_set( \
        THTensor_getStoragePtr(self), \
        self->storage_offset()+i*self->stride(0)+j*self->stride(1), \
        sample_idx \
      );

      /* Once a sample is drawn, it cannot be drawn again. ie sample without replacement */
      if (!with_replacement && j < n_sample - 1)
      {
        /* update cumulative distribution so that sample cannot be drawn again */
        double diff;
        double new_val = 0;
        double sum;

        if (sample_idx != 0)
        {
          new_val = THDoubleStorage_get( \
            THTensor_getStoragePtr(cum_dist), \
            cum_dist->storage_offset()+(sample_idx-1)*cum_dist->stride(0) \
          );
        }
        /* marginal cumulative mass (i.e. original probability) of sample */
        diff = THDoubleStorage_get( \
          THTensor_getStoragePtr(cum_dist), \
          cum_dist->storage_offset()+sample_idx*cum_dist->stride(0) \
        ) - new_val;
        /* new sum of marginals is not one anymore... */
        sum = 1.0 - diff;
        for (k=0; k<n_categories; k++)
        {
          new_val = THDoubleStorage_get( \
            THTensor_getStoragePtr(cum_dist), \
            cum_dist->storage_offset()+k*cum_dist->stride(0) \
          );
          if (k >= sample_idx)
          {
            /* remove sampled probability mass from later cumulative probabilities */
            new_val -= diff;
          }
          /* make total marginals sum to one */
          new_val /= sum;
          THDoubleStorage_set( \
            THTensor_getStoragePtr(cum_dist), \
            cum_dist->storage_offset()+k*cum_dist->stride(0), \
            new_val \
          );
        }
      }
    }
  }

  THDoubleTensor_free(cum_dist);

  if (start_dim == 1)
  {
    THLongTensor_resize1d(self, n_sample);
    THTensor_(squeeze1d)(prob_dist, prob_dist, 0);
  }
}
#endif

#if defined(TH_REAL_IS_BYTE)
void THTensor_(getRNGState)(at::Generator *_generator, THTensor *self)
{
  static const size_t size = sizeof(at::GeneratorState);
  at::GeneratorState* rng_state;
  THTensor_(resize1d)(self, size);
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (at::GeneratorState*)self->data<scalar_t>();
  *rng_state = *(_generator->getState());
}

void THTensor_(setRNGState)(at::Generator *_generator, THTensor *self)
{
  static const size_t size = sizeof(at::GeneratorState);
  at::GeneratorState* rng_state;
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  rng_state = (at::GeneratorState*)self->data<scalar_t>();
  int64_t is_valid = 0;
  if (_generator->getState()->device_type == rng_state->device_type) {
    is_valid = 1;
  }
  THArgCheck(is_valid, 1, "Invalid RNG state");
  _generator->setState(rng_state);
}
#endif
#endif
