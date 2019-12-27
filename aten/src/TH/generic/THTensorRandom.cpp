#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorRandom.cpp"
#else

#include <cmath>

#include <cpuinfo.h>
#include <array>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <ATen/Utils.h>

void THTensor_(random)(THTensor *self, at::Generator *_generator)
{
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (uint8_t)(gen->random() % (UINT8_MAX + 1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int8_t)(gen->random() % (INT8_MAX + 1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int16_t)(gen->random() % (INT16_MAX + 1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (int32_t)(gen->random() % (INT32_MAX + 1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (uint64_t)(gen->random64() % (LONG_MAX + 1ULL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (float)(gen->random() % ((1ULL << FLT_MANT_DIG) + 1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (double)(gen->random64() % ((1ULL << DBL_MANT_DIG) + 1)););
#elif defined(TH_REAL_IS_BOOL)
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (bool)(gen->random() % 2););
#else
#error "Unknown type"
#endif

}

void THTensor_(clampedRandom)(THTensor *self, int64_t min, int64_t max, at::Generator *_generator) {
  THArgCheck(max > min, 2, "max must be greater than min, but got: min = %lld, max = %lld", min, max);
  uint64_t range = max - min;
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    if (range >= 1ULL << 32) {
      TH_TENSOR_APPLY(scalar_t, self, *self_data = static_cast<scalar_t>(static_cast<int64_t>((gen->random64() % range) + min));)
      return;
    }
#endif
    TH_TENSOR_APPLY(scalar_t, self, *self_data = static_cast<scalar_t>(static_cast<int64_t>((gen->random() % range) + min));)
}

void THTensor_(cappedRandom)(THTensor *self, int64_t max, at::Generator *_generator) {
  THArgCheck(max > 0, 1, "max must be positive, but got: max = %lld", max);
  THTensor_(clampedRandom)(self, 0, max, _generator);
}

void THTensor_(geometric)(THTensor *self, double p, at::Generator *_generator)
{
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::geometric_distribution<double> geometric(p);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)geometric(gen););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined(TH_REAL_IS_FLOAT)
#define TH_REAL_MIN FLT_MIN
#elif defined(TH_REAL_IS_DOUBLE)
#define TH_REAL_MIN DBL_MIN
#endif

void THTensor_(uniform)(THTensor *self, double a, double b, at::Generator *_generator)
{
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  #if defined(TH_REAL_IS_FLOAT)
  at::uniform_real_distribution<float> uniform((float)a, (float)b);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)uniform(gen););
  #else
  at::uniform_real_distribution<double> uniform(a, b);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)uniform(gen););
  #endif
}

void THTensor_(normal)(THTensor *self, double mean, double stddev, at::Generator *_generator)
{
  const int64_t size = THTensor_(numel)(self);
  if (size >= 16 && THTensor_(isContiguous)(self)) {
    THVector_(normal_fill)(THStorage_(data)(THTensor_getStoragePtr(self)) + self->storage_offset(), size, _generator, mean, stddev);
  } else {
    auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);

    at::normal_distribution<double> normal(mean, stddev);
    TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)normal(gen););
  }
}

void THTensor_(normal_means)(THTensor *self, THTensor *means, double stddev, at::Generator *gen)
{
  THTensor_(resizeAs)(self, means);
  THTensor_(normal)(self, 0, stddev, gen);
  THTensor_(cadd)(self, self, 1, means);
}

void THTensor_(normal_stddevs)(THTensor *self, double mean, THTensor *stddevs, at::Generator *gen)
{
  THTensor_(resizeAs)(self, stddevs);
  THTensor_(normal)(self, 0, 1, gen);
  THTensor_(cmul)(self, self, stddevs);
  at::Tensor self_wrap = THTensor_wrap(self);
  self_wrap.add_(mean);
}

void THTensor_(normal_means_stddevs)(THTensor *self, THTensor *means, THTensor *stddevs, at::Generator *gen)
{
  THTensor_(resizeAs)(self, means);
  THTensor_(normal)(self, 0, 1, gen);
  THTensor_(cmul)(self, self, stddevs);
  THTensor_(cadd)(self, self, 1, means);
}

void THTensor_(exponential)(THTensor *self, double lambda, at::Generator *_generator)
{
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  at::exponential_distribution<double> exponential(lambda);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)exponential(gen););
}

#undef TH_REAL_MIN

void THTensor_(cauchy)(THTensor *self, double median, double sigma, at::Generator *_generator)
{
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  at::cauchy_distribution<double> cauchy(median, sigma);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)cauchy(gen););
}

void THTensor_(logNormal)(THTensor *self, double mean, double stdv, at::Generator *_generator)
{
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  at::lognormal_distribution<double> logNormal(mean, stdv);
  TH_TENSOR_APPLY(scalar_t, self, *self_data = (scalar_t)logNormal(gen););
}

void THTensor_(multinomialAliasSetup)(THTensor *probs, THLongTensor *J, THTensor *q)
{
  int64_t inputsize = THTensor_(nElement)(probs);
  THArgCheck(probs->dim() == 1, 1,
             "expected 1-D probability tensor, got %d-D probability tensor instead",
             probs->dim());
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
      THLongTensor_fastSet1d(J, i, -1L);
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
  THArgCheckWithCleanup((q_min >= 0),
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
      if(J_data[i] < 0)
        q_data[i] = 1.0;
    }
  THLongTensor_free(smaller);
  THLongTensor_free(larger);
}
void THTensor_(multinomialAliasDraw)(THLongTensor *self, THTensor *q, THLongTensor *J, int n_sample, at::Generator *_generator)
{
  THArgCheck(q->dim() == 1, 1,
             "expected 1-D probability table, got %d-D probability table instead",
             q->dim());
  THArgCheck(J->dim() == 1, 2,
             "expected 1-D alias table, got %d-D alias table instead",
             J->dim());
  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");
  int64_t K = THLongTensor_nElement(J);
  int64_t i = 0, _mask=0;
  scalar_t _q;
  THLongTensor_resize1d(self, n_sample);
  int64_t rand_ind, sample_idx, J_sample;
  auto gen = at::get_generator_or_default<at::CPUGenerator>(_generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  for (i=0; i < n_sample; i++)
    {
      at::uniform_real_distribution<double> uniform(0, K);
      rand_ind = uniform(gen);

      _q = THTensor_(fastGet1d)(q, rand_ind);
      at::bernoulli_distribution<double> bernoulli(_q);
      _mask = static_cast<int64_t>(bernoulli(gen));

      J_sample = THLongTensor_fastGet1d(J, rand_ind);

      sample_idx = J_sample*(1 -_mask) + rand_ind * _mask;

      THLongTensor_fastSet1d(self, i, sample_idx);
    }
}
#endif

#if defined(TH_REAL_IS_BYTE)
void THTensor_(getRNGState)(at::Generator *_generator, THTensor *self)
{
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(_generator->mutex_);

  auto cpu_generator = at::check_generator<at::CPUGenerator>(_generator);

  THTensor_(resize1d)(self, cpu_generator->get_rng_state_size());

  const size_t size = THTensor_(nElement)(self);
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");

  cpu_generator->get_rng_state(self->data(), size);
}

void THTensor_(setRNGState)(at::Generator *_generator, THTensor *self)
{
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(_generator->mutex_);

  auto cpu_generator = at::check_generator<at::CPUGenerator>(_generator);

  const size_t size = THTensor_(nElement)(self);
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");

  cpu_generator->set_rng_state(self->data(), size);
}
#endif
#endif
