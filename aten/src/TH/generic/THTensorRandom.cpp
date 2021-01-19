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
#include <ATen/core/DistributionsHelper.h>

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined(TH_REAL_IS_FLOAT)
#define TH_REAL_MIN FLT_MIN
#elif defined(TH_REAL_IS_DOUBLE)
#define TH_REAL_MIN DBL_MIN
#endif


#undef TH_REAL_MIN

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
void THTensor_(multinomialAliasDraw)(THLongTensor *self, THTensor *q, THLongTensor *J, int n_sample, c10::optional<at::Generator> _generator)
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
  auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(_generator, at::detail::getDefaultCPUGenerator());
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
#endif
