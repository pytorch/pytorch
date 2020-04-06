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
#include <TH/THGenerator.hpp>

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined(TH_REAL_IS_FLOAT)
#define TH_REAL_MIN FLT_MIN
#elif defined(TH_REAL_IS_DOUBLE)
#define TH_REAL_MIN DBL_MIN
#endif

void THTensor_(uniform)(THTensor *self, double a, double b, at::Generator _generator)
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
void THTensor_(multinomialAliasDraw)(THLongTensor *self, THTensor *q, THLongTensor *J, int n_sample, at::Generator _generator)
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
void THTensor_(getRNGState)(at::Generator _generator, THTensor *self)
{
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(_generator->mutex_);
  static const size_t size = sizeof(THGeneratorStateNew);
  THTensor_(resize1d)(self, size);
  THArgCheck(THTensor_(nElement)(self) == size, 1, "RNG state is wrong size");
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  static_assert(std::is_pod<THGeneratorStateNew>::value, "THGeneratorStateNew is not a PODType");

  // cast byte tensor to POD type
  THGeneratorStateNew* rng_state = (THGeneratorStateNew*)self->data<scalar_t>();

  // accumulate generator data to be copied into byte tensor
  auto accum_state = std::make_unique<THGeneratorStateNew>();
  auto cast_generator = at::check_generator<at::CPUGenerator>(_generator);
  auto rng_data = cast_generator->engine().data();
  accum_state->legacy_pod.the_initial_seed = rng_data.seed_;
  accum_state->legacy_pod.left = rng_data.left_;
  accum_state->legacy_pod.seeded = rng_data.seeded_;
  accum_state->legacy_pod.next = rng_data.next_;
  std::copy(rng_data.state_.begin(), rng_data.state_.end(), std::begin(accum_state->legacy_pod.state));
  accum_state->legacy_pod.normal_x = 0.0; // we don't use it anymore and this is just a dummy
  accum_state->legacy_pod.normal_rho = 0.0; // we don't use it anymore and this is just a dummy
  accum_state->legacy_pod.normal_is_valid = false;
  accum_state->legacy_pod.normal_y = 0.0;
  accum_state->next_float_normal_sample = 0.0f;
  accum_state->is_next_float_normal_sample_valid = false;
  if(cast_generator->next_double_normal_sample()) {
    accum_state->legacy_pod.normal_is_valid = true;
    accum_state->legacy_pod.normal_y = *(cast_generator->next_double_normal_sample());
  }
  if(cast_generator->next_float_normal_sample()) {
    accum_state->is_next_float_normal_sample_valid = true;
    accum_state->next_float_normal_sample = *(cast_generator->next_float_normal_sample());
  }

  memcpy(rng_state, accum_state.get(), size);
}

void THTensor_(setRNGState)(at::Generator _generator, THTensor *self)
{
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(_generator->mutex_);
  auto cast_generator = at::check_generator<at::CPUGenerator>(_generator);
  THArgCheck(THTensor_(isContiguous)(self), 1, "RNG state needs to be contiguous");
  static_assert(std::is_pod<THGeneratorState>::value, "THGeneratorState is not a PODType");
  static_assert(std::is_pod<THGeneratorStateNew>::value, "THGeneratorStateNew is not a PODType");

  static const size_t size_legacy = sizeof(THGeneratorState);
  static const size_t size_current = sizeof(THGeneratorStateNew);
  static_assert(size_legacy != size_current, "Legacy THGeneratorState and THGeneratorStateNew can't be of the same size");

  at::mt19937 engine;
  auto float_normal_sample = c10::optional<float>();
  auto double_normal_sample = c10::optional<double>();

  // Construct the state of at::CPUGenerator based on input byte tensor size.
  THGeneratorState* legacy_pod;
  if (THTensor_(nElement)(self) == size_legacy) {
    legacy_pod = (THGeneratorState*)self->data<scalar_t>();
    // Note that in legacy THGeneratorState, we didn't have float version
    // of normal sample and hence we leave the c10::optional<float> as is

    // Update next_double_normal_sample.
    // Note that legacy THGeneratorState stores two uniform values (normal_x, normal_y)
    // and a rho value (normal_rho). These three values were redundant and in the new
    // DistributionsHelper.h, we store the actual extra normal sample, rather than three
    // intermediate values.
    if (legacy_pod->normal_is_valid) {
      auto r = legacy_pod->normal_rho;
      auto theta = 2.0 * M_PI * legacy_pod->normal_x;
      // we return the sin version of the normal sample when in caching mode
      double_normal_sample = c10::optional<double>(r * ::sin(theta));
    }
  } else if (THTensor_(nElement)(self) == size_current) {
    auto rng_state = (THGeneratorStateNew*)self->data<scalar_t>();
    legacy_pod = &rng_state->legacy_pod;
    // update next_float_normal_sample
    if (rng_state->is_next_float_normal_sample_valid) {
      float_normal_sample = c10::optional<float>(rng_state->next_float_normal_sample);
    }

    // Update next_double_normal_sample.
    // Note that in getRNGState, we now return the actual normal sample in normal_y
    // and if it's valid in normal_is_valid. The redundant normal_x and normal_rho
    // are squashed to 0.0.
    if (legacy_pod->normal_is_valid) {
      double_normal_sample = c10::optional<double>(legacy_pod->normal_y);
    }
  } else {
    AT_ERROR("Expected either a THGeneratorState of size ", size_legacy,
             " or a THGeneratorStateNew of size ", size_current,
             " but found the input RNG state size to be ", THTensor_(nElement)(self));
  }

  // construct engine_
  // Note that legacy THGeneratorState stored a state array of 64 bit uints, whereas in our
  // redefined mt19937, we have changed to a state array of 32 bit uints. Hence, we are
  // doing a std::copy.
  at::mt19937_data_pod rng_data;
  std::copy(std::begin(legacy_pod->state), std::end(legacy_pod->state), rng_data.state_.begin());
  rng_data.seed_ = legacy_pod->the_initial_seed;
  rng_data.left_ = legacy_pod->left;
  rng_data.seeded_ = legacy_pod->seeded;
  rng_data.next_ = static_cast<uint32_t>(legacy_pod->next);
  engine.set_data(rng_data);
  THArgCheck(engine.is_valid(), 1, "Invalid mt19937 state");
  cast_generator->set_engine(engine);
  cast_generator->set_next_float_normal_sample(float_normal_sample);
  cast_generator->set_next_double_normal_sample(double_normal_sample);
}
#endif
#endif
