#pragma once

#include <ATen/core/TransformationHelper.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/MathConstants.h>
#include <c10/macros/Macros.h>

#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

/**
 * Distributions kernel adapted from THRandom.cpp
 * The kernels try to follow std::random distributions signature
 * For instance: in ATen
 *      auto gen = at::detail::createCPUGenerator();
 *      at::uniform_real_distribution<double> uniform(0, 1);
 *      auto sample = uniform(gen.get());
 *
 *      vs std::random
 *
 *      std::mt19937 gen;
 *      std::uniform_real_distribution uniform(0, 1);
 *      auto sample = uniform(gen);
 */


namespace at {
namespace {

/**
 * Samples a discrete uniform distribution in the range [base, base+range) of type T
 */
template <typename T>
struct uniform_int_from_to_distribution {

  C10_HOST_DEVICE inline uniform_int_from_to_distribution(uint64_t range, int64_t base) : range_(range), base_(base) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
#ifdef FBCODE_CAFFE2
    if ((
      std::is_same_v<T, int64_t> ||
      std::is_same_v<T, double> ||
      std::is_same_v<T, float> ||
      std::is_same_v<T, at::BFloat16>) && range_ >= 1ULL << 32)
#else
    if (range_ >= 1ULL << 28) // allow approx 5% skew in uniform int generation using %
#endif
    {
      return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
    } else {
      return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
    }
  }

  private:
    uint64_t range_;
    int64_t base_;
};

/**
 * Samples a discrete uniform distribution in the range [min_value(int64_t), max_value(int64_t)]
 */
template <typename T>
struct uniform_int_full_range_distribution {

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    return transformation::uniform_int_full_range<T>(generator->random64());
  }

};

/**
 * Samples a discrete uniform distribution in the range [0, max_value(T)] for integral types
 * and [0, 2^mantissa] for floating-point types.
 */
template <typename T>
struct uniform_int_distribution {

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int64_t>) {
      return transformation::uniform_int<T>(generator->random64());
    } else {
      return transformation::uniform_int<T>(generator->random());
    }
  }

};

/**
 * Samples a uniform distribution in the range [from, to) of type T
 */
template <typename T>
struct uniform_real_distribution {

  C10_HOST_DEVICE inline uniform_real_distribution(T from, T to) : from_(from), to_(to) {
    TORCH_CHECK_IF_NOT_ON_CUDA(from <= to);
    TORCH_CHECK_IF_NOT_ON_CUDA(to - from <= std::numeric_limits<T>::max());
  }

  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG* generator) const {
    if constexpr (std::is_same_v<T, double>) {
      return transformation::uniform_real<T>(generator->random64(), from_, to_);
    } else {
      return transformation::uniform_real<T>(generator->random(), from_, to_);
    }
  }

  private:
    T from_;
    T to_;
};

template <typename RNG,
          typename = decltype(&RNG::next_double_normal_sample),
          typename = decltype(&RNG::set_next_double_normal_sample)>
C10_HOST_DEVICE bool maybe_get_next_normal_sample(RNG* generator, double* ret) {
  const auto sample = generator->next_double_normal_sample();
  if (!sample.has_value())
    return false;
  *ret = sample.value();
  generator->set_next_double_normal_sample(std::nullopt);
  return true;
}

template <typename RNG,
          typename = decltype(&RNG::next_float_normal_sample),
          typename = decltype(&RNG::set_next_float_normal_sample)>
C10_HOST_DEVICE bool maybe_get_next_normal_sample(RNG* generator, float* ret) {
  const auto sample = generator->next_float_normal_sample();
  if (!sample.has_value())
    return false;
  *ret = sample.value();
  generator->set_next_float_normal_sample(std::nullopt);
  return true;
}

template <typename RNG>
C10_HOST_DEVICE bool maybe_get_next_normal_sample(RNG* /* generator */, void* /* ret */) {
  return false;
}

template <typename RNG,
          typename = decltype(&RNG::set_next_double_normal_sample)>
C10_HOST_DEVICE void maybe_set_next_normal_sample(RNG* generator, const double* cache) {
  generator->set_next_double_normal_sample(*cache);
}

template <typename RNG,
          typename = decltype(&RNG::set_next_float_normal_sample)>
C10_HOST_DEVICE void maybe_set_next_normal_sample(RNG* generator, const float* cache) {
  generator->set_next_float_normal_sample(*cache);
}

template <typename RNG>
C10_HOST_DEVICE void maybe_set_next_normal_sample(RNG* /* generator */, const void* /* cache */) {
}

/**
 * Samples a normal distribution using the Box-Muller method
 * Takes mean and standard deviation as inputs
 * Note that Box-muller method returns two samples at a time.
 * Hence, we cache the "next" sample in the CPUGeneratorImpl class.
 */
template <typename T>
struct normal_distribution {

  C10_HOST_DEVICE inline normal_distribution(T mean_in, T stdv_in) : mean(mean_in), stdv(stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in >= 0, "stdv_in must be positive: ", stdv_in);
  }

  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG* generator) const {
    dist_acctype<T> ret;
    // return cached values if available
    if (maybe_get_next_normal_sample(generator, &ret)) {
      return transformation::normal(ret, mean, stdv);
    }

    // otherwise generate new normal values
    uniform_real_distribution<T> uniform(0.0, 1.0);
    const dist_acctype<T> u1 = uniform(generator);
    const dist_acctype<T> u2 = uniform(generator);
    const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log1p(-u2));
    const dist_acctype<T> theta = static_cast<T>(2.0) * c10::pi<T> * u1;
    const dist_acctype<T> sample = r * ::sin(theta);
    maybe_set_next_normal_sample(generator, &sample);

    ret = r * ::cos(theta);
    return transformation::normal(ret, mean, stdv);
  }

  private:
    T mean;
    T stdv;
};

template <typename T>
struct DiscreteDistributionType { using type = float; };

template <> struct DiscreteDistributionType<double> { using type = double; };

/**
 * Samples a bernoulli distribution given a probability input
 */
template <typename T>
struct bernoulli_distribution {

  C10_HOST_DEVICE inline bernoulli_distribution(T p_in) : p(p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in >= 0 && p_in <= 1);
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::bernoulli<T>(uniform(generator), p);
  }

  private:
    T p;
};

/**
 * Samples a geometric distribution given a probability input
 */
template <typename T>
struct geometric_distribution {

  C10_HOST_DEVICE inline geometric_distribution(T p_in) : p(p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in > 0 && p_in < 1);
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::geometric<T>(uniform(generator), p);
  }

  private:
    T p;
};

/**
 * Samples an exponential distribution given a lambda input
 */
template <typename T>
struct exponential_distribution {

  C10_HOST_DEVICE inline exponential_distribution(T lambda_in) : lambda(lambda_in) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::exponential<T>(uniform(generator), lambda);
  }

  private:
    T lambda;
};

/**
 * Samples a cauchy distribution given median and sigma as inputs
 */
template <typename T>
struct cauchy_distribution {

  C10_HOST_DEVICE inline cauchy_distribution(T median_in, T sigma_in) : median(median_in), sigma(sigma_in) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::cauchy<T>(uniform(generator), median, sigma);
  }

  private:
    T median;
    T sigma;
};

/**
 * Samples a lognormal distribution
 * Takes mean and standard deviation as inputs
 * Outputs two samples at a time
 */
template <typename T>
struct lognormal_distribution {

  C10_HOST_DEVICE inline lognormal_distribution(T mean_in, T stdv_in) : mean(mean_in), stdv(stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in > 0);
  }

  template<typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG* generator) const {
    normal_distribution<T> normal(mean, stdv);
    return transformation::log_normal<T>(normal(generator));
  }

  private:
    T mean;
    T stdv;
};
}
} // namespace at
