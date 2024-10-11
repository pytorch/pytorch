#pragma once

#include <ATen/core/Array.h>
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
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    if ((
      std::is_same<T, int64_t>::value ||
      std::is_same<T, double>::value ||
      std::is_same<T, float>::value ||
      std::is_same<T, at::BFloat16>::value) && range_ >= 1ULL << 32)
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
  C10_HOST_DEVICE inline T operator()(RNG generator) {
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
  C10_HOST_DEVICE inline T operator()(RNG generator) {
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

  C10_HOST_DEVICE inline uniform_real_distribution(T from, T to) {
    TORCH_CHECK_IF_NOT_ON_CUDA(from <= to);
    TORCH_CHECK_IF_NOT_ON_CUDA(to - from <= std::numeric_limits<T>::max());
    from_ = from;
    to_ = to;
  }

  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator){
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

// The SFINAE checks introduced in #39816 looks overcomplicated and must revisited
// https://github.com/pytorch/pytorch/issues/40052
#define DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(member)              \
template <typename T>                                                \
struct has_member_##member                                           \
{                                                                    \
    typedef char yes;                                                \
    typedef long no;                                                 \
    template <typename U> static yes test(decltype(&U::member));     \
    template <typename U> static no test(...);                       \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); \
}

DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_double_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_double_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_float_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_float_normal_sample);

#define DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(TYPE)                                      \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            has_member_next_##TYPE##_normal_sample<RNG>::value &&                                   \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* generator, ret_type* ret) {  \
  if (generator->next_##TYPE##_normal_sample()) {                                                   \
    *ret = *(generator->next_##TYPE##_normal_sample());                                             \
    generator->set_next_##TYPE##_normal_sample(std::optional<TYPE>());                              \
    return true;                                                                                    \
  }                                                                                                 \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            !has_member_next_##TYPE##_normal_sample<RNG>::value ||                                  \
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type* /*ret*/) {  \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline void maybe_set_next_##TYPE##_normal_sample(RNG* generator, ret_type cache) { \
  generator->set_next_##TYPE##_normal_sample(cache);                                                \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline void maybe_set_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type /*cache*/) { \
}

DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(double);
DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(float);

/**
 * Samples a normal distribution using the Box-Muller method
 * Takes mean and standard deviation as inputs
 * Note that Box-muller method returns two samples at a time.
 * Hence, we cache the "next" sample in the CPUGeneratorImpl class.
 */
template <typename T>
struct normal_distribution {

  C10_HOST_DEVICE inline normal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in >= 0, "stdv_in must be positive: ", stdv_in);
    mean = mean_in;
    stdv = stdv_in;
  }

  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator){
    dist_acctype<T> ret;
    // return cached values if available
    if constexpr (std::is_same_v<T, double>) {
      if (maybe_get_next_double_normal_sample(generator, &ret)) {
        return transformation::normal(ret, mean, stdv);
      }
    } else {
      if (maybe_get_next_float_normal_sample(generator, &ret)) {
        return transformation::normal(ret, mean, stdv);
      }
    }
    // otherwise generate new normal values
    uniform_real_distribution<T> uniform(0.0, 1.0);
    const dist_acctype<T> u1 = uniform(generator);
    const dist_acctype<T> u2 = uniform(generator);
    const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log1p(-u2));
    const dist_acctype<T> theta = static_cast<T>(2.0) * c10::pi<T> * u1;
    if constexpr (std::is_same_v<T, double>) {
      maybe_set_next_double_normal_sample(generator, r * ::sin(theta));
    } else {
      maybe_set_next_float_normal_sample(generator, r * ::sin(theta));
    }
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

  C10_HOST_DEVICE inline bernoulli_distribution(T p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in >= 0 && p_in <= 1);
    p = p_in;
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
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

  C10_HOST_DEVICE inline geometric_distribution(T p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in > 0 && p_in < 1);
    p = p_in;
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
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
  C10_HOST_DEVICE inline T operator()(RNG generator) {
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
  C10_HOST_DEVICE inline T operator()(RNG generator) {
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

  C10_HOST_DEVICE inline lognormal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in > 0);
    mean = mean_in;
    stdv = stdv_in;
  }

  template<typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator){
    normal_distribution<T> normal(mean, stdv);
    return transformation::log_normal<T>(normal(generator));
  }

  private:
    T mean;
    T stdv;
};
}
} // namespace at
