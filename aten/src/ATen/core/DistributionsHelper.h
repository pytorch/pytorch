#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/core/Array.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Optional.h>
#include <c10/macros/Macros.h>

#include <type_traits>
#include <limits>
#include <cmath>

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

// Using DistAccumType in accumulate types for distributions.
// Note: Ideally we'd be using ATen/AccumulateType.h but looks
// like the there is some inconsistency in how accumulate types
// are mapped currently, e.g. for the cpu side, float is mapped
// to double.
template <typename T>
struct DistAccumType {  };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct DistAccumType<half> { using type = float; };
#endif
template <> struct DistAccumType<Half> { using type = float; };
template <> struct DistAccumType<float> { using type = float; };
template <> struct DistAccumType<double> { using type = double; };

template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

/**
 * Samples a discrete uniform distribution in the range [base, base+range) of type T
 * This is a transformation function for
 * https://pytorch.org/docs/stable/tensors.html?highlight=random#torch.Tensor.random_
 * when both `from` and `to` are specified.
 */
template <typename T>
struct uniform_int_from_to_distribution {

  C10_HOST_DEVICE inline uniform_int_from_to_distribution(uint64_t range_, int64_t base_) {
    range = range_;
    base = base_;
  }

  template <typename V,
            typename std::enable_if<std::is_integral<V>::value, int>::type = 0>
  C10_HOST_DEVICE inline T operator()(V val) {
    return static_cast<T>(static_cast<int64_t>((val % range) + base));
  }

  template <typename RNG,
            typename std::enable_if<(!std::is_fundamental<RNG>::value), int>::type = 0>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    if ((
      std::is_same<T, int64_t>::value ||
      std::is_same<T, double>::value ||
      std::is_same<T, float>::value ||
      std::is_same<T, at::BFloat16>::value) && range >= 1ULL << 32)
    {
      return operator()(generator->random64());
    } else {
      return operator()(generator->random());
    }
  }

  private:
    uint64_t range;
    int64_t base;
};

/**
 * Samples a discrete uniform distribution in the range [min_value(int64_t), max_value(int64_t)]
 * This is a transformation function for
 * https://pytorch.org/docs/stable/tensors.html?highlight=random#torch.Tensor.random_
 * when `from=min_value(int64_t)` and to=None
 */
template <typename T>
struct uniform_int_full_range_distribution {

  template <typename V,
            typename std::enable_if<std::is_integral<V>::value, int>::type = 0>
  C10_HOST_DEVICE inline T operator()(V val) {
    return static_cast<T>(static_cast<int64_t>(val));
  }

  template <typename RNG,
            typename std::enable_if<(!std::is_fundamental<RNG>::value), int>::type = 0>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    return operator()(generator->random64());
  }

};

/**
 * Samples a discrete uniform distribution in the range [0, max_value(T)] for integral types
 * and [0, 2^mantissa] for floating point types. This is a transformation function for
 * https://pytorch.org/docs/stable/tensors.html?highlight=random#torch.Tensor.random_
 * when used without specifing `from` and `to`.
 */
template <typename T>
struct uniform_int_distribution {

  template <typename V,
            typename std::enable_if<std::is_integral<V>::value, int>::type = 0>
  C10_HOST_DEVICE inline T operator()(V val) {
    if (std::is_same<T, bool>::value) {
      return static_cast<bool>(val & 1);
    } else if (std::is_same<T, double>::value) {
      return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
    } else if (std::is_same<T, int64_t>::value) {
      return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
    } else if (std::is_floating_point<T>::value || std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value) {
      return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
    } else if (std::is_integral<T>::value) {
      return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
    } else {
      assert(false);
      return 0;
    }
  }

  template <typename RNG,
            typename std::enable_if<(!std::is_fundamental<RNG>::value), int>::type = 0>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    if (std::is_same<T, double>::value || std::is_same<T, int64_t>::value) {
      return operator()(generator->random64());
    } else {
      return operator()(generator->random());
    }
  }

};

/**
 * Samples a uniform distribution in the range [a, b) of type T
 */
template <typename T>
struct uniform_real_distribution {

  C10_HOST_DEVICE inline uniform_real_distribution(T a_in, T b_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(a_in <= b_in);
    TORCH_CHECK_IF_NOT_ON_CUDA(b_in-a_in <= std::numeric_limits<T>::max());
    a = a_in;
    b = b_in;
  }

  template <typename V,
            typename std::enable_if<std::is_integral<V>::value, int>::type = 0>
  C10_HOST_DEVICE inline T operator()(V val) {
    constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
    constexpr auto DIVISOR = static_cast<T>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
    dist_acctype<T> x = (val & MASK) * DIVISOR;
    return (x * (b - a) + a);
  }

  template <typename RNG,
            typename std::enable_if<(!std::is_fundamental<RNG>::value), int>::type = 0>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator){
    if(std::is_same<T, double>::value) {
      return operator()(generator->random64());
    } else {
      return operator()(generator->random());
    }
  }

  private:
    T a;
    T b;
};

/**
 * Samples a normal distribution using the Box-Muller method
 * Takes mean and standard deviation as inputs
 * Note that Box-muller method returns two samples at a time.
 * Hence, we cache the "next" sample in the CPUGeneratorImpl class.
 */
template <typename T>
struct normal_distribution {

  inline normal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in > 0);
    mean = mean_in;
    stdv = stdv_in;
  }

  template <typename RNG>
  inline dist_acctype<T> operator()(RNG generator){
    dist_acctype<T> ret;
#if !defined(__CUDACC__) && !defined(__HIPCC__)
    // return cached values if available
    if (std::is_same<T, double>::value) {
      if (generator->next_double_normal_sample()) {
        ret = *(generator->next_double_normal_sample()) * stdv + mean;
        // reset c10::optional to null
        generator->set_next_double_normal_sample(c10::optional<double>());
        return ret;
      }
    } else {
      if (generator->next_float_normal_sample()) {
        ret = *(generator->next_float_normal_sample()) * stdv + mean;
        // reset c10::optional to null
        generator->set_next_float_normal_sample(c10::optional<float>());
        return ret;
      }
    }
#endif
    // otherwise generate new normal values
    uniform_real_distribution<T> uniform(0.0, 1.0);
    const dist_acctype<T> u1 = uniform(generator);
    const dist_acctype<T> u2 = uniform(generator);
    const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log(static_cast<T>(1.0)-u2));
    const dist_acctype<T> theta = static_cast<T>(2.0) * static_cast<T>(M_PI) * u1;
#if !defined(__CUDACC__) && !defined(__HIPCC__)
    if (std::is_same<T, double>::value) {
      dist_acctype<double> cache = r * ::sin(theta);
      generator->set_next_double_normal_sample(c10::optional<double>(cache));
    } else {
      dist_acctype<float> cache = r * ::sin(theta);
      generator->set_next_float_normal_sample(c10::optional<float>(cache));
    }
#endif
    ret = r * ::cos(theta) * stdv + mean;
    return ret;
  }

  private:
    T mean;
    T stdv;
};

/**
 * Samples a bernoulli distribution given a probability input
 */
template <typename T>
struct bernoulli_distribution {

  inline bernoulli_distribution(T p_in) {
    TORCH_CHECK(p_in >= 0 && p_in <= 1);
    p = p_in;
  }

  template <typename RNG>
  inline int operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return uniform(generator) < p;
  }

  private:
    T p;
};

/**
 * Samples a geometric distribution given a probability input
 */
template <typename T>
struct geometric_distribution {

  inline geometric_distribution(T p_in) {
    TORCH_CHECK(p_in > 0 && p_in < 1);
    p = p_in;
  }

  template <typename RNG>
  inline int operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    dist_acctype<T> sample = uniform(generator);
    return static_cast<int>(::log(static_cast<T>(1.0)-sample) / ::log(p)) + 1;
  }

  private:
    T p;
};

/**
 * Samples an exponential distribution given a lambda input
 */
template <typename T>
struct exponential_distribution {

  inline exponential_distribution(T lambda_in) {
    lambda = lambda_in;
  }

  template <typename RNG>
  __ubsan_ignore_float_divide_by_zero__ inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    dist_acctype<T> sample = uniform(generator);
    return static_cast<T>(-1.0) / lambda * ::log(static_cast<T>(1.0)-sample);
  }

  private:
    T lambda;
};

/**
 * Samples a cauchy distribution given median and sigma as inputs
 */
template <typename T>
struct cauchy_distribution {

  inline cauchy_distribution(T median_in, T sigma_in) {
    median = median_in;
    sigma = sigma_in;
  }

  template <typename RNG>
  inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return median + sigma * ::tan(static_cast<T>(M_PI) * (uniform(generator)-static_cast<T>(0.5)));
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

  inline lognormal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK(stdv_in > 0);
    mean = mean_in;
    stdv = stdv_in;
  }

  template<typename RNG>
  inline T operator()(RNG generator){
    normal_distribution<T> normal(mean, stdv);
    return ::exp(normal(generator));
  }

  private:
    T mean;
    T stdv;
};

} // namespace at
