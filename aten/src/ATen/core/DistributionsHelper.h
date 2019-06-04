#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <ATen/CPUGenerator.h>
#include <ATen/core/Array.h>
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

// Using VectorType in Box-muller derived distributions to avoid
// code duplication
template <typename T>
struct VectorType {  };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct VectorType<half> { using type = at::detail::Array<float, 2>; };
#endif
template <> struct VectorType<Half> { using type = at::detail::Array<float, 2>; };
template <> struct VectorType<float> { using type = at::detail::Array<float, 2>; };
template <> struct VectorType<double> { using type = at::detail::Array<double, 2>; };

template <typename T>
using vect_type = typename VectorType<T>::type;

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

// Constants for uniform distribution
// doubles have 52 bits of mantissa (fractional part)
constexpr uint64_t DOUBLE_MASK = (1ULL << 53) - 1;
constexpr double DOUBLE_DIVISOR = 1.0 / (1ULL << 53);

// floats have 23 bits of mantissa (fractional part)
constexpr uint32_t FLOAT_MASK = (1 << 24) - 1;
constexpr float FLOAT_DIVISOR = 1.0f / (1 << 24);

/**
 * Samples a uniform distribution in the range [0,1) of type T
 */
template <typename T>
struct uniform_real_distribution {

  inline uniform_real_distribution(T a_in, T b_in) {
    TORCH_CHECK(a_in <= b_in);
    TORCH_CHECK(b_in-a_in <= std::numeric_limits<T>::max());
    a = a_in;
    b = b_in;
  }

  inline dist_acctype<T> operator()(at::CPUGenerator* generator){
    dist_acctype<T> x;
    if(std::is_same<T, double>::value) {
      x = (generator->random64() & DOUBLE_MASK) * DOUBLE_DIVISOR;
    } else {
      x = (generator->random() & FLOAT_MASK) * FLOAT_DIVISOR;
    }
    return (x * (b - a) + a);
  }

  private:
    T a;
    T b; 
};

/**
 * Samples a normal distribution using the Box-Muller method
 * Takes mean and standard deviation as inputs
 * Note that Box-muller method returns two samples at a time.
 * Hence, we cache the "next" sample in the CPUGenerator class.
 */
template <typename T>
struct normal_distribution {

  inline normal_distribution(T mean_in, T stdv_in) {
    TORCH_CHECK(stdv_in > 0);
    mean = mean_in;
    stdv = stdv_in;
  }

  inline dist_acctype<T> operator()(at::CPUGenerator* generator){
    dist_acctype<T> ret;
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
    // otherwise generate new normal values
    uniform_real_distribution<T> uniform(0.0, 1.0);
    const dist_acctype<T> u1 = uniform(generator);
    const dist_acctype<T> u2 = uniform(generator);
    const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log(static_cast<T>(1.0)-u2));
    const dist_acctype<T> theta = static_cast<T>(2.0) * static_cast<T>(M_PI) * u1;
    if (std::is_same<T, double>::value) {
      dist_acctype<double> cache = r * ::sin(theta);
      generator->set_next_double_normal_sample(c10::optional<double>(cache));
    } else {
      dist_acctype<float> cache = r * ::sin(theta);
      generator->set_next_float_normal_sample(c10::optional<float>(cache));
    }
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

  inline T operator()(at::CPUGenerator* generator) { 
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return uniform(generator) <= p;
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

  inline int operator()(at::CPUGenerator* generator) {
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

  inline T operator()(at::CPUGenerator* generator) {
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

  inline T operator()(at::CPUGenerator* generator) {
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

  inline T operator()(at::CPUGenerator* generator){
    normal_distribution<T> normal(mean, stdv);
    return ::exp(normal(generator));
  }

  private:
    T mean;
    T stdv;
};

} // namespace at
