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
static constexpr uint64_t DOUBLE_MASK = (1ULL << 53) - 1;
static constexpr double DOUBLE_DIVISOR = 1.0 / (1ULL << 53);

// floats have 23 bits of mantissa (fractional part)
static constexpr uint32_t FLOAT_MASK = (1 << 24) - 1;
static constexpr float FLOAT_DIVISOR = 1.0f / (1 << 24);

/**
 * Samples a uniform distribution in the range [0,1) of type T
 */
template <typename T>
struct uniform_real_distribution {

  C10_HOST_DEVICE inline uniform_real_distribution(T a_in, T b_in) {
    #ifndef __HIP_PLATFORM_HCC__
      #ifdef __CUDA_ARCH__
        assert(a_in <= b_in);
        assert(b_in-a_in <= std::numeric_limits<T>::max());
      #else
        AT_ASSERT(a_in <= b_in);
        AT_ASSERT(b_in-a_in <= std::numeric_limits<T>::max());
      #endif
    #endif
    a = a_in;
    b = b_in;
  }

  C10_HOST inline dist_acctype<T> operator()(at::CPUGenerator* generator){
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
 */
template <typename T>
struct normal_distribution {

  C10_HOST_DEVICE inline normal_distribution(T mean_in, T stdv_in) {
    #ifndef __HIP_PLATFORM_HCC__
      #ifdef __CUDA_ARCH__
        assert(stdv_in > 0);
      #else
        AT_ASSERT(stdv_in > 0);
      #endif
    #endif
    mean = mean_in;
    stdv = stdv_in;
    cache_index = 0;
  }

  C10_HOST inline T operator()(at::CPUGenerator* generator){
    if(cache_index == 0) {
      uniform_real_distribution<T> uniform(0.0, 1.0);
      const T theta = static_cast<T>(2.0) * static_cast<T>(M_PI) * uniform(generator);
      T u1 = uniform(generator);
      T r = ::sqrt(static_cast<T>(-2.0) * ::log(static_cast<T>(1.0)-u1));
      result[0] = r * ::cos(theta) * stdv + mean;
      result[1] = r * ::sin(theta) * stdv + mean;
    }
    T ret = result[cache_index];
    cache_index = (cache_index + 1) & 1;
    return ret;
  }

  private:
    T mean;
    T stdv;
    vect_type<T> result;
    uint32_t cache_index;
};

/**
 * Samples a bernoulli distribution given a probability input
 */
template <typename T>
struct bernoulli_distribution {

  C10_HOST_DEVICE inline bernoulli_distribution(T p_in) {
    #ifndef __HIP_PLATFORM_HCC__
      #ifdef __CUDA_ARCH__
        assert(p_in >= 0 && p_in <= 1);
      #else
        AT_ASSERT(p_in >= 0 && p_in <= 1);
      #endif
    #endif
    p = p_in;
  }

  C10_HOST inline T operator()(at::CPUGenerator* generator) { 
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

  C10_HOST_DEVICE inline geometric_distribution(T p_in) {
    #ifndef __HIP_PLATFORM_HCC__
      #ifdef __CUDA_ARCH__
        assert(p_in > 0 && p_in < 1);
      #else
        AT_ASSERT(p_in > 0 && p_in < 1);
      #endif
    #endif
    p = p_in;
  }

  C10_HOST inline int operator()(at::CPUGenerator* generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    auto sample = uniform(generator);
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

  C10_HOST_DEVICE inline exponential_distribution(T lambda_in) {
    lambda = lambda_in;
  }

  C10_HOST inline T operator()(at::CPUGenerator* generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    auto sample = uniform(generator);
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

  C10_HOST_DEVICE inline cauchy_distribution(T median_in, T sigma_in) {
    median = median_in;
    sigma = sigma_in;
  }

  C10_HOST inline T operator()(at::CPUGenerator* generator) {
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

  C10_HOST_DEVICE inline lognormal_distribution(T mean_in, T stdv_in) {
    #ifndef __HIP_PLATFORM_HCC__
      #ifdef __CUDA_ARCH__
        assert(stdv_in > 0);
      #else
        AT_ASSERT(stdv_in > 0);
      #endif
    #endif
    mean = mean_in;
    stdv = stdv_in;
  }

  C10_HOST inline T operator()(at::CPUGenerator* generator){
    normal_distribution<T> normal(mean, stdv);
    return ::exp(normal(generator));
  }

  private:
    T mean;
    T stdv;
};

} // namespace at
