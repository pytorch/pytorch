#pragma once

#include <ATen/CPUGenerator.h>
#include <type_traits>

// Distributions kernel adapted from THRandom.cpp
// The kernels try to follow std::random distributions signature
// For instance: in ATen
//      CPUGenerator* gen = new CPUGenerator();
//      at::uniform_real_distribution<double> uniform(0, 1);
//      auto sample = uniform(gen);
//      
//      vs std::random
//
//      std::mt19937 gen;
//      std::uniform_real_distribution uniform(0, 1);
//      auto sample = uniform(gen);
//
// Note: Why are operator() signatures different for CUDA and CPU?
//       This is because, for CUDA, we will be using Philox4_32_10 engine
//       inside a kernel and the number of random samples we would need is known
//       before hand. For the CPU side, it's easier to let CPUGenerator encapsulate
//       Philox4_32_10 and manage its state. Hence, for CUDA, operator() takes
//       the Philox4_32_10 object directly and CPU take CPUGenerator object directly.
namespace at {

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

  template<typename U = T> 
  C10_HOST inline typename std::enable_if<std::is_same<U, double>::value, double>::type
  operator()(at::CPUGenerator* generator){
    double x = (generator->random64() & ((1ULL << 53) - 1)) * ::ldexp(1.0, -53);
    return (x * (b - a) + a);

  }

  template<typename U = T> 
  C10_HOST inline typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, float>::type
  operator()(at::CPUGenerator* generator){
    float x = (generator->random() & ((1 << 24) - 1)) * ::ldexp(1.0, -24);
    return (x * (b - a) + a);
  }

  private:
    T a;
    T b; 
};

/**
 * Samples a normal distribution using the Box-Muller method
 * Takes mean and standard deviation as inputs
 * Outputs two samples at a time
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
  }

  template<typename U = T> 
  C10_HOST inline typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(at::CPUGenerator* generator){
    uniform_real_distribution<double> uniform(0.0, 1.0);
    DOUBLE2 result;
    double u2 = uniform(generator);
    double u1 = uniform(generator);
    // extra pre-caution to make sure log never gets zero
    if (u1 == 0.0) {
      u1 = std::numeric_limits<double>::min();
    }
    double r = ::sqrt(-2.0 * ::log(u1));
    result[0] = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result[1] = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
  }

  template<typename U = T> 
  C10_HOST inline typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(at::CPUGenerator* generator){
    uniform_real_distribution<float> uniform(0.0, 1.0);
    FLOAT2 result;
    float u2 = uniform(generator);
    float u1 = uniform(generator);
    // extra pre-caution to make sure log never gets zero
    if (u1 == 0.0f) {
      u1 = std::numeric_limits<float>::min();
    }
    float r = ::sqrt(-2.0 * ::log(u1));
    result[0] = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result[1] = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
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
    // extra pre-caution to make sure log never gets zero
    if (sample == static_cast<T>(0.0)) {
      sample = std::numeric_limits<T>::min();
    }
    return static_cast<int>(::log(sample) / ::log(p)) + 1;
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
    // extra pre-caution to make sure log never gets zero
    if (sample == static_cast<T>(0.0)) {
      sample = std::numeric_limits<T>::min();
    }
    return -1. / lambda * ::log(sample);
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
    return median + sigma * ::tan(M_PI*(uniform(generator)-0.5));
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

  template<typename U = T> 
  C10_HOST inline typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(at::CPUGenerator* generator){
    normal_distribution<double> normal(mean, stdv);
    DOUBLE2 result;
    DOUBLE2 normal_vals = normal(generator);
    result[0] = ::exp(normal_vals[0]);
    result[1] = ::exp(normal_vals[1]);
    return result;
  }

  template<typename U = T> 
  C10_HOST inline typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(at::CPUGenerator* generator){
    normal_distribution<float> normal(mean, stdv);
    FLOAT2 result;
    FLOAT2 normal_vals = normal(generator);
    result[0] = ::exp(normal_vals[0]);
    result[1] = ::exp(normal_vals[1]);
    return result;
  }

  private:
    T mean;
    T stdv;
};

} // namespace at
