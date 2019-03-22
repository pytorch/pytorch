#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <ATen/CPUGenerator.h>
#include <ATen/AccumulateType.h>
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

/**
 * Note [Uniform Distribution Algorithm]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * The following article summarizes all the problems that arises when one tries to get
 * floats from unsigned integer: 
 * https://experilous.com/1/blog/post/perfect-fast-random-floating-point-numbers
 * 
 * There are two broadly used/debated over methods when mapping unsigned ints
 * to float:
 *        method 1: Sample from [1.0, 2.0) and then subtract 10 from it. This involves 
 *                  losing entropy so that only ints from [0,2^mantissa) are picked. 
 *                  This gives [0.0, 1.0) range of floating point values. Check the 
 *                  article for more details.
 *        method 2: Divide by maximum int and get floats in range [0.0, 1.0] and then use
 *                  rejection sampling to get rid of the 1s to make it [0.0, 1.0).
 * 
 * The article argues that method 1 gives uniformly distributed floats but involves 
 * loss of absolute precision (e.g. it is very unlikely to produce 0.03125). On the 
 * other hand, method 2 results in non-uniform clumpiness, but can produce small values.
 * 
 * In PyTorch, we have selected the uniform distribution algorithm to be method 2 and 
 * our rational is that it still conforms to definition of uniform, i.e. the number of 
 * generated numbers on the [x,x+dx] segment is proportional to dx, as long as 
 * dx >> method granularity. The granularity would be gradually increasing 
 * from 2^-32 to 2^-25 for method 2, and would stay at 2^-24 for method 1. In addition
 * to the division, we squash 1.0s to just under 1 and hence, achieve the range [0.0,1.0). 
 */
namespace at {

// Using VectorType in Box-muller derived distributions to avoid
// code duplication
template <typename T>
struct VectorType {  };

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct VectorType<half> { using type = FLOAT2; };
#endif
template <> struct VectorType<Half> { using type = FLOAT2; };
template <> struct VectorType<float> { using type = FLOAT2; };
template <> struct VectorType<double> { using type = DOUBLE2; };

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
constexpr float POW_2_32_INV = 1.0f/std::numeric_limits<uint32_t>::max();
constexpr double POW_2_64_INV = 1.0/std::numeric_limits<uint64_t>::max();

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
    // See Note [Uniform Distribution Algorithm]
    dist_acctype<T> x;
    if(std::is_same<T, double>::value) {
      x = generator->random64() * POW_2_64_INV;
    } else {
      x = generator->random() * POW_2_32_INV;
    }
    if (x == static_cast<T>(1.0)) {
      x = std::nextafter(static_cast<T>(1.0), static_cast<T>(0.0));
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

  C10_HOST inline vect_type<T> operator()(at::CPUGenerator* generator){
    uniform_real_distribution<T> uniform(0.0, 1.0);
    vect_type<T> result;
    const T theta = static_cast<T>(2.0) * static_cast<T>(M_PI) * uniform(generator);
    T u1 = uniform(generator);
    // extra pre-caution to make sure log never gets zero
    if (u1 == static_cast<T>(0.0)) {
      u1 = std::numeric_limits<T>::min();
    }
    T r = ::sqrt(static_cast<T>(-2.0) * ::log(u1));
    result[0] = r * ::cos(theta) * stdv + mean;
    result[1] = r * ::sin(theta) * stdv + mean;
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
    return static_cast<T>(-1.0) / lambda * ::log(sample);
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

  C10_HOST inline vect_type<T> operator()(at::CPUGenerator* generator){
    normal_distribution<T> normal(mean, stdv);
    vect_type<T> result;
    vect_type<T> normal_vals = normal(generator);
    result[0] = ::exp(normal_vals[0]);
    result[1] = ::exp(normal_vals[1]);
    return result;
  }

  private:
    T mean;
    T stdv;
};

} // namespace at
