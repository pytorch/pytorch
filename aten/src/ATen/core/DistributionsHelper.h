#pragma once

#include <ATen/CPUGenerator.h>
#include <ATen/core/PhiloxRNGEngine.h>

namespace at {

/*
* Produces a uniform distribution in the range [0,1) of type T
* Note: how to get a range of [0,1) from {0,1,..2^32-1}
* https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
*/
template <typename T>
struct uniform_real_distribution {

  C10_HOST_DEVICE_INLINE uniform_real_distribution(T a_in, T b_in) {
    #ifdef __CUDA_ARCH__
      assert(a_in <= b_in);
      assert(b_in-a_in <= std::numeric_limits<T>::max());
    #else
      AT_ASSERT(a_in <= b_in);
      AT_ASSERT(b_in-a_in <= std::numeric_limits<T>::max());
    #endif
    a = a_in;
    b = b_in;
  }

  template<typename U = T> 
  C10_DEVICE_INLINE typename std::enable_if<std::is_same<U, double>::value, double>::type
  operator()(Philox4_32_10& engine){
    uint64_t hi = (((uint64_t)engine()) << 32);
    uint64_t lo = (uint64_t)engine();
    uint64_t random = hi | lo;
    double x = (random & ((1ULL << 53) - 1)) * ::ldexp(1.0, -53);    
    return (x * (b - a) + a);

  }

  template<typename U = T> 
  C10_DEVICE_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, float>::type
  operator()(Philox4_32_10& engine){
    float x = (engine() & ((1 << 24) - 1)) * ::ldexp(1.0, -24);
    return (x * (b - a) + a);
  }

  template<typename U = T> 
  C10_HOST_INLINE typename std::enable_if<std::is_same<U, double>::value, double>::type
  operator()(at::CPUGenerator* generator){
    double x = (generator->random64() & ((1ULL << 53) - 1)) * ::ldexp(1.0, -53);    
    return (x * (b - a) + a);

  }

  template<typename U = T> 
  C10_HOST_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, float>::type
  operator()(at::CPUGenerator* generator){
    float x = (generator->random() & ((1 << 24) - 1)) * ::ldexp(1.0, -24);
    return (x * (b - a) + a);
  }

  private:
    T a;
    T b; 
};

template <typename T>
struct normal_distribution {

  C10_HOST_DEVICE_INLINE normal_distribution(T mean_in, T stdv_in) {
    #ifdef __CUDA_ARCH__
      assert(stdv_in > 0);
    #else
      AT_ASSERT(stdv_in > 0);
    #endif
    mean = mean_in;
    stdv = stdv_in;
  }

  template<typename U = T> 
  C10_DEVICE_INLINE typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(Philox4_32_10& engine){
    uniform_real_distribution<double> uniform(0.0, 1.0);
    DOUBLE2 result;
    double u2 = uniform(engine);
    double u1 = uniform(engine);
    double r = ::sqrt(-2.0 * ::log(1.0-u1));
    result[0] = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result[1] = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
  }

  template<typename U = T> 
  C10_DEVICE_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(Philox4_32_10& engine){
    uniform_real_distribution<float> uniform(0.0, 1.0);
    FLOAT2 result;
    float u2 = uniform(engine);
    float u1 = uniform(engine);
    float r = ::sqrt(-2.0 * ::log(1.0-u1));
    result[0] = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result[1] = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
  }

  template<typename U = T> 
  C10_HOST_INLINE typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(at::CPUGenerator* generator){
    uniform_real_distribution<double> uniform(0.0, 1.0);
    DOUBLE2 result;
    double u2 = uniform(generator);
    double u1 = uniform(generator);
    double r = ::sqrt(-2.0 * ::log(1.0-u1));
    result[0] = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result[1] = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
  }

  template<typename U = T> 
  C10_HOST_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(at::CPUGenerator* generator){
    uniform_real_distribution<float> uniform(0.0, 1.0);
    FLOAT2 result;
    float u2 = uniform(generator);
    float u1 = uniform(generator);
    float r = ::sqrt(-2.0 * ::log(1.0-u1));
    result[0] = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result[1] = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
  }

  private:
    T mean;
    T stdv;
};

template <typename T>
struct bernoulli_distribution {

  C10_HOST_DEVICE_INLINE bernoulli_distribution(T p_in) {
    #ifdef __CUDA_ARCH__
      assert(p_in >= 0 && p_in <= 1);
    #else
      AT_ASSERT(p_in >= 0 && p_in <= 1);
    #endif
    p = p_in;
  }

  C10_DEVICE_INLINE T operator()(Philox4_32_10& engine) { 
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(uniform(engine) <= p);
  }

  C10_HOST_INLINE T operator()(at::CPUGenerator* generator) { 
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(uniform(generator) <= p);
  }

  private:
    T p;
};

template <typename T>
struct geometric_distribution {

  C10_HOST_DEVICE_INLINE geometric_distribution(T p_in) {
    #ifdef __CUDA_ARCH__
      assert(p_in > 0 && p_in < 1);
    #else
      AT_ASSERT(p_in > 0 && p_in < 1);
    #endif
    p = p_in;
  }

  C10_DEVICE_INLINE int operator()(Philox4_32_10& engine) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return((int)(::log(1-uniform(engine)) / ::log(p)) + 1);
  }

  C10_HOST_INLINE int operator()(at::CPUGenerator* generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return((int)(::log(1-uniform(generator)) / ::log(p)) + 1);
  }

  private:
    T p;
};

template <typename T>
struct exponential_distribution {

  C10_HOST_DEVICE_INLINE exponential_distribution(T lambda_in) {
    lambda = lambda_in;
  }

  C10_DEVICE_INLINE T operator()(Philox4_32_10& engine) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(-1. / lambda * ::log(1-uniform(engine)));
  }

  C10_HOST_INLINE T operator()(at::CPUGenerator* generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(-1. / lambda * ::log(1-uniform(generator)));
  }

  private:
    T lambda;
};

template <typename T>
struct cauchy_distribution {

  C10_HOST_DEVICE_INLINE cauchy_distribution(T median_in, T sigma_in) {
    median = median_in;
    sigma = sigma_in;
  }

  C10_DEVICE_INLINE T operator()(Philox4_32_10& engine) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(median + sigma * ::tan(M_PI*(uniform(engine)-0.5)));
  }

  C10_HOST_INLINE T operator()(at::CPUGenerator* generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(median + sigma * ::tan(M_PI*(uniform(generator)-0.5)));
  }

  private:
    T median;
    T sigma;
};

template <typename T>
struct lognormal_distribution {

  C10_HOST_DEVICE_INLINE lognormal_distribution(T mean_in, T stdv_in) {
    #ifdef __CUDA_ARCH__
      assert(stdv_in > 0);
    #else
      AT_ASSERT(stdv_in > 0);
    #endif
    mean = mean_in;
    stdv = stdv_in;
  }

  template<typename U = T> 
  C10_DEVICE_INLINE typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(Philox4_32_10& engine){
    normal_distribution<double> normal(mean, stdv);
    DOUBLE2 result;
    DOUBLE2 normal_vals = normal(engine);
    result[0] = ::exp(normal_vals[0]);
    result[1] = ::exp(normal_vals[1]);
    return result;
  }

  template<typename U = T> 
  C10_DEVICE_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(Philox4_32_10& engine){
    normal_distribution<float> normal(mean, stdv);
    FLOAT2 result;
    FLOAT2 normal_vals = normal(engine);
    result[0] = ::exp(normal_vals[0]);
    result[1] = ::exp(normal_vals[1]);
    return result;
  }

  template<typename U = T> 
  C10_HOST_INLINE typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(at::CPUGenerator* generator){
    normal_distribution<double> normal(mean, stdv);
    DOUBLE2 result;
    DOUBLE2 normal_vals = normal(generator);
    result[0] = ::exp(normal_vals[0]);
    result[1] = ::exp(normal_vals[1]);
    return result;
  }

  template<typename U = T> 
  C10_HOST_INLINE typename std::enable_if<std::is_same<U, float>::value
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
