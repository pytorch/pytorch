#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <stdint.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/Half.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Defining vector structs for CPU
#define UINT4 uint4
#define UINT2 uint2
#define DOUBLE2 double2
#define FLOAT2 float2
#define PHILOX_INLINE __forceinline__
#else
#define UINT4 uint4_cpu
#define UINT2 uint2_cpu
#define DOUBLE2 double2_cpu
#define FLOAT2 float2_cpu
#define PHILOX_INLINE __inline__
#endif
#define C10_HOST_DEVICE_INLINE C10_HOST_DEVICE PHILOX_INLINE

typedef struct uint4_cpu {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
} uint4_cpu;

typedef struct uint2_cpu {
  unsigned int x;
  unsigned int y;
} uint2_cpu;

typedef struct double2_cpu {
  double x;
  double y;
} double2_cpu;

typedef struct float2_cpu {
  float x;
  float y;
} float2_cpu;

namespace at {

/*
* Philox Engine implementation
* Originally implemented in PyTorch's fusion compiler
* Refer to: http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
* for details regarding the engine.
*
* The Philox engine is currently used in CUDA distributions
* kernels as its random engine. 
* 
* It takes a seed value, a subsequeunce
* for starting the generation and an offset for the sequence.
*
* Think of this engine as an algorithm producing a huge array. We are 
* parallelizing this array by partitioning the huge array and assigning 
* a thread index to each partition. In other words, each seed value 
* (there are 2^64 possible seed values) gives a sub array of size 
* 2^128 (each element in that array is a 128 bit number). Reasoning
* behind the array being of size 2^128 is, there are 2^64 possible
* thread index value and there is an array of size 2^64 for each of
* those thread index. Hence 2^64 * 2^64 = 2^128 for each seed value.
*
* In short, this generator can produce 2^64 (seed values) * 2^128 (number
* of elements in an array given by a seed value) = 2^192 values.
*
* Arguments:
* seed:        Seed values could be any number from 0 to 2^64-1.
* subsequence: Subsequence is just the cuda thread indexing with:
*              - blockIdx.x * blockDim.x + threadIdx.x
* offset:      The offset variable in PhiloxEngine  decides how many 128-bit 
*              random numbers to skip (i.e. how many groups of 4, 32-bit numbers to skip)
*              and hence really decides the total number of randoms that can be achieved 
*              for the given subsequence.
*/

class philox_engine {
public:

  /*
  * Constructor
  */
  C10_HOST_DEVICE_INLINE philox_engine(unsigned long long seed = 67280421310721,
                                 unsigned long long subsequence = 0,
                                 unsigned long long offset = 0) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    #ifdef __CUDA_ARCH__
      counter = make_uint4(0, 0, 0, 0);
    #else
      counter = {0, 0, 0, 0};
    #endif
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset);
  }

  /*
  * Produces a unique 32-bit pseudo random number on every invocation
  */
  C10_HOST_DEVICE_INLINE unsigned long operator()() {
    if(STATE == 0) {
      UINT4 counter_ = counter;
      UINT2 key_ = key;
      for(int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A); key_.y += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret;
    switch(STATE) {
      case 0: ret = output.x; break;
      case 1: ret = output.y; break;
      case 2: ret = output.z; break;
      case 3: ret = output.w; break;
    }
    STATE = (STATE + 1) % 4;
    return ret;
  }

  /*
  * Function that Skips N 128 bit numbers in a subsequence
  */
  C10_HOST_DEVICE_INLINE void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    // if overflow in x has occured, carry over to nhi
    if (counter.x < nlo) {
      nhi++;
      // if overflow in nhi has occured during carry over,
      // propagate that overflow to y and exit to increment z
      // otherwise return
      counter.y += nhi;
      if(nhi != 0) {
        if (nhi <= counter.y) {
          return;
        }
      }
    } else {
      // if overflow in y has occured during addition,
      // exit to increment z
      // otherwise return
      counter.y += nhi;
      if (nhi <= counter.y) {
        return;
      }
    }
    if (++counter.z)
      return;
    ++counter.w;
  }

  /*
  * Function that Skips one 128 bit number in a subsequence
  */
  C10_HOST_DEVICE_INLINE void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z) {
      return;
    }
    ++counter.w;
  }

private:
  UINT4 counter;
  UINT4 output;
  UINT2 key;
  unsigned int STATE;

  C10_HOST_DEVICE unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    #ifdef __CUDA_ARCH__
      *result_high = __umulhi(a, b);
      return a*b;
    #else
      const unsigned long long product = static_cast<unsigned long long>(a) * b;
      *result_high = (unsigned int)(product >> 32);
      return (unsigned int)(product);
    #endif
  }

  C10_HOST_DEVICE_INLINE UINT4 single_round(UINT4 ctr, UINT2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    UINT4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};

typedef philox_engine Philox4_32_10;

/*
* Distribution implementations adapted from THRandom
* Note: These functions are currently here as stubs and may be
* removed in the future. There may be better ways of implementing 
* them, especially for CUDA.
*/

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
  C10_HOST_DEVICE_INLINE typename std::enable_if<std::is_same<U, double>::value, double>::type
  operator()(Philox4_32_10& engine){
    uint64_t hi = (((uint64_t)engine()) << 32);
    uint64_t lo = (uint64_t)engine();
    uint64_t random = hi | lo;
    double x = (random & ((1ULL << 53) - 1)) * ::ldexp(1.0, -53);    
    return (x * (b - a) + a);

  }

  template<typename U = T> 
  C10_HOST_DEVICE_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, float>::type
  operator()(Philox4_32_10& engine){
    float x = (engine() & ((1 << 24) - 1)) * ::ldexp(1.0, -24);
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
  C10_HOST_DEVICE_INLINE typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(Philox4_32_10& engine){
    uniform_real_distribution<double> uniform(0.0, 1.0);
    DOUBLE2 result;
    double u2 = uniform(engine);
    double u1 = uniform(engine);
    double r = ::sqrt(-2.0 * ::log(1.0-u1));
    result.x = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result.y = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
    return result;
  }

  template<typename U = T> 
  C10_HOST_DEVICE_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(Philox4_32_10& engine){
    uniform_real_distribution<float> uniform(0.0, 1.0);
    FLOAT2 result;
    float u2 = uniform(engine);
    float u1 = uniform(engine);
    float r = ::sqrt(-2.0 * ::log(1.0-u1));
    result.x = r * ::cos(2.0 * M_PI * u2) * stdv + mean;
    result.y = r * ::sin(2.0 * M_PI * u2) * stdv + mean;
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

  C10_HOST_DEVICE_INLINE T operator()(Philox4_32_10& engine) { 
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(uniform(engine) <= p);
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

  C10_HOST_DEVICE_INLINE int operator()(Philox4_32_10& engine) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return((int)(::log(1-uniform(engine)) / ::log(p)) + 1);
  }

  private:
    T p;
};

template <typename T>
struct exponential_distribution {

  C10_HOST_DEVICE_INLINE exponential_distribution(T lambda_in) {
    lambda = lambda_in;
  }

  C10_HOST_DEVICE_INLINE T operator()(Philox4_32_10& engine) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(-1. / lambda * ::log(1-uniform(engine)));
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

  C10_HOST_DEVICE_INLINE T operator()(Philox4_32_10& engine) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return(median + sigma * ::tan(M_PI*(uniform(engine)-0.5)));
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
  C10_HOST_DEVICE_INLINE typename std::enable_if<std::is_same<U, double>::value, DOUBLE2>::type
  operator()(Philox4_32_10& engine){
    normal_distribution<double> normal(mean, stdv);
    DOUBLE2 result;
    DOUBLE2 normal_vals = normal(engine);
    result.x = ::exp(normal_vals.x);
    result.y = ::exp(normal_vals.y);
    return result;
  }

  template<typename U = T> 
  C10_HOST_DEVICE_INLINE typename std::enable_if<std::is_same<U, float>::value
                                          || std::is_same<U, at::Half>::value, FLOAT2>::type
  operator()(Philox4_32_10& engine){
    normal_distribution<float> normal(mean, stdv);
    FLOAT2 result;
    FLOAT2 normal_vals = normal(engine);
    result.x = ::exp(normal_vals.x);
    result.y = ::exp(normal_vals.y);
    return result;
  }

  private:
    T mean;
    T stdv;
};

} // namespace at
