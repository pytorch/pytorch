#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <stdint.h>
#include <cuda.h>
#include <c10/macros/Macros.h>

namespace at {
namespace cuda {

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
  C10_DEVICE inline philox_engine(unsigned long long seed,
                                 unsigned long long subsequence,
                                 unsigned long long offset) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset);
  }

  /*
  * Produces a unique 32-bit pseudo random number on every invocation
  */
  C10_DEVICE inline unsigned long operator()() {
    if(STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
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
  C10_DEVICE inline void incr_n(unsigned long long n) {
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
  C10_DEVICE inline void incr() {
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
  uint4 counter;
  uint4 output;
  uint2 key;
  unsigned int STATE;

  C10_DEVICE unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    *result_high = __umulhi(a, b);
    return a*b;
  }

  C10_DEVICE inline uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
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
* Produces a uniform distribution in the range [0,1) of type double
* Note: how to get a range of [0,1) from {0,1,..2^32-1}
* https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
*/
C10_DEVICE  __inline__ float standard_uniform_distribution(Philox4_32_10& engine) {
  const uint32_t random32_val = engine();
  float result = (random32_val & ((1ULL << 24) - 1)) * ::ldexp(1.0, -24);
  return result;
}

/*
* Produces a normal distribution given philox random, with mean = 0 and standard deviation = 1
*/
C10_DEVICE  __inline__ float2 normal_distribution(Philox4_32_10& engine) {
  // Box-Muller method (https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
  // We are not caching and just returning one of the variables.
  float2 result;
  float u2 = standard_uniform_distribution(engine);
  float u1 = standard_uniform_distribution(engine);
  float r = ::sqrtf(-2.0 * ::logf(1.0-u1));
  result.x = r * ::cosf(2.0 * M_PI * u2);
  result.y = r * ::sinf(2.0 * M_PI * u2);
  return result;
}

/*
* Produces a lognormal distribution given philox random, mean and standard deviation
*/
C10_DEVICE  __inline__ float2 lognormal_distribution(Philox4_32_10& engine, float mean, float stdv) {
  float2 result;
  result.x = ::expf((normal_distribution(engine).x * stdv) + mean);
  result.y = ::expf((normal_distribution(engine).y * stdv) + mean);
  return result;
}

} // namespace cuda
} // namespace at
