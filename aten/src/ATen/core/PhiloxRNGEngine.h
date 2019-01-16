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

#include <ATen/cuda/Array.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Defining vector structs for CPU
#define PHILOX_INLINE __forceinline__
#else
#define PHILOX_INLINE __inline__
#endif
#define C10_HOST_DEVICE_INLINE C10_HOST_DEVICE PHILOX_INLINE
#define C10_HOST_INLINE C10_HOST PHILOX_INLINE
#define C10_DEVICE_INLINE C10_DEVICE PHILOX_INLINE

// typedefs for holding vector data
namespace {

typedef at::cuda::Array<unsigned int, 4> UINT4;
typedef at::cuda::Array<unsigned int, 2> UINT2;
typedef at::cuda::Array<double, 2> DOUBLE2;
typedef at::cuda::Array<float, 2> FLOAT2;

} // anonymous namespace

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
    key[0] = (unsigned int)seed;
    key[1] = (unsigned int)(seed >> 32);
    counter = UINT4(0);
    counter[2] = (unsigned int)(subsequence);
    counter[3] = (unsigned int)(subsequence >> 32);
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
        key_[0] += (kPhilox10A); key_[1] += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret;
    switch(STATE) {
      case 0: ret = output[0]; break;
      case 1: ret = output[1]; break;
      case 2: ret = output[2]; break;
      case 3: ret = output[3]; break;
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
    counter[0] += nlo;
    // if overflow in x has occured, carry over to nhi
    if (counter[0] < nlo) {
      nhi++;
      // if overflow in nhi has occured during carry over,
      // propagate that overflow to y and exit to increment z
      // otherwise return
      counter[1] += nhi;
      if(nhi != 0) {
        if (nhi <= counter[1]) {
          return;
        }
      }
    } else {
      // if overflow in y has occured during addition,
      // exit to increment z
      // otherwise return
      counter[1] += nhi;
      if (nhi <= counter[1]) {
        return;
      }
    }
    if (++counter[2])
      return;
    ++counter[3];
  }

private:
  UINT4 counter;
  UINT4 output;
  UINT2 key;
  unsigned int STATE;

  /*
  * Function that Skips one 128 bit number in a subsequence
  */
  C10_HOST_DEVICE_INLINE void incr() {
    if (++counter[0])
      return;
    if (++counter[1])
      return;
    if (++counter[2]) {
      return;
    }
    ++counter[3];
  }

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
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
    UINT4 ret;
    ret[0] = hi1 ^ ctr[1] ^ key[0];
    ret[1] = lo1;
    ret[2] = hi0 ^ ctr[3] ^ key[1];
    ret[3] = lo0;
    return ret;
  }
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};

typedef philox_engine Philox4_32_10;

} // namespace at