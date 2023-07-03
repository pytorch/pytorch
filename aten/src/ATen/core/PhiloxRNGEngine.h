#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <stdint.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

#include <ATen/core/Array.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <cmath>

namespace at {

// typedefs for holding vector data
namespace detail {

typedef at::detail::Array<uint32_t, 4> UINT4;
typedef at::detail::Array<uint32_t, 2> UINT2;
typedef at::detail::Array<double, 2> DOUBLE2;
typedef at::detail::Array<float, 2> FLOAT2;

} // namespace detail

/**
 * Note [Philox Engine implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Originally implemented in PyTorch's fusion compiler
 * Refer to: http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 * for details regarding the engine.
 *
 * Note that currently this implementation of the philox engine is not used
 * anywhere except for tests in cpu_generator_test.cpp. However, this engine
 * will replace curandStatePhilox4_32_10_t in the future.
 *
 * The philox engine takes a seed value, a subsequeunce
 * for starting the generation and an offset for the subsequence.
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

  C10_HOST_DEVICE inline explicit philox_engine(uint64_t seed = 67280421310721,
                                 uint64_t subsequence = 0,
                                 uint64_t offset = 0) {

    reset_state(seed, subsequence);
    incr_n(offset);
  }

  C10_HOST_DEVICE inline void reset_state(uint64_t seed = 67280421310721,
                                 uint64_t subsequence = 0) {
    key_[0] = static_cast<uint32_t>(seed);
    key_[1] = static_cast<uint32_t>(seed >> 32);
    counter_ = detail::UINT4(0);
    counter_[2] = static_cast<uint32_t>(subsequence);
    counter_[3] = static_cast<uint32_t>(subsequence >> 32);
    STATE = 0;
  }

  /**
   * Set the offset field of Philox Generator to the desired offset.
   */
  C10_HOST_DEVICE inline void set_offset(uint64_t offset) {
    counter_[0] = static_cast<uint32_t>(offset);
    counter_[1] = static_cast<uint32_t>(offset >> 32);
  }

  /**
   * Gets the current offset of the Philox Generator.
   */
  C10_HOST_DEVICE uint64_t get_offset() const {
    uint64_t lo = static_cast<uint64_t>(counter_[0]);
    uint64_t hi = static_cast<uint64_t>(counter_[1]) << 32;
    return lo | hi;
  }

  /**
   * Produces a unique 32-bit pseudo random number on every invocation. Bookeeps state to avoid waste.
   */
  C10_HOST_DEVICE inline uint32_t operator()(int32_t n_rounds = 10) { // 10 here to preserve back-compat behavior
    if(STATE == 0) {
      detail::UINT4 counter = counter_;
      detail::UINT2 key = key_;
      output_ = rand(counter, key, n_rounds);
      incr();
    }
    uint32_t ret = output_[STATE];
    STATE = (STATE + 1) & 3;
    return ret;
  }

  inline float randn(uint32_t n_rounds) {
    #ifdef __CUDA_ARCH__
    AT_ASSERT(false, "Unsupported invocation of randn on CUDA");
    #endif
    if(STATE == 0) {
      detail::UINT4 counter = counter_;
      detail::UINT2 key = key_;
      output_ = rand(counter, key, n_rounds);
      incr();
    }
    // TODO(min-jean-cho) change to Polar method, a more efficient version of Box-Muller method
    // TODO(voz) We use std:: below, and thus need a separate impl for CUDA.
    float u1 = 1 - uint32_to_uniform_float(output_[0]); // uint32_to_uniform_float returns [0,1), we need (0,1] to avoid passing 0 to log.
    float u2 = 1 - uint32_to_uniform_float(output_[1]);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  }

  /**
   * Function that Skips N 128 bit numbers in a subsequence
   */
  C10_HOST_DEVICE inline void incr_n(uint64_t n) {
    uint32_t nlo = static_cast<uint32_t>(n);
    uint32_t nhi = static_cast<uint32_t>(n >> 32);
    counter_[0] += nlo;
    // if overflow in x has occurred, carry over to nhi
    if (counter_[0] < nlo) {
      nhi++;
      // if overflow in nhi has occurred during carry over,
      // propagate that overflow to y and exit to increment z
      // otherwise return
      counter_[1] += nhi;
      if(nhi != 0) {
        if (nhi <= counter_[1]) {
          return;
        }
      }
    } else {
      // if overflow in y has occurred during addition,
      // exit to increment z
      // otherwise return
      counter_[1] += nhi;
      if (nhi <= counter_[1]) {
        return;
      }
    }
    if (++counter_[2])
      return;
    ++counter_[3];
  }

  /**
   * Function that Skips one 128 bit number in a subsequence
   */
  C10_HOST_DEVICE inline void incr() {
    if (++counter_[0])
      return;
    if (++counter_[1])
      return;
    if (++counter_[2]) {
      return;
    }
    ++counter_[3];
  }

private:
  detail::UINT4 counter_;
  detail::UINT4 output_;
  detail::UINT2 key_;
  uint32_t STATE;

  C10_HOST_DEVICE inline uint32_t mulhilo32(uint32_t a, uint32_t b,
                                    uint32_t *result_high) {
    #ifdef __CUDA_ARCH__
      *result_high = __umulhi(a, b);
      return a*b;
    #else
      const uint64_t product = static_cast<uint64_t>(a) * b;
      *result_high = static_cast<uint32_t>(product >> 32);
      return static_cast<uint32_t>(product);
    #endif
  }

  C10_HOST_DEVICE inline detail::UINT4 single_round(detail::UINT4 ctr, detail::UINT2 in_key) {
    uint32_t hi0;
    uint32_t hi1;
    uint32_t lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
    uint32_t lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
    detail::UINT4 ret;
    ret[0] = hi1 ^ ctr[1] ^ in_key[0];
    ret[1] = lo1;
    ret[2] = hi0 ^ ctr[3] ^ in_key[1];
    ret[3] = lo0;
    return ret;
  }

  C10_HOST_DEVICE constexpr float uint32_to_uniform_float(uint32_t value) {
      // maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
      constexpr float scale = 4.6566127342e-10;
      return static_cast<float>(value & 0x7FFFFFFF) * scale;
  }



  C10_HOST_DEVICE inline detail::UINT4 rand(detail::UINT4& counter, detail::UINT2& key, uint32_t n_rounds) {
    for (uint32_t round = 0; round < (n_rounds - 1); round++) {
        counter = single_round(counter, key);
        key[0] += (kPhilox10A); key[1] += (kPhilox10B);
      }
    return single_round(counter, key);
  }


  static const uint32_t kPhilox10A = 0x9E3779B9;
  static const uint32_t kPhilox10B = 0xBB67AE85;
  static const uint32_t kPhiloxSA = 0xD2511F53;
  static const uint32_t kPhiloxSB = 0xCD9E8D57;
};

typedef philox_engine Philox4_32;

} // namespace at
