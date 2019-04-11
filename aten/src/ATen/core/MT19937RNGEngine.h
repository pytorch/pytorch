#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <stdint.h>
#include <cmath>
#include <array>

namespace at {

constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr int INIT_KEY_MULTIPLIER = 3;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

/**
 * Note [Mt19937 Engine implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Originally implemented in: 
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/MTARCOK/mt19937ar-cok.c
 * and modified with C++ constructs. Moreover the state array of the engine
 * has been modified to hold 32 bit uints instead of 64 bits.
 * 
 * Note that we reimplemented mt19937 instead of using std::mt19937 because,
 * at::mt19937 turns out to be faster in the pytorch codebase. PyTorch builds with -O2
 * by default and following are the benchmark numbers (benchmark code can be found at
 * https://github.com/syed-ahmed/benchmark-rngs):
 * 
 * with -O2
 * Time to get 100000000 philox randoms with at::uniform_real_distribution = 0.462759s
 * Time to get 100000000 at::mt19937 randoms with at::uniform_real_distribution = 0.39628s
 * Time to get 100000000 std::mt19937 randoms with std::uniform_real_distribution = 0.352087s
 * Time to get 100000000 std::mt19937 randoms with at::uniform_real_distribution = 0.419454s
 * 
 * std::mt19937 is faster when used in conjuction with std::uniform_real_distribution,
 * however we can't use std::uniform_real_distribution because of this bug:
 * http://open-std.org/JTC1/SC22/WG21/docs/lwg-active.html#2524. Plus, even if we used
 * std::uniform_real_distribution and filtered out the 1's, it is a different algorithm
 * than what's in pytorch currently and that messes up the tests in tests_distributions.py.
 * The other option, using std::mt19937 with at::uniform_real_distribution is a tad bit slower
 * than at::mt19937 with at::uniform_real_distribution and hence, we went with the latter.
 * 
 * Copyright notice:
 * A C-program for MT19937, with initialization improved 2002/2/10.
 * Coded by Takuji Nishimura and Makoto Matsumoto.
 * This is a faster version by taking Shawn Cokus's optimization,
 * Matthe Bellew's simplification, Isaku Wada's real version.
 *
 * Before using, initialize the state by using init_genrand(seed) 
 * or init_by_array(init_key, key_length).
 *
 * Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote 
 *   products derived from this software without specific prior written 
 *   permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * Any feedback is very welcome.
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
 * email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
 */

class mt19937_engine {
public:
  /**
   * Note: This constructor is only used for testing this
   * implementation against std::mt19937. If you use this
   * constructor, chances are you'll encounter the same seeding
   * surprises as mentioned in this article:
   * http://www.pcg-random.org/posts/cpp-seeding-surprises.html
   */
  inline explicit mt19937_engine(uint64_t seed = 5489) : seed_(seed) {
    init_with_uint32(seed);
  }

  inline uint64_t seed() const {
    return seed_;
  }

  inline uint32_t operator()() {
    uint32_t y;
    
    if (--(left_) == 0) {
        next_state();
    }
    y = *(state_ + next_++);
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
  }

private:
  uint64_t seed_;
  int left_;
  uint32_t next_;
  uint32_t state_[MERSENNE_STATE_N];

  inline void init_with_uint32(uint64_t seed) {
    state_[0] = seed & 0xffffffff;
    for(int j = 1; j < MERSENNE_STATE_N; j++) {
      state_[j] = (1812433253 * (state_[j-1] ^ (state_[j-1] >> 30)) + j);
      state_[j] &= 0xffffffff;
    }
    left_ = 1;
  }

  inline uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u,v) >> 1) ^ (v & 1 ? MATRIX_A : 0);
  }

  inline void next_state() {
    uint32_t* p = state_;
    left_ = MERSENNE_STATE_N;
    next_ = 0;

    for(int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
      *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for(int j = MERSENNE_STATE_M; --j; p++) {
      *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], state_[0]);
  }

};

typedef mt19937_engine mt19937;

} // namespace at
