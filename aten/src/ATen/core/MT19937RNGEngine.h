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
constexpr uint32_t MATRIX_A = 0x9908b0dfUL;
constexpr uint32_t UMASK = 0x80000000UL;
constexpr uint32_t LMASK = 0x7fffffffUL;

/**
 * Note [Mt19937 Engine implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Originally implemented in: 
 * http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/MTARCOK/mt19937ar-cok.c
 * and modified with C++ constructs. Moreover the state array of the engine
 * has been modified to hold 32 bit uints instead of 64 bits.
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

  inline explicit mt19937_engine(uint64_t seed, std::array<uint32_t, MERSENNE_STATE_N*INIT_KEY_MULTIPLIER>& init_key) 
    : seed_(seed) {
    init_with_uint32(19650218);
    init_with_array(init_key);
  }

  inline uint64_t seed() const {
    return seed_;
  }

  inline uint64_t operator()() {
    uint64_t y;
    
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
  uint64_t next_;
  uint64_t state_[MERSENNE_STATE_N];

  inline void init_with_uint32(uint64_t seed) {
    state_[0] = seed & 0xffffffffUL;
    for(int j = 1; j < MERSENNE_STATE_N; j++) {
      state_[j] = (1812433253UL * (state_[j-1] ^ (state_[j-1] >> 30)) + j);
      state_[j] &= 0xffffffffUL;
    }
    left_ = 1;
  }

  inline void init_with_array(std::array<uint32_t, MERSENNE_STATE_N*INIT_KEY_MULTIPLIER>& init_key) {
    int i = 0;
    int j = 0;
    int k = MERSENNE_STATE_N > init_key.size() ? MERSENNE_STATE_N : init_key.size();
    for(; k; k--) {
      state_[i] = (state_[i] ^ ((state_[i-1] ^ (state_[i-1] >> 30)) * 1664525)) + init_key[j] + j;
      state_[i] &= 0xffffffff;
      i++; j++;
      if (i >= MERSENNE_STATE_N) { state_[0] = state_[MERSENNE_STATE_N - 1]; i = 1; }
      if (j >= init_key.size()) { j = 0; }
    }
    for (k = MERSENNE_STATE_N - 1; k; k--) {
      state_[i] = (state_[i] ^ ((state_[i-1] ^ (state_[i-1] >> 30)) * 1566083941)) - i;
      state_[i] &= 0xffffffff;
      i++;
      if (i >= MERSENNE_STATE_N) { state_[0] = state_[MERSENNE_STATE_N - 1]; i = 1; }
    }
    state_[0] = UMASK;
    left_ = 1;
  }

  inline uint64_t mix_bits(uint64_t u, uint64_t v) {
    return (u & UMASK) | (v & LMASK);
  }

  inline uint64_t twist(uint64_t u, uint64_t v) {
    return (mix_bits(u,v) >> 1) ^ (v & 1UL ? MATRIX_A : 0UL);
  }

  inline void next_state() {
    uint64_t* p = state_;
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
