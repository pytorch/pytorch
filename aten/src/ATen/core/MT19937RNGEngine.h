#pragma once

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <stdint.h>
#include <cmath>

namespace at {

constexpr int MERSENNE_STATE_N = 624;
constexpr int MERSENNE_STATE_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UMASK = 0x80000000;
constexpr uint32_t LMASK = 0x7fffffff;

class mt19937_engine {
public:
  inline explicit mt19937_engine(uint32_t seed = 5489) {
    state_[0] = seed;
    for(int j = 1; j < MERSENNE_STATE_N; j++) {
      state_[j] = (1812433253 * (state_[j-1] ^ (state_[j-1] >> 30)) + j);
    }
    left_ = 1;
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
  int left_;
  uint32_t next_;
  uint32_t state_[MERSENNE_STATE_N];

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