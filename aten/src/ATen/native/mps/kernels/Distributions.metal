/*
 * Metal kernels for distribution sampling on MPS backend.
 * 
 * The standard gamma sampling uses the Marsaglia-Tsang method (2000)
 * "A Simple Method for Generating Gamma Variables"
 * doi:10.1145/358407.358414
 *
 * This implementation is adapted from the NumPy/PyTorch CPU implementation.
 * See note in aten/src/ATen/native/Distributions.h for the original license.
 */

#include <metal_stdlib>
using namespace metal;

// Philox4x32 random number generator constants
constant uint32_t PHILOX_M0 = 0xD2511F53;
constant uint32_t PHILOX_M1 = 0xCD9E8D57;
constant uint32_t PHILOX_W0 = 0x9E3779B9;
constant uint32_t PHILOX_W1 = 0xBB67AE85;

// Single round of Philox
inline void philox_single_round(thread uint32_t& c0, thread uint32_t& c1,
                                thread uint32_t& c2, thread uint32_t& c3,
                                uint32_t k0, uint32_t k1) {
  uint64_t prod0 = uint64_t(PHILOX_M0) * c0;
  uint64_t prod1 = uint64_t(PHILOX_M1) * c2;
  
  c0 = uint32_t(prod1 >> 32) ^ c1 ^ k0;
  c1 = uint32_t(prod1);
  c2 = uint32_t(prod0 >> 32) ^ c3 ^ k1;
  c3 = uint32_t(prod0);
}

// Philox4x32-10 generator (10 rounds)
inline void philox4x32_10(thread uint32_t& c0, thread uint32_t& c1,
                          thread uint32_t& c2, thread uint32_t& c3,
                          uint32_t k0, uint32_t k1) {
  for (int i = 0; i < 10; i++) {
    philox_single_round(c0, c1, c2, c3, k0, k1);
    k0 += PHILOX_W0;
    k1 += PHILOX_W1;
  }
}

// Generate uniform random float in (0, 1)
inline float rand_uniform(thread uint32_t& counter, uint32_t key0, uint32_t key1,
                          uint32_t seed0, uint32_t seed1) {
  uint32_t c0 = counter++;
  uint32_t c1 = seed0;
  uint32_t c2 = seed1;
  uint32_t c3 = 0;
  
  philox4x32_10(c0, c1, c2, c3, key0, key1);
  
  // Convert to float in (0, 1) - exclude 0 to avoid log(0)
  constexpr uint32_t MASK = (1u << 24) - 1;
  constexpr float DIVISOR = 1.0f / float(1u << 24);
  return (float(c0 & MASK) + 0.5f) * DIVISOR;
}

// Generate standard normal using Box-Muller transform
inline float rand_normal(thread uint32_t& counter, uint32_t key0, uint32_t key1,
                         uint32_t seed0, uint32_t seed1) {
  float u1 = rand_uniform(counter, key0, key1, seed0, seed1);
  float u2 = rand_uniform(counter, key0, key1, seed0, seed1);
  
  // Box-Muller transform
  float radius = sqrt(-2.0f * log(u1));
  float theta = 2.0f * M_PI_F * u2;
  return radius * cos(theta);
}

// Sample from Gamma distribution using Marsaglia-Tsang method
// For alpha >= 1
inline float sample_gamma_ge1(float alpha, thread uint32_t& counter,
                               uint32_t key0, uint32_t key1,
                               uint32_t seed0, uint32_t seed1) {
  float d = alpha - 1.0f / 3.0f;
  float c = 1.0f / sqrt(9.0f * d);
  
  while (true) {
    float x, v;
    do {
      x = rand_normal(counter, key0, key1, seed0, seed1);
      v = 1.0f + c * x;
    } while (v <= 0.0f);
    
    v = v * v * v;
    float u = rand_uniform(counter, key0, key1, seed0, seed1);
    float x2 = x * x;
    
    // Quick acceptance
    if (u < 1.0f - 0.0331f * x2 * x2) {
      return d * v;
    }
    
    // Slow acceptance
    if (log(u) < 0.5f * x2 + d * (1.0f - v + log(v))) {
      return d * v;
    }
  }
}

// Sample from Gamma(alpha, 1) for any alpha > 0
inline float sample_gamma(float alpha, thread uint32_t& counter,
                          uint32_t key0, uint32_t key1,
                          uint32_t seed0, uint32_t seed1) {
  if (alpha == 0.0f) {
    return 0.0f;
  }
  
  float scale = 1.0f;
  float alpha_use = alpha;
  
  // For alpha < 1, use: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
  if (alpha < 1.0f) {
    float u = rand_uniform(counter, key0, key1, seed0, seed1);
    scale = pow(u, 1.0f / alpha);
    alpha_use = alpha + 1.0f;
  }
  
  float sample = sample_gamma_ge1(alpha_use, counter, key0, key1, seed0, seed1);
  return max(FLT_MIN, scale * sample);
}

template <typename T>
kernel void standard_gamma(
    constant T* alpha [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint32_t* rng_state [[buffer(2)]],  // [counter, seed0, seed1, key0, key1]
    uint id [[thread_position_in_grid]]) {
  
  // Each thread gets unique counter based on its ID
  uint32_t counter = rng_state[0] + id * 16;  // Offset by 16 to avoid overlap
  uint32_t seed0 = rng_state[1];
  uint32_t seed1 = rng_state[2];
  uint32_t key0 = rng_state[3];
  uint32_t key1 = rng_state[4];
  
  float alpha_val = float(alpha[id]);
  float result = sample_gamma(alpha_val, counter, key0, key1, seed0, seed1);
  
  output[id] = T(result);
}

#define INSTANTIATE_GAMMA(DTYPE)                                          \
  template [[host_name("standard_gamma_" #DTYPE)]] kernel void            \
  standard_gamma<DTYPE>(                                                  \
      constant DTYPE* alpha [[buffer(0)]],                                \
      device DTYPE* output [[buffer(1)]],                                 \
      constant uint32_t* rng_state [[buffer(2)]],                         \
      uint id [[thread_position_in_grid]]);

INSTANTIATE_GAMMA(float);
INSTANTIATE_GAMMA(half);
INSTANTIATE_GAMMA(bfloat);
