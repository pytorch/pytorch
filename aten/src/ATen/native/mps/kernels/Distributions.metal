/*
 * Metal kernels for distribution sampling on MPS backend.
 * 
 * Implements:
 * - Standard Gamma: Marsaglia-Tsang method (2000)
 * - Poisson: Transformed rejection (Hörmann 1993) + Knuth algorithm
 *
 * This implementation is adapted from the NumPy/PyTorch CPU implementation.
 * See note in aten/src/ATen/native/Distributions.cpp for the original license.
 *
 * Poisson algorithm adapted from:
 * Copyright 2005 Robert Kern (robert.kern@gmail.com) - MIT License
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

// ============================================================================
// Standard Gamma Distribution - Marsaglia-Tsang method
// ============================================================================

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
    
    // Squeeze test
    if (u < 1.0f - 0.0331f * (x * x) * (x * x)) {
      return d * v;
    }
    
    // Full test
    if (log(u) < 0.5f * x * x + d * (1.0f - v + log(v))) {
      return d * v;
    }
  }
}

// Sample from Gamma(alpha, 1) for any alpha > 0
template<typename T>
inline T sample_gamma(T alpha, thread uint32_t& counter,
                      uint32_t key0, uint32_t key1,
                      uint32_t seed0, uint32_t seed1) {
  float alpha_f = float(alpha);
  
  if (alpha_f < 1.0f) {
    // For alpha < 1, use: Gamma(alpha) = Gamma(alpha + 1) * U^(1/alpha)
    float u = rand_uniform(counter, key0, key1, seed0, seed1);
    float gamma_sample = sample_gamma_ge1(alpha_f + 1.0f, counter, key0, key1, seed0, seed1);
    return T(gamma_sample * pow(u, 1.0f / alpha_f));
  } else {
    return T(sample_gamma_ge1(alpha_f, counter, key0, key1, seed0, seed1));
  }
}

// ============================================================================
// Poisson Distribution
// ============================================================================

// Poisson sampling for lambda < 10 using Knuth's algorithm
inline int sample_poisson_small(float lambda, thread uint32_t& counter,
                                 uint32_t key0, uint32_t key1,
                                 uint32_t seed0, uint32_t seed1) {
  float enlam = exp(-lambda);
  int X = 0;
  float prod = 1.0f;
  
  while (true) {
    float U = rand_uniform(counter, key0, key1, seed0, seed1);
    prod *= U;
    if (prod > enlam) {
      X += 1;
    } else {
      return X;
    }
  }
}

// Poisson sampling for lambda >= 10 using transformed rejection method (Hörmann, 1993)
inline int sample_poisson_large(float lambda, thread uint32_t& counter,
                                 uint32_t key0, uint32_t key1,
                                 uint32_t seed0, uint32_t seed1) {
  float slam = sqrt(lambda);
  float loglam = log(lambda);
  float b = 0.931f + 2.53f * slam;
  float a = -0.059f + 0.02483f * b;
  float invalpha = 1.1239f + 1.1328f / (b - 3.4f);
  float vr = 0.9277f - 3.6224f / (b - 2.0f);
  
  while (true) {
    float U = rand_uniform(counter, key0, key1, seed0, seed1) - 0.5f;
    float V = rand_uniform(counter, key0, key1, seed0, seed1);
    float us = 0.5f - abs(U);
    float k = floor((2.0f * a / us + b) * U + lambda + 0.43f);
    
    if (k < 0.0f) {
      continue;
    }
    
    if ((us >= 0.07f) && (V <= vr)) {
      return int(k);
    }
    
    if ((us < 0.013f) && (V > us)) {
      continue;
    }
    
    // lgamma approximation using Stirling's formula for GPU
    // lgamma(k+1) ≈ (k+0.5)*log(k+1) - (k+1) + 0.5*log(2*pi)
    float kp1 = k + 1.0f;
    float lgamma_kp1 = (k + 0.5f) * log(kp1) - kp1 + 0.9189385332f;  // 0.5*log(2*pi)
    
    if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <= 
        (-lambda + k * loglam - lgamma_kp1)) {
      return int(k);
    }
  }
}

// Sample from Poisson(lambda)
template<typename T>
inline T sample_poisson(T lambda, thread uint32_t& counter,
                        uint32_t key0, uint32_t key1,
                        uint32_t seed0, uint32_t seed1) {
  float lambda_f = float(lambda);
  
  if (lambda_f == 0.0f) {
    return T(0);
  } else if (lambda_f < 10.0f) {
    return T(sample_poisson_small(lambda_f, counter, key0, key1, seed0, seed1));
  } else {
    return T(sample_poisson_large(lambda_f, counter, key0, key1, seed0, seed1));
  }
}

// ============================================================================
// Kernel Implementations
// ============================================================================

template<typename T>
kernel void standard_gamma(device const T* alpha [[buffer(0)]],
                           device T* output [[buffer(1)]],
                           constant uint32_t* rng_state [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  // RNG state: [counter, seed0, seed1, key0, key1]
  uint32_t counter = rng_state[0] + tid * 10;  // offset by tid to ensure different random streams
  uint32_t seed0 = rng_state[1];
  uint32_t seed1 = rng_state[2];
  uint32_t key0 = rng_state[3];
  uint32_t key1 = rng_state[4];
  
  T alpha_val = alpha[tid];
  output[tid] = sample_gamma<T>(alpha_val, counter, key0, key1, seed0, seed1);
}

template<typename T>
kernel void poisson_kernel(device const T* lambda [[buffer(0)]],
                           device T* output [[buffer(1)]],
                           constant uint32_t* rng_state [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  // RNG state: [counter, seed0, seed1, key0, key1]
  uint32_t counter = rng_state[0] + tid * 10;  // offset by tid to ensure different random streams
  uint32_t seed0 = rng_state[1];
  uint32_t seed1 = rng_state[2];
  uint32_t key0 = rng_state[3];
  uint32_t key1 = rng_state[4];
  
  T lambda_val = lambda[tid];
  output[tid] = sample_poisson<T>(lambda_val, counter, key0, key1, seed0, seed1);
}

// Instantiate kernels for supported types
template [[host_name("standard_gamma_float")]] kernel void standard_gamma<float>(
    device const float*, device float*, constant uint32_t*, uint);
template [[host_name("standard_gamma_half")]] kernel void standard_gamma<half>(
    device const half*, device half*, constant uint32_t*, uint);
template [[host_name("standard_gamma_bfloat")]] kernel void standard_gamma<bfloat>(
    device const bfloat*, device bfloat*, constant uint32_t*, uint);

template [[host_name("poisson_float")]] kernel void poisson_kernel<float>(
    device const float*, device float*, constant uint32_t*, uint);
template [[host_name("poisson_half")]] kernel void poisson_kernel<half>(
    device const half*, device half*, constant uint32_t*, uint);
template [[host_name("poisson_bfloat")]] kernel void poisson_kernel<bfloat>(
    device const bfloat*, device bfloat*, constant uint32_t*, uint);
