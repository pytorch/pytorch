/*
 * Metal kernels for distribution sampling on MPS backend.
 *
 * Implements:
 * - Standard Gamma: Marsaglia-Tsang method (2000)
 * - Poisson: Transformed rejection (Hörmann 1993) + Knuth algorithm
 * - Binomial: BTRD algorithm (Hörmann 1993)
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

// Type-specific minimum positive normal value for clamping gamma samples
template <typename T>
inline T type_min();

template <>
inline float type_min<float>() {
  return FLT_MIN;  // ~1.175e-38
}

template <>
inline half type_min<half>() {
  return HALF_MIN;  // ~6.1e-5
}

template <>
inline bfloat type_min<bfloat>() {
  // bfloat16 min positive normal is approximately 1.175e-38 (same exponent range as float)
  // but with less precision. Use a safe minimum.
  return bfloat(1.0e-38f);
}

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

// Philox state structure for proper 128-bit counter management
// counter[0..1] = 64-bit offset, counter[2..3] = 64-bit subsequence (tid)
struct PhiloxState {
  uint32_t counter[4];  // 128-bit counter
  uint32_t key[2];      // 64-bit key (seed)
  uint32_t output[4];   // cached random output
  int output_idx;       // index into output (0-3)
};

inline void philox_next_output(thread PhiloxState& state) {
  // Copy counter to output
  state.output[0] = state.counter[0];
  state.output[1] = state.counter[1];
  state.output[2] = state.counter[2];
  state.output[3] = state.counter[3];

  // Apply Philox4x32-10
  philox4x32_10(state.output[0], state.output[1], state.output[2], state.output[3],
                state.key[0], state.key[1]);

  // Increment 128-bit counter (only increment low 64 bits, sufficient for our use)
  state.counter[0]++;
  if (state.counter[0] == 0) {
    state.counter[1]++;
  }

  state.output_idx = 0;
}

inline uint32_t philox_rand32(thread PhiloxState& state) {
  if (state.output_idx >= 4) {
    philox_next_output(state);
  }
  return state.output[state.output_idx++];
}

// Generate uniform random float in (0, 1)
inline float rand_uniform(thread PhiloxState& state) {
  uint32_t bits = philox_rand32(state);
  // Convert to float in (0, 1) - exclude 0 to avoid log(0)
  constexpr uint32_t MASK = (1u << 24) - 1;
  constexpr float DIVISOR = 1.0f / float(1u << 24);
  return (float(bits & MASK) + 0.5f) * DIVISOR;
}

// Generate standard normal using Box-Muller transform
inline float rand_normal(thread PhiloxState& state) {
  float u1 = rand_uniform(state);
  float u2 = rand_uniform(state);

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
inline float sample_gamma_ge1(float alpha, thread PhiloxState& state) {
  float d = alpha - 1.0f / 3.0f;
  float c = 1.0f / sqrt(9.0f * d);

  while (true) {
    float x, v;
    do {
      x = rand_normal(state);
      v = 1.0f + c * x;
    } while (v <= 0.0f);

    v = v * v * v;
    float u = rand_uniform(state);

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

// Sample from Gamma(alpha, 1) for any alpha >= 0
// Matches CPU/CUDA behavior: clamps output to type_min to avoid underflow
template<typename T>
inline T sample_gamma(T alpha, thread PhiloxState& state) {
  float alpha_f = float(alpha);

  // Compute raw gamma sample in float
  float sample_f = 0.0f;

  if (alpha_f == 0.0f) {
    // Avoid division by zero in the alpha < 1 branch.
    // A raw sample of 0 will be clamped to the minimal positive value below.
    sample_f = 0.0f;
  } else if (alpha_f < 1.0f) {
    // For alpha < 1, use: Gamma(alpha) = Gamma(alpha + 1) * U^(1/alpha)
    float u = rand_uniform(state);
    float gamma_sample = sample_gamma_ge1(alpha_f + 1.0f, state);
    sample_f = gamma_sample * pow(u, 1.0f / alpha_f);
  } else {
    sample_f = sample_gamma_ge1(alpha_f, state);
  }

  // Clamp to the minimal positive normal value for T to avoid underflow to 0
  float min_val = float(type_min<T>());
  if (sample_f < min_val) {
    sample_f = min_val;
  }

  return T(sample_f);
}

// ============================================================================
// Poisson Distribution
// ============================================================================

// Poisson sampling for lambda < 10 using Knuth's algorithm
inline int sample_poisson_small(float lambda, thread PhiloxState& state) {
  float enlam = exp(-lambda);
  int X = 0;
  float prod = 1.0f;

  while (true) {
    float U = rand_uniform(state);
    prod *= U;
    if (prod > enlam) {
      X += 1;
    } else {
      return X;
    }
  }
}

// Poisson sampling for lambda >= 10 using transformed rejection method (Hörmann, 1993)
inline int sample_poisson_large(float lambda, thread PhiloxState& state) {
  float slam = sqrt(lambda);
  float loglam = log(lambda);
  float b = 0.931f + 2.53f * slam;
  float a = -0.059f + 0.02483f * b;
  float invalpha = 1.1239f + 1.1328f / (b - 3.4f);
  float vr = 0.9277f - 3.6224f / (b - 2.0f);

  while (true) {
    float U = rand_uniform(state) - 0.5f;
    float V = rand_uniform(state);
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
// Note: lambda < 0 is handled in the host-side code with TORCH_CHECK
template<typename T>
inline T sample_poisson(T lambda, thread PhiloxState& state) {
  float lambda_f = float(lambda);

  if (lambda_f <= 0.0f) {
    return T(0);
  } else if (lambda_f < 10.0f) {
    return T(sample_poisson_small(lambda_f, state));
  } else {
    return T(sample_poisson_large(lambda_f, state));
  }
}

// ============================================================================
// Kernel Implementations
// ============================================================================

// Initialize PhiloxState from RNG state buffer
// rng_state layout: [seed_lo, seed_hi, offset_lo, offset_hi]
// tid is used as subsequence to ensure independent streams per thread
inline PhiloxState init_philox_state(constant uint32_t* rng_state, uint tid) {
  PhiloxState state;
  // Key (seed)
  state.key[0] = rng_state[0];
  state.key[1] = rng_state[1];
  // Counter: offset in low bits, tid as subsequence in high bits
  state.counter[0] = rng_state[2];
  state.counter[1] = rng_state[3];
  state.counter[2] = uint32_t(tid);
  state.counter[3] = uint32_t(tid >> 32);
  // Initialize output index to force first generation
  state.output_idx = 4;
  return state;
}

template<typename T>
kernel void standard_gamma(device const T* alpha [[buffer(0)]],
                           device T* output [[buffer(1)]],
                           constant uint32_t* rng_state [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  PhiloxState state = init_philox_state(rng_state, tid);
  T alpha_val = alpha[tid];
  output[tid] = sample_gamma<T>(alpha_val, state);
}

template<typename T>
kernel void poisson_kernel(device const T* lambda [[buffer(0)]],
                           device T* output [[buffer(1)]],
                           constant uint32_t* rng_state [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  PhiloxState state = init_philox_state(rng_state, tid);
  T lambda_val = lambda[tid];
  output[tid] = sample_poisson<T>(lambda_val, state);
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

// ============================================================================
// Binomial Distribution - BTRD algorithm (Hörmann 1993)
// ============================================================================

// Binomial sampling for small n*p using inversion method
inline int sample_binomial_inversion(int n, float p, thread PhiloxState& state) {
  float q = 1.0f - p;
  float s = p / q;
  float a = float(n + 1) * s;
  float r = pow(q, float(n));
  float u = rand_uniform(state);

  int x = 0;
  while (u > r) {
    u -= r;
    x += 1;
    r *= (a / float(x) - s);
  }
  return x;
}

// Binomial sampling for larger n*p using BTRD algorithm
inline int sample_binomial_btrd(int n, float p, thread PhiloxState& state) {
  float nf = float(n);
  float q = 1.0f - p;
  float npq = nf * p * q;
  float f = sqrt(npq);
  float m = floor(nf * p + p);

  // BTRD parameters
  float p1 = floor(2.195f * f - 4.6f * q) + 0.5f;
  float xm = m + 0.5f;
  float xl = xm - p1;
  float xr = xm + p1;
  float c = 0.134f + 20.5f / (15.3f + m);
  float aa = (floor((xm - xl) / (xm - xl - (xl * p))) + 1.0f) / p1;
  float laml = aa * (xm - xl);
  float lamr = aa * (xr - xm);
  float p2 = p1 * (1.0f + 2.0f * c);
  float p3 = p2 + c / laml;
  float p4 = p3 + c / lamr;

  while (true) {
    float u = rand_uniform(state) * p4;
    float v = rand_uniform(state);
    int y;

    if (u <= p1) {
      y = int(floor(xm - p1 * v + u));
    } else if (u <= p2) {
      float x = xl + (u - p1) / c;
      v = v * c + 1.0f - abs(m - x + 0.5f) / p1;
      if (v > 1.0f) continue;
      y = int(floor(x));
    } else if (u <= p3) {
      y = int(floor(xl + log(v) / laml));
      if (y < 0) continue;
      v = v * (u - p2) * laml;
    } else {
      y = int(floor(xr - log(v) / lamr));
      if (y > n) continue;
      v = v * (u - p3) * lamr;
    }

    // Acceptance/rejection with Stirling's approximation
    float k = abs(float(y) - m);
    if (k <= 20.0f || k >= npq / 2.0f - 1.0f) {
      float s = p / q;
      float aaa = float(n + 1) * s;
      float F = 1.0f;
      if (m < float(y)) {
        for (int i = int(m) + 1; i <= y; i++) {
          F *= (aaa / float(i) - s);
        }
      } else if (m > float(y)) {
        for (int i = y + 1; i <= int(m); i++) {
          F /= (aaa / float(i) - s);
        }
      }
      if (v <= F) return y;
    } else {
      // Use Stirling's approximation
      float rho = (k / npq) * ((k * (k / 3.0f + 0.625f) + 1.0f / 6.0f) / npq + 0.5f);
      float t = -k * k / (2.0f * npq);
      float A = log(v);
      if (A < t - rho) return y;
      if (A <= t + rho) {
        // Full comparison with lgamma
        float yf = float(y);
        float x1 = yf + 1.0f;
        float f1 = m + 1.0f;
        float z = nf + 1.0f - m;
        float w = nf - yf + 1.0f;
        // Stirling approximation for lgamma
        float lgamma_x1 = (yf + 0.5f) * log(x1) - x1 + 0.9189385332f;
        float lgamma_f1 = (m + 0.5f) * log(f1) - f1 + 0.9189385332f;
        float lgamma_z = (nf - m + 0.5f) * log(z) - z + 0.9189385332f;
        float lgamma_w = (nf - yf + 0.5f) * log(w) - w + 0.9189385332f;
        if (A <= lgamma_x1 - lgamma_f1 + lgamma_z - lgamma_w + (m - yf) * log(p / q)) {
          return y;
        }
      }
    }
  }
}

// Sample from Binomial(n, p)
inline int sample_binomial(int count, float prob, thread PhiloxState& state) {
  if (count == 0 || prob == 0.0f) {
    return 0;
  }
  if (prob == 1.0f) {
    return count;
  }

  // Symmetry: if p > 0.5, sample Binomial(n, 1-p) and return n - result
  float p = prob;
  bool flipped = false;
  if (p > 0.5f) {
    p = 1.0f - p;
    flipped = true;
  }

  int result;
  float np = float(count) * p;

  // Use inversion for small np, BTRD for larger np
  if (np < 10.0f) {
    result = sample_binomial_inversion(count, p, state);
  } else {
    result = sample_binomial_btrd(count, p, state);
  }

  if (flipped) {
    return count - result;
  }
  return result;
}

template<typename T>
kernel void binomial_kernel(device const T* count [[buffer(0)]],
                            device const T* prob [[buffer(1)]],
                            device T* output [[buffer(2)]],
                            constant uint32_t* rng_state [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {
  PhiloxState state = init_philox_state(rng_state, tid);
  int n = int(count[tid]);
  float p = float(prob[tid]);
  output[tid] = T(sample_binomial(n, p, state));
}

template [[host_name("binomial_float")]] kernel void binomial_kernel<float>(
    device const float*, device const float*, device float*, constant uint32_t*, uint);
template [[host_name("binomial_half")]] kernel void binomial_kernel<half>(
    device const half*, device const half*, device half*, constant uint32_t*, uint);
template [[host_name("binomial_bfloat")]] kernel void binomial_kernel<bfloat>(
    device const bfloat*, device const bfloat*, device bfloat*, constant uint32_t*, uint);
