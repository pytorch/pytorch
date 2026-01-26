/*
 * Metal kernels for distribution sampling on MPS backend.
 *
 * Implements:
 * - Standard Gamma: Marsaglia-Tsang method (2000)
 * - Poisson: Transformed rejection (Hörmann 1993) + Knuth algorithm
 *
 * Note on RNG offset management:
 * The offset is incremented conservatively by numel * MAX_RANDOMS_PER_SAMPLE
 * to account for worst-case random consumption in rejection sampling loops.
 * This ensures non-overlapping sequences between kernel launches at the cost
 * of some RNG state "wastage".
 *
 * Note on Philox RNG:
 * We use a stateful PhiloxState wrapper around the c10::metal::philox4 core
 * because rejection sampling requires consuming a variable number of random
 * values per thread. The stateless API in c10/metal/random.h
 * (rand(seed, index)) assumes a fixed index per call, but our loops need
 * to advance state dynamically.
 */

#include <c10/metal/random.h>
#include <metal_stdlib>
using namespace metal;

// Maximum iterations for rejection sampling loops to prevent GPU hangs
// These values are chosen to be extremely conservative - in practice,
// rejection sampling converges much faster for valid input parameters.
constant int MAX_GAMMA_ITERATIONS = 1000000;
constant int MAX_POISSON_ITERATIONS = 1000000;

// Get minimum positive normal value using metal::numeric_limits
template <typename T>
constexpr float type_min() {
  return metal::numeric_limits<T>::min();
}

// Philox state structure for rejection sampling.
// Uses c10::metal::philox4 core, wrapped for stateful generation.
// counter[0..1] = 64-bit offset, counter[2..3] = 64-bit subsequence (tid)
struct PhiloxState {
  uint4 counter;
  uint2 key;
  uint4 output;
  int output_idx;
};

inline void philox_next_output(thread PhiloxState& state) {
  // Use c10::metal::philox4::multiple_rounds for Philox computation
  state.output =
      c10::metal::philox4::multiple_rounds(state.counter, state.key, 10);

  // Advance counter
  state.counter.x++;
  if (state.counter.x == 0) {
    state.counter.y++;
  }

  state.output_idx = 0;
}

inline uint philox_rand32(thread PhiloxState& state) {
  if (state.output_idx >= 4) {
    philox_next_output(state);
  }
  int idx = state.output_idx++;
  // Access uint4 components by index
  switch (idx) {
    case 0:
      return state.output.x;
    case 1:
      return state.output.y;
    case 2:
      return state.output.z;
    default:
      return state.output.w;
  }
}

// Generate uniform random float in (0, 1)
inline float rand_uniform(thread PhiloxState& state) {
  uint bits = philox_rand32(state);
  constexpr uint MASK = (1u << 24) - 1;
  constexpr float DIVISOR = 1.0f / float(1u << 24);
  return (float(bits & MASK) + 0.5f) * DIVISOR;
}

// Generate standard normal using Box-Muller transform
inline float rand_normal(thread PhiloxState& state) {
  float u1 = rand_uniform(state);
  float u2 = rand_uniform(state);
  float radius = sqrt(-2.0f * log(u1));
  float theta = 2.0f * M_PI_F * u2;
  return radius * cos(theta);
}

// ============================================================================
// Standard Gamma Distribution - Marsaglia-Tsang method
// ============================================================================

inline float sample_gamma_ge1(float alpha, thread PhiloxState& state) {
  const float d = alpha - 1.0f / 3.0f;
  const float c = rsqrt(9.0f * d);

  for (int iter = 0; iter < MAX_GAMMA_ITERATIONS; iter++) {
    float x, v;
    int inner_iter = 0;
    do {
      x = rand_normal(state);
      v = 1.0f + c * x;
      inner_iter++;
    } while (v <= 0.0f && inner_iter < MAX_GAMMA_ITERATIONS);

    if (inner_iter >= MAX_GAMMA_ITERATIONS) {
      // Fallback: return mean of Gamma(alpha, 1) = alpha
      return alpha;
    }

    v = v * v * v;
    float u = rand_uniform(state);

    if (u < 1.0f - 0.0331f * (x * x) * (x * x)) {
      return d * v;
    }

    if (log(u) < 0.5f * x * x + d * (1.0f - v + log(v))) {
      return d * v;
    }
  }

  // Fallback after max iterations (should never happen for valid alpha)
  return alpha;
}

template <typename T>
inline T sample_gamma(T alpha, thread PhiloxState& state) {
  float alpha_f = float(alpha);
  float sample_f = 0.0f;

  if (alpha_f == 0.0f) {
    sample_f = 0.0f;
  } else if (alpha_f < 1.0f) {
    float u = rand_uniform(state);
    float gamma_sample = sample_gamma_ge1(alpha_f + 1.0f, state);
    sample_f = gamma_sample * pow(u, 1.0f / alpha_f);
  } else {
    sample_f = sample_gamma_ge1(alpha_f, state);
  }

  float min_val = type_min<T>();
  if (sample_f < min_val) {
    sample_f = min_val;
  }

  return T(sample_f);
}

// ============================================================================
// Poisson Distribution
// ============================================================================

inline int sample_poisson_small(float lambda, thread PhiloxState& state) {
  float enlam = exp(-lambda);
  int x = 0;
  float prod = 1.0f;

  for (int iter = 0; iter < MAX_POISSON_ITERATIONS; iter++) {
    float u = rand_uniform(state);
    prod *= u;
    if (prod > enlam) {
      x += 1;
    } else {
      return x;
    }
  }

  // Fallback: return expected value (should never reach for valid lambda < 10)
  return int(lambda);
}

inline int sample_poisson_large(float lambda, thread PhiloxState& state) {
  float slam = sqrt(lambda);
  float loglam = log(lambda);
  float b = 0.931f + 2.53f * slam;
  float a = -0.059f + 0.02483f * b;
  float invalpha = 1.1239f + 1.1328f / (b - 3.4f);
  float vr = 0.9277f - 3.6224f / (b - 2.0f);

  for (int iter = 0; iter < MAX_POISSON_ITERATIONS; iter++) {
    float u = rand_uniform(state) - 0.5f;
    float v = rand_uniform(state);
    float us = 0.5f - abs(u);
    float k = floor((2.0f * a / us + b) * u + lambda + 0.43f);

    if (k < 0.0f) {
      continue;
    }

    if ((us >= 0.07f) && (v <= vr)) {
      return int(k);
    }

    if ((us < 0.013f) && (v > us)) {
      continue;
    }

    float kp1 = k + 1.0f;
    // Stirling approximation:
    // log(k!) ≈ (k+0.5)*log(k+1) - (k+1) + log(sqrt(2*pi))
    // log(sqrt(2*pi)) = 0.918938...
    float lgamma_kp1 = (k + 0.5f) * log(kp1) - kp1 + 0.91893853320467274f;

    if ((log(v) + log(invalpha) - log(a / (us * us) + b)) <=
        (-lambda + k * loglam - lgamma_kp1)) {
      return int(k);
    }
  }

  // Fallback: return expected value (should never reach for valid lambda)
  return int(lambda);
}

template <typename T>
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

inline PhiloxState init_philox_state(constant uint* rng_state, uint tid) {
  PhiloxState state;
  state.key = uint2(rng_state[0], rng_state[1]);
  state.counter = uint4(rng_state[2], rng_state[3], tid, 0);
  state.output_idx = 4; // Force generation on first use
  return state;
}

template <typename T>
kernel void standard_gamma(
    device const T* alpha [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint* rng_state [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  PhiloxState state = init_philox_state(rng_state, tid);
  output[tid] = sample_gamma<T>(alpha[tid], state);
}

template <typename T>
kernel void poisson_kernel(
    device const T* lambda [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint* rng_state [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  PhiloxState state = init_philox_state(rng_state, tid);
  output[tid] = sample_poisson<T>(lambda[tid], state);
}

template [[host_name("standard_gamma_float")]]
kernel void standard_gamma<float>(
    device const float*,
    device float*,
    constant uint*,
    uint);

template [[host_name("standard_gamma_half")]]
kernel void standard_gamma<half>(
    device const half*,
    device half*,
    constant uint*,
    uint);

template [[host_name("standard_gamma_bfloat")]]
kernel void standard_gamma<bfloat>(
    device const bfloat*,
    device bfloat*,
    constant uint*,
    uint);

template [[host_name("poisson_float")]]
kernel void poisson_kernel<float>(
    device const float*,
    device float*,
    constant uint*,
    uint);

template [[host_name("poisson_half")]]
kernel void poisson_kernel<half>(
    device const half*,
    device half*,
    constant uint*,
    uint);

template [[host_name("poisson_bfloat")]]
kernel void poisson_kernel<bfloat>(
    device const bfloat*,
    device bfloat*,
    constant uint*,
    uint);
