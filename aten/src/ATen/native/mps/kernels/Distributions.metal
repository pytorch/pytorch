#include <c10/metal/indexing.h>
#include <c10/metal/random.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;

constant constexpr float eps = ::metal::numeric_limits<float>::epsilon();

// Cauchy, geometric, and log_normal each draw 4 samples per thread from a
// single Philox-4x32-10 round, matching the existing exponential / bernoulli
// pattern. This eliminates the bit-level waste that was 75% of the philox
// output in the old 1-element-per-thread implementations and pairs with the
// host-side change that advances the generator offset by ceil(N/4) instead
// of N.

template <typename T>
kernel void cauchy(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  float median = params.x;
  float sigma = params.y;
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    // Clamp to avoid tan(+- pi / 2) singularities.
    float u = clamp(
        c10::metal::detail::uint32_to_uniform_float(raw[i]), eps, 1.0f - eps);
    output[base + i] = static_cast<T>(
        median + sigma * ::metal::precise::tan(M_PI_F * (u - 0.5f)));
  }
}

template <typename T>
kernel void log_normal(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  // One Philox round yields two (u1, u2) pairs; each pair produces two
  // independent N(0, 1) samples via Box-Muller, so we get 4 normals total.
  float2 za = c10::metal::box_muller_from_philox(raw.xy);
  float2 zb = c10::metal::box_muller_from_philox(raw.zw);
  float z[4] = {za.x, za.y, zb.x, zb.y};
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    output[base + i] =
        static_cast<T>(::metal::precise::exp(params.x + params.y * z[i]));
  }
}

template <typename T>
kernel void geometric(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    // Only `log(0)` is the failure mode; `log(1) = 0` gives a valid 0 sample.
    float u =
        ::metal::max(c10::metal::detail::uint32_to_uniform_float(raw[i]), eps);
    output[base + i] =
        static_cast<T>(ceil(::metal::precise::log(u) / params.x));
  }
}

template <typename T>
kernel void exponential(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  float lambda = params.x;
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    // Only `u = 1` is the failure mode (`log(0)`); `u = 0` gives `log(1) = 0`.
    float u = ::metal::min(
        c10::metal::detail::uint32_to_uniform_float(raw[i]), 1.0f - eps);
    output[base + i] =
        static_cast<T>(-::metal::precise::log(1.0f - u) / lambda);
  }
}

// Uniform[from, to). One Philox round per 4 outputs.
template <typename T>
kernel void uniform_dist(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  float from = params.x;
  float scale = params.y - params.x;
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    float u = c10::metal::detail::uint32_to_uniform_float(raw[i]);
    output[base + i] = static_cast<T>(from + scale * u);
  }
}

// Normal(mean, std) via Box-Muller. One Philox round yields two (u1, u2)
// pairs and 4 standard normals; we then apply the affine `mean + std * z`.
template <typename T>
kernel void normal(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  float2 za = c10::metal::box_muller_from_philox(raw.xy);
  float2 zb = c10::metal::box_muller_from_philox(raw.zw);
  float z[4] = {za.x, za.y, zb.x, zb.y};
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    output[base + i] = static_cast<T>(params.x + params.y * z[i]);
  }
}

// Random integer in `[base, base + range)`. `params.x` is `base`, `params.y` is
// the range. `range == 0` is the sentinel for "full T-bit-width range" which
// we use both for the int64 `[int64_min, int64_max]` case (range = 2^64
// doesn't fit in any signed type) and for `random_(self)` on any integer dtype
// when the caller wants the full uniform output domain.
//
// Per Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3"
// (Random123, SC '11), the 128 bits a single Philox-4x32-10 round emits can be
// consumed as 4 x uint32 or 2 x uint64 with full per-bit uniformity at either
// projection. We deliberately keep the layout dtype-stable for narrow types: 4
// outputs per thread for `sizeof(T) <= 4`, 2 outputs for `sizeof(T) == 8`. Each
// narrow output takes one full `uint32` (low bits cast to T or `% range`),
// which costs some throughput vs. byte/short packing but keeps `randint_like`
// values identical across float/half/int dtypes - inductor's `fallback_random`
// equivalence tests upcast to float for the reference and compare against the
// half compiled output, and any dtype-dependent Philox stride breaks them.
template <typename T>
kernel void random_int(
    device T* output [[buffer(0)]],
    constant long2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr uint kBytes = sizeof(T);
  constexpr uint kElts = (kBytes == 8) ? 2u : 4u;

  uint base_idx = tid * kElts;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  long add = params.x;
  long range = params.y;
  uint count = min(kElts, numel - base_idx);

  for (uint i = 0; i < count; ++i) {
    long out;
    if IF_CONSTEXPR (kBytes == 8) {
      ulong v = (static_cast<ulong>(raw[2 * i]) << 32) |
          static_cast<ulong>(raw[2 * i + 1]);
      out = (range == 0) ? static_cast<long>(v)
                         : static_cast<long>(v % static_cast<ulong>(range));
    } else {
      uint v = raw[i];
      out = (range == 0) ? static_cast<long>(static_cast<int>(v))
                         : static_cast<long>(v % static_cast<uint>(range));
    }
    output[base_idx + i] = static_cast<T>(out + add);
  }
}

#define REGISTER_OP(NAME, DTYPE)                                    \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void NAME<DTYPE>( \
      device DTYPE*, constant float2&, constant long2&, constant uint&, uint)

REGISTER_OP(cauchy, float);
REGISTER_OP(cauchy, half);
REGISTER_OP(cauchy, bfloat);

REGISTER_OP(log_normal, float);
REGISTER_OP(log_normal, half);
REGISTER_OP(log_normal, bfloat);

REGISTER_OP(geometric, float);
REGISTER_OP(geometric, half);
REGISTER_OP(geometric, bfloat);
REGISTER_OP(geometric, int);
REGISTER_OP(geometric, long);
REGISTER_OP(geometric, short);
REGISTER_OP(geometric, char);
REGISTER_OP(geometric, uchar);

REGISTER_OP(uniform_dist, float);
REGISTER_OP(uniform_dist, half);
REGISTER_OP(uniform_dist, bfloat);

REGISTER_OP(normal, float);
REGISTER_OP(normal, half);
REGISTER_OP(normal, bfloat);

// `random_int` takes `long2` params (different buffer layout from the float-
// param distributions above), so it can't go through `REGISTER_OP`.
#define REGISTER_RANDOM_INT(DTYPE)                                            \
  template [[host_name("random_int_" #DTYPE)]] kernel void random_int<DTYPE>( \
      device DTYPE*, constant long2&, constant long2&, constant uint&, uint)

REGISTER_RANDOM_INT(int);
REGISTER_RANDOM_INT(long);
REGISTER_RANDOM_INT(short);
REGISTER_RANDOM_INT(char);
REGISTER_RANDOM_INT(uchar);
REGISTER_RANDOM_INT(ushort);
REGISTER_RANDOM_INT(uint);
REGISTER_RANDOM_INT(ulong);
REGISTER_RANDOM_INT(bool);
REGISTER_RANDOM_INT(float);
REGISTER_RANDOM_INT(half);
REGISTER_RANDOM_INT(bfloat);

#define REGISTER_EXPONENTIAL(DTYPE)                         \
  template [[host_name("exponential_" #DTYPE)]] kernel void \
  exponential<DTYPE>(                                       \
      device DTYPE*, constant float2&, constant long2&, constant uint&, uint)

REGISTER_EXPONENTIAL(float);
REGISTER_EXPONENTIAL(half);
REGISTER_EXPONENTIAL(bfloat);

// Bernoulli with scalar probability p. Each thread processes 4 elements,
// amortizing one Philox-4x32-10 round (4 uint32s) across the group so the
// kernel becomes bandwidth-bound rather than RNG-bound. The mask bit is
// converted to T via `c10::metal::cast_to`, which routes complex destinations
// to `T(value, 0)` — matching the previous MPSGraph behaviour where bool was
// cast to complex as "1+0j" / "0+0j".
template <typename T>
kernel void bernoulli_scalar(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  float p = params.x;
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    float u = c10::metal::detail::uint32_to_uniform_float(raw[i]);
    output[base + i] = c10::metal::cast_to<T>(u < p ? 1u : 0u);
  }
}

// Bernoulli with per-element probability tensor (float). The probability
// buffer must be flattened and have the same numel as `output`; broadcasting
// is handled host-side so this kernel is a straight zip over two contiguous
// buffers plus a single Philox round per thread.
template <typename T>
kernel void bernoulli_tensor(
    device T* output [[buffer(0)]],
    device const float* probs [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid * 4;
  uint4 raw =
      c10::metal::philox4::rand(seed_base_offset.x, seed_base_offset.y + tid);
  uint count = min(4u, numel - base);
  for (uint i = 0; i < count; ++i) {
    float u = c10::metal::detail::uint32_to_uniform_float(raw[i]);
    float p = probs[base + i];
    output[base + i] = c10::metal::cast_to<T>(u < p ? 1u : 0u);
  }
}

#define REGISTER_BERNOULLI(DTYPE)                                              \
  template [[host_name("bernoulli_scalar_" #DTYPE)]] kernel void               \
  bernoulli_scalar<DTYPE>(                                                     \
      device DTYPE*, constant float2&, constant long2&, constant uint&, uint); \
  template [[host_name("bernoulli_tensor_" #DTYPE)]] kernel void               \
  bernoulli_tensor<DTYPE>(                                                     \
      device DTYPE*,                                                           \
      device const float*,                                                     \
      constant long2&,                                                         \
      constant uint&,                                                          \
      uint)

REGISTER_BERNOULLI(float);
REGISTER_BERNOULLI(half);
REGISTER_BERNOULLI(bfloat);
REGISTER_BERNOULLI(bool);
REGISTER_BERNOULLI(uchar);
REGISTER_BERNOULLI(char);
REGISTER_BERNOULLI(short);
REGISTER_BERNOULLI(int);
REGISTER_BERNOULLI(long);
REGISTER_BERNOULLI(float2);
REGISTER_BERNOULLI(half2);

// Marsaglia & Tsang (2000) acceptance-rejection method for Gamma distribution.
// Adapted from aten/src/ATen/native/Distributions.h sample_gamma(),
// which originates from NumPy's random module (Copyright 2005 Robert Kern).
// Each thread uses a per-thread RNG offset stride to allow variable-length
// rejection loops without colliding with other threads' random streams.
constant constexpr int GAMMA_RANDOMS_STRIDE = 32;

template <typename T>
kernel void standard_gamma(
    device T* output [[buffer(0)]],
    device const T* alpha_in [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  float alpha = static_cast<float>(alpha_in[tid]);
  float scale = 1.0f;
  long base =
      seed_base_offset.y + static_cast<long>(tid) * GAMMA_RANDOMS_STRIDE;
  long seed = seed_base_offset.x;
  int rng_idx = 0;

  // Boost alpha < 1 for higher acceptance probability
  if (alpha < 1.0f) {
    if (alpha == 0.0f) {
      output[tid] = static_cast<T>(0.0f);
      return;
    }
    float u = c10::metal::rand(seed, base + rng_idx++);
    scale = ::metal::precise::pow(1.0f - u, 1.0f / alpha);
    alpha += 1.0f;
  }

  // Marsaglia & Tsang acceptance-rejection
  float d = alpha - 1.0f / 3.0f;
  float c = 1.0f / ::metal::precise::sqrt(9.0f * d);
  for (;;) {
    float x, y;
    do {
      x = c10::metal::randn(seed, base + rng_idx++);
      y = 1.0f + c * x;
    } while (y <= 0.0f);
    float v = y * y * y;
    float u = 1.0f - c10::metal::rand(seed, base + rng_idx++);
    float xx = x * x;
    if (u < 1.0f - 0.0331f * xx * xx) {
      float result = scale * d * v;
      output[tid] = static_cast<T>(max(result, FLT_MIN));
      return;
    }
    if (::metal::precise::log(u) <
        0.5f * xx + d * (1.0f - v + ::metal::precise::log(v))) {
      float result = scale * d * v;
      output[tid] = static_cast<T>(max(result, FLT_MIN));
      return;
    }
  }
}

#define REGISTER_GAMMA(DTYPE)                      \
  template [[host_name("standard_gamma_" #DTYPE)]] \
  kernel void standard_gamma<DTYPE>(               \
      device DTYPE*, device const DTYPE*, constant long2&, uint)

REGISTER_GAMMA(float);
REGISTER_GAMMA(half);
REGISTER_GAMMA(bfloat);

// Reparameterized gradient for Gamma distribution.
// Computes -(d/dalpha cdf(x;alpha)) / pdf(x;alpha).
// Adapted from aten/src/ATen/native/Distributions.h standard_gamma_grad_one().

constant constexpr float GAMMA_GRAD_COEF_UV[3][8] = {
    {0.16009398f,
     -0.094634809f,
     0.025146376f,
     -0.0030648343f,
     1.0f,
     0.32668115f,
     0.10406089f,
     0.0014179084f},
    {0.53487893f,
     0.1298071f,
     0.065735949f,
     -0.0015649758f,
     0.16639465f,
     0.020070113f,
     -0.0035938915f,
     -0.00058392623f},
    {0.040121004f,
     -0.0065914022f,
     -0.0026286047f,
     -0.0013441777f,
     0.017050642f,
     -0.0021309326f,
     0.00085092367f,
     -1.5247877e-07f},
};

template <typename T>
kernel void standard_gamma_grad(
    device T* output [[buffer(0)]],
    device const T* self_data [[buffer(1)]],
    device const T* output_data [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  float alpha = static_cast<float>(self_data[tid]);
  float x = static_cast<float>(output_data[tid]);

  // Region 1: Small x - Taylor series expansion
  if (x < 0.8f) {
    float numer = 1.0f;
    float denom = alpha;
    float series1 = numer / denom;
    float series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / static_cast<float>(i);
      denom += 1.0f;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    float pow_x_alpha = ::metal::precise::pow(x, alpha);
    float gamma_pdf =
        ::metal::precise::pow(x, alpha - 1.0f) * ::metal::precise::exp(-x);
    float gamma_cdf = pow_x_alpha * series1;
    float gamma_cdf_alpha =
        (::metal::precise::log(x) - c10::metal::digamma(alpha)) * gamma_cdf -
        pow_x_alpha * series2;
    float result = -gamma_cdf_alpha / gamma_pdf;
    output[tid] = static_cast<T>(isnan(result) ? 0.0f : result);
    return;
  }

  // Region 2: Large alpha - Rice saddle point expansion
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      float numer_1 = 1.0f + 24.0f * alpha * (1.0f + 12.0f * alpha);
      float numer_2 = 1440.0f * (alpha * alpha) +
          6.0f * x * (53.0f - 120.0f * x) - 65.0f * x * x / alpha +
          alpha * (107.0f + 3600.0f * x);
      float denom = 1244160.0f * (alpha * alpha) * (alpha * alpha);
      output[tid] = static_cast<T>(numer_1 * numer_2 / denom);
      return;
    }
    float denom = ::metal::precise::sqrt(8.0f * alpha);
    float term2 = denom / (alpha - x);
    float term3 = ::metal::precise::pow(
        x - alpha - alpha * ::metal::precise::log(x / alpha), -1.5f);
    float term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    float term1 = ::metal::precise::log(x / alpha) * term23 -
        ::metal::precise::sqrt(2.0f / alpha) * (alpha + x) /
            ((alpha - x) * (alpha - x));
    float stirling =
        1.0f + 1.0f / (12.0f * alpha) * (1.0f + 1.0f / (24.0f * alpha));
    float numer = x * term1;
    output[tid] = static_cast<T>(-stirling * numer / denom);
    return;
  }

  // Region 3: Moderate alpha - bivariate rational approximation
  float u = ::metal::precise::log(x / alpha);
  float v = ::metal::precise::log(alpha);
  float coef_v[8];
  for (int i = 0; i < 8; ++i) {
    coef_v[i] = GAMMA_GRAD_COEF_UV[0][i] +
        u * (GAMMA_GRAD_COEF_UV[1][i] + u * GAMMA_GRAD_COEF_UV[2][i]);
  }
  float p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  float q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  output[tid] = static_cast<T>(::metal::precise::exp(p / q));
}

#define REGISTER_GAMMA_GRAD(DTYPE)                      \
  template [[host_name("standard_gamma_grad_" #DTYPE)]] \
  kernel void standard_gamma_grad<DTYPE>(               \
      device DTYPE*, device const DTYPE*, device const DTYPE*, uint)

REGISTER_GAMMA_GRAD(float);
REGISTER_GAMMA_GRAD(half);
REGISTER_GAMMA_GRAD(bfloat);

// Adapted from aten/src/ATen/native/Distributions.h _beta_grad_alpha_mid(),
// _beta_grad_alpha_small() and _beta_grad_beta_small().

inline float beta_grad_alpha_small(float x, float alpha, float beta) {
  const float factor = c10::metal::digamma(alpha) -
      c10::metal::digamma(alpha + beta) - ::metal::precise::log(x);
  float numer = 1.0f;
  float series = numer / alpha * (factor + 1.0f / alpha);
  for (int i = 1; i <= 10; ++i) {
    const float casted_i = static_cast<float>(i);
    numer *= (casted_i - beta) * x / casted_i;
    const float denom = alpha + casted_i;
    series += numer / denom * (factor + 1.0f / denom);
  }
  const float result = x * ::metal::precise::pow(1.0f - x, -beta) * series;
  return isnan(result) ? 0.0f : result;
}

inline float beta_grad_beta_small(float x, float alpha, float beta) {
  const float factor =
      c10::metal::digamma(alpha + beta) - c10::metal::digamma(beta);
  float numer = 1.0f, betas = 1.0f, dbetas = 0.0f, series = factor / alpha;
  for (int i = 1; i <= 8; ++i) {
    const float casted_i = static_cast<float>(i);
    numer *= -x / casted_i;
    dbetas = dbetas * (beta - casted_i) + betas;
    betas = betas * (beta - casted_i);
    series += numer / (alpha + casted_i) * (dbetas + factor * betas);
  }
  const float result = -::metal::precise::pow(1.0f - x, 1.0f - beta) * series;
  return isnan(result) ? 0.0f : result;
}

inline float beta_grad_alpha_mid(float x, float alpha, float beta) {
  const float total = alpha + beta;
  const float mean = alpha / total;
  const float std =
      ::metal::precise::sqrt(alpha * beta / (total + 1.0f)) / total;
  if (mean - 0.1f * std <= x && x <= mean + 0.1f * std) {
    const float poly = 47 * x * (beta * beta) * (beta * beta) +
        alpha *
            ((43.0f + 20.0f * (16.0f + 27.0f * beta) * x) * (beta * beta) *
                 beta +
             alpha *
                 (3.0f * (59.0f + 180.0f * beta - 90.0f * x) * (beta * beta) +
                  alpha *
                      ((453.0f + 1620.0f * beta * (1.0f - x) - 455.0f * x) *
                           beta +
                       alpha * (8.0f * (1.0f - x) * (135.0f * beta - 11.0f)))));
    const float prefactor_num =
        (1.0f + 12.0f * alpha) * (1.0f + 12.0f * beta) / (total * total);
    const float prefactor_den =
        12960.0f * alpha * alpha * alpha * beta * beta * (1.0f + 12.0f * total);
    return prefactor_num / (1.0f - x) * poly / prefactor_den;
  }
  const float prefactor =
      -x / ::metal::precise::sqrt(2.0f * alpha * beta / total);
  const float stirling =
      (1.0f + 1.0f / (12.0f * alpha) + 1.0f / (288.0f * alpha * alpha)) *
      (1.0f + 1.0f / (12.0f * beta) + 1.0f / (288.0f * beta * beta)) /
      (1.0f + 1.0f / (12.0f * total) + 1.0f / (288.0f * total * total));
  const float term1_num = 2.0f * (alpha * alpha) * (x - 1.0f) +
      alpha * beta * (x - 1.0f) - x * (beta * beta);
  const float axbx = alpha * (x - 1.0f) + beta * x;
  const float term1_den = ::metal::precise::sqrt(2.0f * alpha / beta) *
      ::metal::precise::pow(total, 1.5f) * axbx * axbx;
  const float term1 = term1_num / term1_den;
  const float term2 = 0.5f * ::metal::precise::log(alpha / (total * x));
  const float term3_num = ::metal::precise::sqrt(8.0f * alpha * beta / total);
  const float term3_den = beta * x + alpha * (x - 1.0f);
  const float term3 = term3_num / term3_den;
  const float term4_base =
      beta * ::metal::precise::log(beta / (total * (1.0f - x))) +
      alpha * ::metal::precise::log(alpha / (total * x));
  const float term4 = ::metal::precise::pow(term4_base, -1.5f);
  const float term1234 = term1 + term2 * (term3 + (x < mean ? term4 : -term4));
  return stirling * prefactor * term1234;
}

// Adapted from aten/src/ATen/native/cuda/Distributions.cu
// launch_dirichlet_kernel().

struct dirichlet_functor {
  template <typename T>
  inline T operator()(const T gamma, const T gamma_sum) {
    auto ret_val = gamma / gamma_sum;
    auto min_val = ::metal::numeric_limits<T>::min();
    auto max_val = static_cast<T>(1) - ::metal::numeric_limits<T>::epsilon();
    ret_val = c10::metal::max(ret_val, min_val);
    ret_val = c10::metal::min(ret_val, max_val);
    return ret_val;
  }
};

REGISTER_BINARY_OP(dirichlet, float, float);

// Reparameterized gradient for Dirichlet distribution.
// Adapted from aten/src/ATen/native/Distributions.h dirichlet_grad_one().

constant constexpr float DIRICHLET_GRAD_C[2][3][3][4] = {
    {{{1.003668233f, -0.01061107488f, -0.0657888334f, 0.01201642863f},
      {0.6336835991f, -0.3557432599f, 0.05486251648f, -0.001465281033f},
      {-0.03276231906f, 0.004474107445f, 0.002429354597f, -0.0001557569013f}},
     {{0.221950385f, -0.3187676331f, 0.01799915743f, 0.01074823814f},
      {-0.2951249643f, 0.06219954479f, 0.01535556598f, 0.001550077057f},
      {0.02155310298f, 0.004170831599f, 0.001292462449f, 6.976601077e-05f}},
     {{-0.05980841433f, 0.008441916499f, 0.01085618172f, 0.002319392565f},
      {0.02911413504f, 0.01400243777f, -0.002721828457f, 0.000751041181f},
      {0.005900514878f,
       -0.001936558688f,
       -9.495446725e-06f,
       5.385558597e-05f}}},
    {{{1.0f, -0.02924021934f, -0.04438342661f, 0.007285809825f},
      {0.6357567472f, -0.3473456711f, 0.05454656494f, -0.002407477521f},
      {-0.03301322327f, 0.004845219414f, 0.00231480583f, -0.0002307248149f}},
     {{0.5925320577f, -0.1757678135f, 0.01505928619f, 0.000564515273f},
      {0.1014815858f, -0.06589186703f, 0.01272886114f, -0.0007316646956f},
      {-0.007258481865f, 0.001096195486f, 0.0003934994223f, -4.12701925e-05f}},
     {{0.06469649321f, -0.0236701437f, 0.002902096474f, -5.896963079e-05f},
      {0.001925008108f, -0.002869809258f, 0.0008000589141f, -6.063713228e-05f},
      {-0.0003477407336f,
       6.959756487e-05f,
       1.097287507e-05f,
       -1.650964693e-06f}}},
};

template <typename T>
kernel void dirichlet_grad(
    device T* output [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device const T* alpha [[buffer(2)]],
    device const T* total [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  float x_ = static_cast<float>(x[tid]);
  float alpha_ = static_cast<float>(alpha[tid]);
  float total_ = static_cast<float>(total[tid]);

  const float beta_ = total_ - alpha_;
  const float boundary = total_ * x_ * (1.0f - x_);

  if (x_ <= 0.5f && boundary < 2.5f) {
    output[tid] = static_cast<T>(beta_grad_alpha_small(x_, alpha_, beta_));
    return;
  }

  if (x_ >= 0.5f && boundary < 0.75f) {
    output[tid] =
        static_cast<T>(-beta_grad_beta_small(1.0f - x_, beta_, alpha_));
    return;
  }

  // Use an asymptotic approximation when alpha and (total - alpha) are both
  // large.
  if (alpha_ > 6.0f && beta_ > 6.0f) {
    output[tid] = static_cast<T>(beta_grad_alpha_mid(x_, alpha_, beta_));
    return;
  }

  // Use a rational correction to an analytic approximation.
  const float u = ::metal::precise::log(x_);
  const float a = ::metal::precise::log(alpha_) - u;
  const float b = ::metal::precise::log(total_) - a;
  const float pow_u[3] = {1.0f, u, u * u};
  const float pow_a[3] = {1.0f, a, a * a};
  float p = 0.0f;
  float q = 0.0f;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const float ua = pow_u[i] * pow_a[j];
      p += ua *
          (DIRICHLET_GRAD_C[0][i][j][0] +
           b *
               (DIRICHLET_GRAD_C[0][i][j][1] +
                b *
                    (DIRICHLET_GRAD_C[0][i][j][2] +
                     b * DIRICHLET_GRAD_C[0][i][j][3])));
      q += ua *
          (DIRICHLET_GRAD_C[1][i][j][0] +
           b *
               (DIRICHLET_GRAD_C[1][i][j][1] +
                b *
                    (DIRICHLET_GRAD_C[1][i][j][2] +
                     b * DIRICHLET_GRAD_C[1][i][j][3])));
    }
  }
  const float approx =
      x_ * (c10::metal::digamma(total_) - c10::metal::digamma(alpha_)) / beta_;
  output[tid] = static_cast<T>(p / q * approx);
  return;
}

#define REGISTER_DIRICHLET_GRAD(DTYPE)             \
  template [[host_name("dirichlet_grad_" #DTYPE)]] \
  kernel void dirichlet_grad<DTYPE>(               \
      device DTYPE*,                               \
      device const DTYPE*,                         \
      device const DTYPE*,                         \
      device const DTYPE*,                         \
      uint)

REGISTER_DIRICHLET_GRAD(float);
REGISTER_DIRICHLET_GRAD(half);
REGISTER_DIRICHLET_GRAD(bfloat);
