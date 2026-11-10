#include <ATen/native/mps/kernels/FFT.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

// Stockham FFT with four-step twiddles and chirps. F holds kFFT* variant bits.

inline float2 mul_i(float2 a) {
  return float2(-a.y, a.x);
}

// cos and sin of 2 pi k / R for the radix-3, -5 and -7 butterflies, whose
// twiddles are W_R^k = (cos, -sin); inverse transforms flip the sign of sin
constexpr constant float kSin2Pi_3 = 0.86602540378443865f;
constexpr constant float kCos2Pi_5 = 0.30901699437494742f;
constexpr constant float kCos4Pi_5 = -0.80901699437494742f;
constexpr constant float kSin2Pi_5 = 0.95105651629515357f;
constexpr constant float kSin4Pi_5 = 0.58778525229247313f;
constexpr constant float kCos2Pi_7 = 0.62348980185873353f;
constexpr constant float kCos4Pi_7 = -0.22252093395631440f;
constexpr constant float kCos6Pi_7 = -0.90096886790241913f;
constexpr constant float kSin2Pi_7 = 0.78183148246802980f;
constexpr constant float kSin4Pi_7 = 0.97492791218182360f;
constexpr constant float kSin6Pi_7 = 0.43388373911755812f;

// W_16^m = exp(-i pi m / 8) = (cos(pi m / 8), -sin(pi m / 8)) for m < 10: the
// twiddles of the radix-8 and -16 composites (W_8^m = W_16^(2 m))
constexpr constant float kCosPi_8 = 0.92387953251128676f;
constexpr constant float kSinPi_8 = 0.38268343236508977f;
constexpr constant float kCosPi_4 = 0.70710678118654752f;
constant float2 kW16[] = {
    float2(1.0f, 0.0f),
    float2(kCosPi_8, -kSinPi_8),
    float2(kCosPi_4, -kCosPi_4),
    float2(kSinPi_8, -kCosPi_8),
    float2(0.0f, -1.0f),
    float2(-kSinPi_8, -kCosPi_8),
    float2(-kCosPi_4, -kCosPi_4),
    float2(-kCosPi_8, -kSinPi_8),
    float2(-1.0f, 0.0f),
    float2(-kCosPi_8, kSinPi_8)};

template <uint R> // First factor of a composite radix; 1 if prime
constexpr uint comp_r1() {
  return R > 7 && R % 4 == 0 ? 4 : R > 7 && R % 3 == 0 ? 3 : R == 25 ? 5 : 1;
}

// Radix-R butterfly over v[0], v[S], ... in place (composite r1 r2: X[k1 + r1
// k2] lands at v[(k2 + r2 k1) S]); tw holds W_R at stride ts
template <uint R, uint S>
inline void bfly(thread float2* v, bool inv, constant float2* tw, uint ts) {
  if (R == 2) {
    const float2 a = v[0], b = v[S];
    v[0] = a + b;
    v[S] = a - b;
  } else if (R == 3) {
    const float2 s = v[S] + v[2 * S], d = v[S] - v[2 * S], t = v[0] - 0.5f * s;
    const float2 u = (inv ? kSin2Pi_3 : -kSin2Pi_3) * mul_i(d);
    v[0] = v[0] + s;
    v[S] = t + u;
    v[2 * S] = t - u;
  } else if (R == 4) {
    const float2 a0 = v[0] + v[2 * S], a1 = v[0] - v[2 * S],
                 a2 = v[S] + v[3 * S];
    const float2 a3 = inv ? mul_i(v[S] - v[3 * S]) : -mul_i(v[S] - v[3 * S]);
    v[0] = a0 + a2;
    v[2 * S] = a0 - a2;
    v[S] = a1 + a3;
    v[3 * S] = a1 - a3;
  } else if (R == 5) {
    const float2 s1 = v[S] + v[4 * S], d1 = v[S] - v[4 * S],
                 s2 = v[2 * S] + v[3 * S], d2 = v[2 * S] - v[3 * S];
    const float sg = inv ? 1.0f : -1.0f;
    const float sn1 = sg * kSin2Pi_5, sn2 = sg * kSin4Pi_5;
    const float2 t1 = v[0] + kCos2Pi_5 * s1 + kCos4Pi_5 * s2,
                 t2 = v[0] + kCos4Pi_5 * s1 + kCos2Pi_5 * s2;
    const float2 u1 = mul_i(sn1 * d1 + sn2 * d2),
                 u2 = mul_i(sn2 * d1 - sn1 * d2);
    v[0] = v[0] + s1 + s2;
    v[S] = t1 + u1;
    v[4 * S] = t1 - u1;
    v[2 * S] = t2 + u2;
    v[3 * S] = t2 - u2;
  } else if (R == 7) {
    const float2 s1 = v[S] + v[6 * S], d1 = v[S] - v[6 * S],
                 s2 = v[2 * S] + v[5 * S], d2 = v[2 * S] - v[5 * S];
    const float2 s3 = v[3 * S] + v[4 * S], d3 = v[3 * S] - v[4 * S];
    const float sg = inv ? 1.0f : -1.0f;
    const float n1 = sg * kSin2Pi_7, n2 = sg * kSin4Pi_7, n3 = sg * kSin6Pi_7;
    const float2 t1 = v[0] + kCos2Pi_7 * s1 + kCos4Pi_7 * s2 + kCos6Pi_7 * s3,
                 t2 = v[0] + kCos4Pi_7 * s1 + kCos6Pi_7 * s2 + kCos2Pi_7 * s3,
                 t3 = v[0] + kCos6Pi_7 * s1 + kCos2Pi_7 * s2 + kCos4Pi_7 * s3;
    const float2 u1 = mul_i(n1 * d1 + n2 * d2 + n3 * d3),
                 u2 = mul_i(n2 * d1 - n3 * d2 - n1 * d3),
                 u3 = mul_i(n3 * d1 - n1 * d2 + n2 * d3);
    v[0] = v[0] + s1 + s2 + s3;
    v[S] = t1 + u1;
    v[6 * S] = t1 - u1;
    v[2 * S] = t2 + u2;
    v[5 * S] = t2 - u2;
    v[3 * S] = t3 + u3;
    v[4 * S] = t3 - u3;
  } else if (comp_r1<R>() > 1) {
    constexpr uint r1 = comp_r1<R>(), r2 = R / r1;

    for (uint n2 = 0; n2 < r2; n2++) {
      bfly<r1, S * r2>(v + n2 * S, inv, tw, ts);
    }

    for (uint n2 = 1; n2 < r2; n2++) {
      for (uint k1 = 1; k1 < r1; k1++) {
        const uint m = n2 * k1; // W_R^m = W_16^(16 m / R)
        const float2 w = R % 8 == 0 ? kW16[16 / R * m] : tw[m * ts];
        v[(n2 + r2 * k1) * S] = mul(v[(n2 + r2 * k1) * S], inv ? conj(w) : w);
      }
    }

    for (uint k1 = 0; k1 < r1; k1++) {
      bfly<r2, S>(v + r2 * k1 * S, inv, tw, ts);
    }
  }
}

template <uint R> // Composite butterflies leave X[k1 + r1 k2] at k2 + r2 k1
inline uint opos(uint r) {
  constexpr uint r1 = comp_r1<R>();
  return r1 == 1 ? r : (r % r1) * (R / r1) + r / r1;
}

template <class index_t>
inline vec<index_t, 2> batch_off(
    constant FFTParams<index_t>& p,
    uint y,
    uint z,
    uint d0) {
  vec<index_t, 2> off =
      index_t(y) * vec<index_t, 2>(p.in_strides[d0], p.out_strides[d0]);
  for (uint d = d0 + 1; d < p.ndim; z /= p.sizes[d++]) {
    off += index_t(z % p.sizes[d]) *
        vec<index_t, 2>(p.in_strides[d], p.out_strides[d]);
  }
  return off;
}

inline float2 twid(device const float2* tt, uint m) {
  return mul(
      tt[m & (kFFTTwiddleBlock - 1)],
      tt[kFFTTwiddleBlock + m / kFFTTwiddleBlock]);
}

inline uint pad(uint i, uint sh) { // Padding separates banks at pow2 strides
  return i + (i >> sh);
}

template <class I, class O>
struct Bufs {
  device const I* in;
  device O* out;
  device const float2 *cl, *cs, *tt;
  constant float2* tw;
  threadgroup float2* buf;
  uint i0; // four-step slice index
};

template <bool G, class I, class O> // Stage twiddle W^m (two-level table if G)
inline float2 wget(Bufs<I, O> b, uint m) {
  return G ? twid(b.tt, m) : b.tw[m];
}

template <class I, class O, uint F, bool C, class index_t>
inline float2 load_pt(Bufs<I, O> b, constant FFTParams<index_t>& p, uint e) {
  const index_t s = C ? 1 : p.in_stride;

  if (!(F & kFFTExtendedLoads)) {
    return cast_to<float2>(b.in[index_t(e) * s]);
  }

  const uint i0e = p.ld_mul > 1 ? b.i0 : 0, f = e * p.ld_mul + i0e;
  if (f >= p.n_sig) {
    return 0.0f;
  }

  const float2 x = cast_to<float2>(b.in[index_t(e) * s]);
  return b.cl ? mul(x, b.cl[f]) : x;
}

template <class I, class O, uint F, bool C, class index_t>
inline void store_pt(
    Bufs<I, O> b,
    constant FFTParams<index_t>& p,
    uint k,
    float2 v,
    float2 w) {
  constexpr bool extended = F & kFFTExtendedStores;
  v *= p.scale;
  const index_t s = C ? 1 : p.out_stride;

  v = (F & kFFTFourStepTwiddles) ? mul(v, (F & kFFTInverse) ? conj(w) : w) : v;
  const uint kf = extended && p.out_k_mul > 1 ? k * p.out_k_mul + b.i0 : k;
  if ((F & kFFTGlobalStage) || kf < p.n_out) {
    b.out[index_t(k) * s] = cast_to<O>(extended && b.cs ? mul(v, b.cs[kf]) : v);
  }
}

template <bool P2, bool G>
inline uint mod_ns(uint j, uint ns, float inv_ns) {
  if (P2) {
    return j & (ns - 1);
  }

  int k = int(j) - int(ns) * int(float(j) * inv_ns);
  // Global stages reach j = 2^31, past float's exact integers: a second step
  // on the small remainder makes it exact
  if (G) {
    k -= int(ns) * int(float(k) * inv_ns);
  }
  k = k < 0 ? k + int(ns) : k;
  return k >= int(ns) ? k - ns : k;
}

// The radix-R butterflies of stage s (Stockham stride Ns), `butterflies` per
// thread; smem exchanges stages. global: one global-memory stage.
template <
    uint R,
    bool P2,
    class I,
    class O,
    uint F,
    uint N,
    uint NS,
    bool C,
    class index_t>
inline void stage(
    Bufs<I, O> b,
    constant FFTParams<index_t>& p,
    uint s,
    uint lane,
    bool valid) {
  constexpr bool global = F & kFFTGlobalStage;
  constexpr bool inverse = F & kFFTInverse;
  constexpr bool four_step = F & kFFTFourStepTwiddles;
  constexpr uint ept = global ? R : fft_points_per_thread(N, F);
  constexpr uint butterflies = (ept + R - 1) / R;
  const uint n = N ? N : p.n, jt = N ? N / ept : p.jt, nR = n / R,
             Ns = NS ? NS : p.ns[s], twstep = NS ? N / NS / R : p.twstep[s],
             sh = C ? fft_contig_pad_shift(N) : p.padshift;
  const bool first = NS ? NS == 1 : s == 0,
             last = global || (NS ? NS * R == N : s + 1 == p.nstages);

  // Pow2 plans keep nR, Ns multiples of 16: padded smem slots are affine in r
  const uint nRv = valid ? nR : 0;
  const uint es = C ? 1 : p.es, rs = (nR + (nR >> sh)) * es,
             ws = (Ns + (Ns >> sh)) * es;
  const float inv_ns = NS ? 1.0f / float(NS) : p.inv_ns[s];
  float2 v[butterflies][R];

  for (uint m = 0; m < butterflies; m++) {
    const uint j = lane + m * jt;
    if (j < nR) {
      const uint k = mod_ns<P2, global>(j, Ns, inv_ns);
      if (first && !global) {
        for (uint r = 0; r < R; r++) {
          v[m][r] = load_pt<I, O, F, C>(b, p, j + r * nR);
        }
      } else {
        const uint bj = pad(j, sh) * es;
        v[m][0] = global ? load_pt<I, O, F, C>(b, p, j) : b.buf[bj];
        // W^(r k) from gathered W^1, W^4, W^8 and W^4 steps: gathers cost more
        const float2 w1 = wget<global>(b, k * twstep);
        float2 wa = 0.0f, wb = 0.0f;
        for (uint r = 1; r < R; r++) {
          wb = r == 4 || (r == 8 && !global) ? wget<global>(b, r * k * twstep)
              : (r & 3)                      ? wb
                        : mul(wb, wget<global>(b, 4 * k * twstep));
          wa = (r & 3) == 0  ? float2(1.0f, 0.0f)
              : (r & 3) == 1 ? w1
                             : mul(wa, w1);
          const float2 w = r < 4 ? wa : (r & 3) ? mul(wa, wb) : wb;
          const float2 x = global
              ? load_pt<I, O, F, C>(b, p, j + r * nR)
              : b.buf[P2 ? bj + r * rs : pad(j + r * nR, sh) * es];
          v[m][r] = mul(x, inverse ? conj(w) : w);
        }
      }
      bfly<R, 1>(v[m], inverse, b.tw, nR);
    }
  }

  if (last && (C || global || !p.via_smem)) {
    // Four-step twiddles W^(k i0) of the outputs k = j + r nR step by W^(nR i0)
    const float2 ws = four_step ? twid(b.tt, nR * b.i0) : 0.0f;
    for (uint m = 0; m < butterflies; m++) {
      const uint j = lane + m * jt;
      if (j < nRv) {
        float2 w = four_step ? twid(b.tt, j * b.i0) : 0.0f;
        const uint k = global ? mod_ns<P2, global>(j, Ns, inv_ns) : 0;
        const uint base = global ? (j - k) * R + k : j, step = global ? Ns : nR;
        for (uint r = 0; r < R; r++) {
          store_pt<I, O, F, C>(b, p, base + r * step, v[m][opos<R>(r)], w);
          w = mul(w, ws);
        }
      }
    }
    return;
  }

  if (!first) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m = 0; m < butterflies; m++) {
    const uint j = lane + m * jt;
    if (j < nR) {
      const uint k = mod_ns<P2, global>(j, Ns, inv_ns);
      const uint base = (j - k) * R + k, bb = pad(base, sh) * es;
      for (uint r = 0; r < R; r++) {
        b.buf[P2 ? bb + r * ws : pad(base + r * Ns, sh) * es] =
            v[m][opos<R>(r)];
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Same order as radices() in FastFourierTransform.mm: a power of two up to 16,
// then 3, then 5
constexpr uint first_radix(uint n) {
  const uint p2 = (n & -n) < 16 ? n & -n : 16;
  return p2 > 1 ? p2 : n % 3 == 0 ? 3 : n % 5 == 0 ? 5 : 1;
}

template <class I, class O, uint MASK, uint F, uint N, bool C, class index_t>
[[max_total_threads_per_threadgroup(kFFTMaxThreads)]] kernel void fft_kernel(
    device const I* in,
    device O* out,
    device const float2* cl,
    device const float2* cs,
    device const float2* tt,
    constant float2* tw,
    constant FFTParams<index_t>& p,
    threadgroup float2* buf [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr bool global = F & kFFTGlobalStage;
  constexpr bool pow2 = !global && MASK && (MASK & ~kFFTMaskP2) == 0;
  constexpr bool four_step = F & kFFTFourStepTwiddles;
  constexpr uint ept = fft_points_per_thread(N, F);
  const uint n = N ? N : p.n, tgb = C || global ? 1 : p.tgb,
             jt = N ? N / ept : p.jt;
  const uint first_b = global ? 0 : tg.x * tgb,
             nb = min(tgb, p.sizes[0] - first_b);

  const bool nfast = C || p.in_stride == 1;
  const uint b = global ? 0 : nfast ? tid / jt : tid % tgb;
  const uint lane = global ? tg.x * jt + tid : nfast ? tid % jt : tid / tgb;
  const vec<index_t, 2> off = batch_off(p, tg.y, tg.z, global ? 0 : 1);
  const bool valid = global || b < nb;
  const uint r0 = first_b + (valid ? b : 0), i0 = p.i0_dim1 ? tg.y : r0;

  device const I* ib = in + off.x + index_t(r0) * p.in_strides[0];
  device O* ob = out + off.y + index_t(r0) * p.out_strides[0];
  const Bufs<I, O> bufs{ib, ob, cl, cs, tt, tw, buf + b * p.row, i0};

  if IF_CONSTEXPR (global) {
    stage<__builtin_ctz(MASK), pow2, I, O, F, 0, 0, C>(bufs, p, 0, lane, valid);
    return;
  }

  if IF_CONSTEXPR (N) {
    constexpr uint r0 = first_radix(N), r1 = first_radix(N / r0),
                   r2 = first_radix(N / r0 / r1);
    stage<r0, pow2, I, O, F, N, 1, C>(bufs, p, 0, lane, valid);
    stage<r1, pow2, I, O, F, N, r0, C>(bufs, p, 1, lane, valid);
    if IF_CONSTEXPR (r2 > 1) {
      stage<r2, pow2, I, O, F, N, r0 * r1, C>(bufs, p, 2, lane, valid);
    }
  } else
    for (uint s = 0; s < p.nstages; s++) {
#define STAGE(RR)                                                 \
  case RR:                                                        \
    if IF_CONSTEXPR (MASK & (1u << RR)) {                         \
      stage<RR, pow2, I, O, F, 0, 0, C>(bufs, p, s, lane, valid); \
    }                                                             \
    break;
#define RADICES(_) _(1) FFT_GLOBAL_RADICES(_)
      switch (p.radix[s]) {
        RADICES(STAGE)
      } // Ascending cases compile 3% slower
#undef STAGE
    }

  if (C || global || !p.via_smem) {
    return;
  }

  const bool ofast = C || p.out_stride == 1;
  const uint bs = ofast ? tid / jt : tid % tgb,
             ls = ofast ? tid % jt : tid / tgb;
  threadgroup const float2* rb = buf + bs * p.row;
  Bufs<I, O> bo = bufs;
  bo.out = out + off.y + index_t(first_b + bs) * p.out_strides[0],
  bo.i0 = p.i0_dim1 ? tg.y : first_b + bs;

  const uint es = C ? 1 : p.es, sh = C ? fft_contig_pad_shift(N) : p.padshift;
  float2 w = four_step ? twid(tt, ls * bo.i0) : 0.0f,
         ws = four_step ? twid(tt, jt * bo.i0) : 0.0f;

  if (bs >= nb) {
    return;
  }

  for (uint k = ls; k < n; k += jt) {
    store_pt<I, O, F, false>(bo, p, k, rb[pad(k, sh) * es], w);
    w = mul(w, ws);
  }
}

#define FFT_ARGS(IN, OUT, INDEX_T)                                           \
  device const IN*, device OUT*, device const float2*, device const float2*, \
      device const float2*, constant float2*, constant FFTParams<INDEX_T>&

#define INST2(IN, OUT, F, MASK, NAME, N, C, INDEX_T)                  \
  template [[host_name("fft_" #IN "_" #OUT "_" NAME #F)]] kernel void \
  fft_kernel<IN, OUT, MASK, F, N, C, INDEX_T>(                        \
      FFT_ARGS(IN, OUT, INDEX_T), threadgroup float2*, uint, uint3);

#define INST1(...) INST2(__VA_ARGS__)
#define UNPAREN(...) __VA_ARGS__
#define INSTF(IN, OUT, F, X) INST1(IN, OUT, F, UNPAREN X)
#define INST(IK, OK, F, X) \
  INSTF(IK##_f, OK##_f, F, X) INSTF(IK##_h, OK##_h, F, X)
// Only single-pass kernels come in fp16: anything longer runs in fp32
#define INST_FP32(IK, OK, F, X) INSTF(IK##_f, OK##_f, F, X)

// Lists of (input kind, output kind, F) the planner requests, F as the literal
// sum of the kFFT* bits (FFT.h) since #F ends the kernel name: 1 inverse, 2
// packed loads, 4 four-step twiddles, 8 extended stores, 16 extended loads, 32
// global stage, 64 packed real stores
#define C_f float2
#define C_h half2

#define BASIC(_, X) _(C, C, 0, X) _(C, C, 1, X)
#define EXT1(_, X) _(C, C, 4, X) _(C, C, 5, X) _(C, C, 8, X) _(C, C, 9, X)
#define EXT2(_, X) _(C, C, 20, X) _(C, C, 21, X) _(C, C, 24, X)
#define EXT(_, X) EXT1(_, X) EXT2(_, X)

// The smallest family factoring n (radix branches cost registers); p16 serves
// lengths 1 and 16, which never take several passes. The w kernels take 64-bit
// offsets: smooth transforms spanning 2^31 elements, never chirped
#define FAMILY(MASK, NAME, N)                     \
  BASIC(INST, (MASK, NAME "_", N, false, int))    \
  EXT(INST_FP32, (MASK, NAME "_", N, false, int)) \
  BASIC(INST, (MASK, NAME "w_", N, false, long))  \
  EXT1(INST_FP32, (MASK, NAME "w_", N, false, long))
FAMILY(kFFTMaskP2, "p2", 0)
FAMILY(kFFTMaskP235, "p235", 0)
FAMILY(kFFTMaskP2357, "p2357", 0)
BASIC(INST, (kFFTMaskP16, "p16_", 0, false, int))
BASIC(INST, (kFFTMaskP16, "p16w_", 0, false, long))

#define FIXED(N) FAMILY((N & (N - 1) ? kFFTMaskP235 : kFFTMaskP2), #N, N)
FFT_FIXED_SIZES(FIXED)

#define CONTIG(N) BASIC(INST, (kFFTMaskP2, #N "c_", N, true, int))
FFT_CONTIG_SIZES(CONTIG)

#define GLOBAL1(_, X) _(float2, float2, 32, X) _(float2, float2, 33, X)
#define GLOBAL(R) GLOBAL1(INSTF, ((1u << R), "g" #R "_", 0, false, int))
FFT_GLOBAL_RADICES(GLOBAL)
