#include <ATen/native/mps/kernels/FFT.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

// Stockham FFT with fused real packing, four-step twiddles and chirps. F holds
// kFFT* variant bits.

inline float2 mul_i(float2 a) {
  return float2(-a.y, a.x);
}

constant float W16[] = { // W_16^m for m < 10, as (re, im) pairs
    1.0f,          0.0f,          0.923879533f,  -0.382683432f, 0.707106781f,
    -0.707106781f, 0.382683432f,  -0.923879533f, 0.0f,          -1.0f,
    -0.382683432f, -0.923879533f, -0.707106781f, -0.707106781f, -0.923879533f,
    -0.382683432f, -1.0f,         0.0f,          -0.923879533f, 0.382683432f};

// Radix-R butterfly over v[0], v[S], ... in place; radices 8 and 16 are 4 x R2
// composites whose X[k1 + 4 k2] lands at v[(k2 + R2 k1) S]
template <uint R, uint S>
inline void bfly(thread float2* v, bool inv) {
  if (R == 2) {
    const float2 a = v[0], b = v[S];
    v[0] = a + b;
    v[S] = a - b;
  } else if (R == 3) {
    const float2 s = v[S] + v[2 * S], d = v[S] - v[2 * S], t = v[0] - 0.5f * s;
    // +-sin(2 pi / 3)
    const float2 u =
        (inv ? 0.86602540378443865f : -0.86602540378443865f) * mul_i(d);
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
    // cos(2 pi k / 5) and sin(2 pi k / 5) for k = 1, 2
    const float c1 = 0.30901699437494742f, c2 = -0.80901699437494742f,
                s = inv ? 1.0f : -1.0f;
    const float sn1 = s * 0.95105651629515357f, sn2 = s * 0.58778525229247313f;
    const float2 t1 = v[0] + c1 * s1 + c2 * s2, t2 = v[0] + c2 * s1 + c1 * s2;
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
    // cos(2 pi k / 7) and sin(2 pi k / 7) for k = 1, 2, 3
    const float c1 = 0.62348980185873353f, c2 = -0.22252093395631440f,
                c3 = -0.90096886790241913f, sg = inv ? 1.0f : -1.0f;
    const float n1 = sg * 0.78183148246802980f, n2 = sg * 0.97492791218182360f,
                n3 = sg * 0.43388373911755812f;
    const float2 t1 = v[0] + c1 * s1 + c2 * s2 + c3 * s3,
                 t2 = v[0] + c2 * s1 + c3 * s2 + c1 * s3,
                 t3 = v[0] + c3 * s1 + c1 * s2 + c2 * s3;
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
  } else if (R >= 8) {
    constexpr uint R2 = R / 4;

    for (uint n2 = 0; n2 < R2; n2++) {
      bfly<4, S * R2>(v + n2 * S, inv);
    }

    for (uint n2 = 1; n2 < R2; n2++) {
      for (uint k1 = 1; k1 < 4; k1++) {
        // W_R^m = W_16^(16 m / R), and W16 holds two floats per root
        const uint i = 32 / R * n2 * k1;
        const float2 w = float2(W16[i], W16[i + 1]);
        v[(n2 + R2 * k1) * S] = mul(v[(n2 + R2 * k1) * S], inv ? conj(w) : w);
      }
    }

    for (uint k1 = 0; k1 < 4; k1++) {
      bfly<R2, S>(v + R2 * k1 * S, inv);
    }
  }
}

template <uint R> // Composite butterflies leave X[k1 + 4 k2] at k2 + R2 k1
inline uint opos(uint r) {
  return R < 8 ? r : (r % 4) * (R / 4) + r / 4;
}

inline int2 batch_off(constant FFTParams& p, uint y, uint z) {
  int2 off = int(y) * int2(p.in_strides[1], p.out_strides[1]);
  for (uint d = 2; d < p.ndim; z /= p.sizes[d++]) {
    off += int(z % p.sizes[d]) * int2(p.in_strides[d], p.out_strides[d]);
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

typedef constant FFTParams& Params;

template <class I, class O, uint F, uint N, bool C>
inline float2 load_pt(Bufs<I, O> b, Params p, uint e) {
  const uint n = N ? N : p.n;
  const int s = C ? 1 : p.in_stride;
  constexpr bool packed = F & kFFTPackedLoads;

  if (!is_complex_v<I> && packed) {
    const float2 x = cast_to<float2>(b.in[int(2 * e) * s]);
    return float2(x.x, cast_to<float2>(b.in[int(2 * e + 1) * s]).x);
  }

  if (is_complex_v<I> && packed) {
    float2 a = cast_to<float2>(b.in[int(e) * s]),
           c = cast_to<float2>(b.in[int(n - e) * s]);
    // DC and Nyquist are real; discard imaginary parts like pocketfft.
    a.y = e ? a.y : 0.0f;
    c.y = e ? c.y : 0.0f;
    return (a + conj(c)) + mul_i(mul(a - conj(c), conj(b.tw[e])));
  }

  if (!(F & kFFTExtendedLoads)) {
    const bool mirror = is_complex_v<I> && e >= p.n_in;
    const float2 x = cast_to<float2>(b.in[int(mirror ? n - e : e) * s]);
    return mirror ? conj(x) : x;
  }

  const uint i0e = p.ld_mul > 1 ? b.i0 : 0, f = e * p.ld_mul + i0e;
  if (f >= p.n_sig) {
    return 0.0f;
  }

  const float2 x = is_complex_v<I> && p.n_in < p.n_sig && f >= p.n_in
      ? conj(cast_to<float2>(b.in[int(p.n_sig - f - i0e) * p.in_unit]))
      : cast_to<float2>(b.in[int(e) * s]);
  return b.cl ? mul(x, b.cl[f]) : x;
}

template <typename T>
using packed2 = conditional_t<is_same_v<T, half>, packed_half2, packed_float2>;

template <class I, class O, uint F, bool C>
inline void store_pt(Bufs<I, O> b, Params p, uint k, float2 v, float2 w) {
  constexpr bool extended = F & kFFTExtendedStores;
  v *= p.scale;
  const int s = C ? 1 : p.out_stride;

  if (F & kFFTPackedRealStores) {
    if (s == 1) {
      reinterpret_cast<device packed2<O>*>(b.out)[k] = packed2<O>(v);
    } else {
      b.out[int(2 * k) * s] = O(v.x);
      b.out[int(2 * k + 1) * s] = O(v.y);
    }
    return;
  }

  v = (F & kFFTFourStepTwiddles) ? mul(v, (F & kFFTInverse) ? conj(w) : w) : v;
  const uint kf = extended && p.out_k_mul > 1 ? k * p.out_k_mul + b.i0 : k;
  if (kf < p.n_out) {
    b.out[int(k) * s] = cast_to<O>(extended && b.cs ? mul(v, b.cs[kf]) : v);
  }
}

// Unpack z = (Z[k], Z[n-k]): X[k] = E + T, X[n-k] = conj(E - T), w = W_2n^k
template <class I, class O, uint N, bool C>
inline void unpack(Bufs<I, O> b, Params p, uint k, float2 w, float4 z) {
  const uint n = N ? N : p.n, os = C ? 1 : p.out_stride;
  const float2 a = z.xy, c = z.zw;
  const bool two = n - k != k;
  const float2 e = 0.5f * (a + conj(c)), t = mul(w, -0.5f * mul_i(a - conj(c)));
  const float2 x = (e + t) * p.scale, y = conj(e - t) * p.scale;

  b.out[int(k) * os] = cast_to<O>(x);
  if (two && n - k < p.n_out) {
    b.out[int(n - k) * os] = cast_to<O>(y);
  }

  if (p.n_out > n + 1 && k) {
    b.out[int(2 * n - k) * os] = cast_to<O>(conj(x));
    if (two) {
      b.out[int(n + k) * os] = cast_to<O>(conj(y));
    }
  }
}

template <bool P2> // j % ns: j stays below 2^13, exact in float
inline uint mod_ns(uint j, uint ns, float inv_ns) {
  if (P2) {
    return j & (ns - 1);
  }

  int k = int(j) - int(ns) * int(float(j) * inv_ns);
  k = k < 0 ? k + int(ns) : k;
  return k >= int(ns) ? k - ns : k;
}

// M radix-R butterflies per thread for stage s (Stockham stride Ns); smem
// exchanges stages and holds the final real unpack
template <uint R, bool P2, class I, class O, uint F, uint N, uint NS, bool C>
inline void stage(Bufs<I, O> b, Params p, uint s, uint lane, bool valid) {
  constexpr bool INV = F & kFFTInverse;
  constexpr bool packed = F & kFFTPackedLoads;
  constexpr bool four_step = F & kFFTFourStepTwiddles;
  constexpr bool UNPACK = !is_complex_v<I> && packed;
  constexpr uint EPT = fft_points_per_thread(N, F);
  constexpr uint M = (EPT + R - 1) / R;
  const uint n = N ? N : p.n, J = N ? N / EPT : p.jt, nR = n / R,
             Ns = NS ? NS : p.ns[s],
             twstep = NS ? N / NS / R * (packed ? 2 : 1) : p.twstep[s],
             sh = C ? fft_contig_pad_shift(N) : p.padshift;
  const bool first = NS ? NS == 1 : s == 0,
             last = NS ? NS * R == N : s + 1 == p.nstages;

  // Pow2 plans keep nR, Ns multiples of 16: padded smem slots are affine in r
  const uint nRv = valid ? nR : 0;
  const uint es = C ? 1 : p.es, rs = (nR + (nR >> sh)) * es,
             ws = (Ns + (Ns >> sh)) * es;
  const float inv_ns = NS ? 1.0f / float(NS) : p.inv_ns[s];
  float2 v[M][R];

  for (uint m = 0; m < M; m++) {
    const uint j = lane + m * J;
    if (j < nR) {
      const uint k = mod_ns<P2>(j, Ns, inv_ns);
      if (first) {
        for (uint r = 0; r < R; r++) {
          v[m][r] = load_pt<I, O, F, N, C>(b, p, j + r * nR);
        }
      } else {
        const uint bj = pad(j, sh) * es;
        v[m][0] = b.buf[bj];
        // W^(r k) from gathered W^1, W^4, W^8 and W^4 steps: gathers cost more
        const float2 w1 = b.tw[k * twstep];
        float2 wa = 0.0f, wb = 0.0f;
        for (uint r = 1; r < R; r++) {
          wb = r == 4 || r == 8 ? b.tw[r * k * twstep]
              : (r & 3)         ? wb
                                : mul(wb, b.tw[4 * k * twstep]);
          wa = (r & 3) == 0  ? float2(1.0f, 0.0f)
              : (r & 3) == 1 ? w1
                             : mul(wa, w1);
          const float2 w = r < 4 ? wa : (r & 3) ? mul(wa, wb) : wb;
          const float2 x = b.buf[P2 ? bj + r * rs : pad(j + r * nR, sh) * es];
          v[m][r] = mul(x, INV ? conj(w) : w);
        }
      }
      bfly<R, 1>(v[m], INV);
    }
  }

  if (last && !UNPACK && (C || !p.via_smem)) {
    // Four-step twiddles W^(k i0) of the outputs k = j + r nR step by W^(nR i0)
    const float2 ws = four_step ? twid(b.tt, nR * b.i0) : 0.0f;
    for (uint m = 0; m < M; m++) {
      const uint j = lane + m * J;
      if (j < nRv) {
        float2 w = four_step ? twid(b.tt, j * b.i0) : 0.0f;
        for (uint r = 0; r < R; r++) {
          store_pt<I, O, F, C>(b, p, j + r * nR, v[m][opos<R>(r)], w);
          w = mul(w, ws);
        }
      }
    }
    return;
  }

  if (!first) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint m = 0; m < M; m++) {
    const uint j = lane + m * J;
    if (j < nR) {
      const uint k = mod_ns<P2>(j, Ns, inv_ns);
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

// Radices of the dynamic kernel families, one bit per radix. Bit 1 is length 1.
#define MASK_P2 ((1u << 1) | (1u << 2) | (1u << 4) | (1u << 8) | (1u << 16))
#define MASK_P235 (MASK_P2 | (1u << 3) | (1u << 5))
#define MASK_P2357 (MASK_P235 | (1u << 7))

template <class I, class O, uint MASK, uint F, uint N = 0, bool C = false>
kernel void fft_kernel(
    device const I* in,
    device O* out,
    device const float2* cl,
    device const float2* cs,
    device const float2* tt,
    constant float2* tw,
    constant FFTParams& p,
    threadgroup float2* buf [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr bool P2 = MASK && (MASK & ~MASK_P2) == 0;
  constexpr bool four_step = F & kFFTFourStepTwiddles;
  constexpr uint EPT = fft_points_per_thread(N, F);
  const uint n = N ? N : p.n, B = C ? 1 : p.tgb, J = N ? N / EPT : p.jt;
  const uint first_b = tg.x * B, nb = min(B, p.sizes[0] - first_b);

  const bool nfast = C || p.in_stride == 1;
  const uint b = nfast ? tid / J : tid % B, lane = nfast ? tid % J : tid / B;
  const int2 off = batch_off(p, tg.y, tg.z);
  const bool valid = b < nb;
  const uint r0 = first_b + (valid ? b : 0), i0 = p.i0_dim1 ? tg.y : r0;

  device const I* ib = in + off.x + int(r0) * p.in_strides[0];
  device O* ob = out + off.y + int(r0) * p.out_strides[0];
  const Bufs<I, O> bufs{ib, ob, cl, cs, tt, tw, buf + b * p.row, i0};

  if IF_CONSTEXPR (N) {
    constexpr uint r0 = first_radix(N), r1 = first_radix(N / r0),
                   r2 = first_radix(N / r0 / r1);
    stage<r0, P2, I, O, F, N, 1, C>(bufs, p, 0, lane, valid);
    stage<r1, P2, I, O, F, N, r0, C>(bufs, p, 1, lane, valid);
    if IF_CONSTEXPR (r2 > 1) {
      stage<r2, P2, I, O, F, N, r0 * r1, C>(bufs, p, 2, lane, valid);
    }
  } else
    for (uint s = 0; s < p.nstages; s++) {
#define STAGE(RR)                                               \
  case RR:                                                      \
    if IF_CONSTEXPR (MASK & (1u << RR)) {                       \
      stage<RR, P2, I, O, F, 0, 0, C>(bufs, p, s, lane, valid); \
    }                                                           \
    break;
      switch (p.radix[s]) { // Descending: ascending compiles 3% slower
        STAGE(1) STAGE(16) STAGE(8) STAGE(7) STAGE(5) STAGE(4) STAGE(3) STAGE(2)
      }
#undef STAGE
    }

  constexpr bool UNPACK = !is_complex_v<I> && (F & kFFTPackedLoads);
  if (!UNPACK && (C || !p.via_smem)) {
    return;
  }

  const bool ofast = C || p.out_stride == 1;
  const uint bs = ofast ? tid / J : tid % B, ls = ofast ? tid % J : tid / B;
  threadgroup const float2* rb = buf + bs * p.row;
  Bufs<I, O> bo = bufs;
  bo.out = out + off.y + int(first_b + bs) * p.out_strides[0],
  bo.i0 = p.i0_dim1 ? tg.y : first_b + bs;

  const uint es = C ? 1 : p.es, sh = C ? fft_contig_pad_shift(N) : p.padshift;
  float2 w = four_step ? twid(tt, ls * bo.i0) : 0.0f,
         ws = four_step ? twid(tt, J * bo.i0) : 0.0f;

  if (bs >= nb) {
    return;
  }

  for (uint k = ls; k < (UNPACK ? n / 2 + 1 : n); k += J) {
    const float2 a = rb[pad(k, sh) * es], c = rb[pad(k ? n - k : 0, sh) * es];
    if (UNPACK) {
      unpack<I, O, N, C>(bo, p, k, tw[k], float4(a, c));
    } else {
      store_pt<I, O, F, false>(bo, p, k, a, w);
      w = mul(w, ws);
    }
  }
}

#define FFT_ARGS(IN, OUT)                                                    \
  device const IN*, device OUT*, device const float2*, device const float2*, \
      device const float2*, constant float2*, constant FFTParams&

#define INST2(IN, OUT, F, MASK, NAME, N, C)                           \
  template [[host_name("fft_" #IN "_" #OUT "_" NAME #F)]] kernel void \
  fft_kernel<IN, OUT, MASK, F, N, C>(                                 \
      FFT_ARGS(IN, OUT), threadgroup float2*, uint, uint3);

#define INST1(...) INST2(__VA_ARGS__)
#define UNPAREN(...) __VA_ARGS__
#define INSTF(IN, OUT, F, X) INST1(IN, OUT, F, UNPAREN X)
#define INST(IK, OK, F, X) \
  INSTF(IK##_f, OK##_f, F, X) INSTF(IK##_h, OK##_h, F, X)
// Only single-pass kernels come in fp16: anything longer runs in fp32
#define INST_FP32(IK, OK, F, X) INSTF(IK##_f, OK##_f, F, X)

// Lists of (input kind, output kind, F) the planner requests, F as the literal
// sum of the kFFT* bits (FFT.h) since #F ends the kernel name: 1 inverse, 2
// packed loads, 4 four-step twiddles, 8 extended stores, 16 extended loads, 64
// packed real stores
#define C_f float2
#define C_h half2
#define R_f float
#define R_h half

#define BASIC(_, X) \
  _(C, C, 0, X)     \
  _(C, C, 1, X) _(R, C, 2, X) _(C, R, 67, X)
#define EXT1(_, X) _(C, C, 4, X) _(C, C, 5, X) _(C, C, 8, X) _(C, C, 9, X)
#define EXT2(_, X) \
  _(C, C, 20, X)   \
  _(C, C, 21, X) _(C, C, 24, X) _(R, C, 4, X)
#define EXT3(_, X) _(R, C, 20, X) _(R, C, 24, X) _(C, R, 9, X)
#define EXT(_, X) EXT1(_, X) EXT2(_, X) EXT3(_, X)
#define PLAIN(_, X) BASIC(_, X) _(R, C, 0, X) _(C, R, 1, X)

// The smallest family factoring n (radix branches cost registers); p16: 1, 16
#define FAMILY(MASK, NAME) \
  PLAIN(INST, (MASK, NAME, 0, false)) EXT(INST_FP32, (MASK, NAME, 0, false))
FAMILY(MASK_P2, "p2_")
FAMILY(MASK_P235, "p235_")
FAMILY(MASK_P2357, "p2357_")
PLAIN(INST, (((1u << 16) | (1u << 1)), "p16_", 0, false))

#define FIXED_X(N) ((N & (N - 1) ? MASK_P235 : MASK_P2), #N "_", N, false)
#define FIXED(N) BASIC(INST, FIXED_X(N)) EXT(INST_FP32, FIXED_X(N))
FFT_FIXED_SIZES(FIXED)

#define CONTIG(N) BASIC(INST, (MASK_P2, #N "c_", N, true))
FFT_CONTIG_SIZES(CONTIG)
