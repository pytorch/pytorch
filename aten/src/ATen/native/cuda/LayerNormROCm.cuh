// Cached-row LayerNorm / RMSNorm kernels for ROCm (wave64, CDNA).
//
// Forward: the row is loaded once into registers (NV 16-byte vectors per thread), reduced with DPP lane
// butterflies (plus one LDS slot and one barrier per reduction when a row spans several waves) and normalised
// from registers, so X is read exactly once. Statistics are a corrected two-pass (mean, then one reduction of
// the centred sum and sum of squares), which keeps the accuracy of the Welford kernel on large-mean inputs.
// Narrow rows are packed several to a wave (8 or 16 lanes per row) so no lane idles.
//
// Backward: one persistent kernel writes dX and accumulates per-block dgamma/dbeta partials in registers (gamma
// is loaded once per block); a small second kernel reduces the [blocks, N] fp32 workspace.
//
//   dx = rstd * (g - mean(g) - xhat * mean(g * xhat))        (LayerNorm, g = dY * gamma)
//   dx = rstd * (g - xhat * mean(g * xhat))                  (RMSNorm,   xhat = x * rstd)
//
// Every entry point returns false when it does not take the problem (odd N, unaligned pointers, no
// configuration that fits in registers, ...) and the caller keeps its existing kernels.
#pragma once

#ifdef USE_ROCM

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <ATen/native/cuda/LayerNormROCm.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace at::native::rocm_ln {

namespace detail {

// ---- DPP lane reductions ----------------------------------------------------------------------------------
template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}

// Sum over G lanes (G = 8, 16 or 64); all lanes of each group end with the same bits.
template <int G>
__device__ __forceinline__ float group_sum(float v) {
  static_assert(G == 8 || G == 16 || G == 64, "unsupported group size");
  v += dpp<0xB1>(v);                         // quad_perm [1,0,3,2]
  v += dpp<0x4E>(v);                         // quad_perm [2,3,0,1]
  if constexpr (G >= 8) v += dpp<0x141>(v);  // row_half_mirror
  if constexpr (G >= 16) v += dpp<0x140>(v); // row_mirror
  if constexpr (G == 64) {
    // the four 16-lane rows -> wave-uniform scalar
    const float a = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0));
    const float b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
    const float c = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32));
    const float d = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
    v = (a + b) + (c + d);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ float up(T v) {
  return static_cast<float>(v);
}
template <typename T>
__device__ __forceinline__ T down(float v) {
  return static_cast<T>(v);
}

template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Vec {
  T e[VEC];
};

// Threads per row group layout shared by both kernels.
//   G    lanes per row group (8, 16 or 64)
//   WPR  waves per row (only with G == 64)
//   rows per block = BLOCK / (G * WPR)
template <int G, int WPR>
constexpr int block_size() {
  return G == 64 ? (WPR == 1 ? 256 : 64 * WPR) : 256;
}

// ---- forward --------------------------------------------------------------------------------------------
// AFF: 0 no affine, 1 gamma only, 2 gamma and beta. Compile-time so every load stays unconditional and the
// loads are batched. FULL: N == NV * G * WPR * VEC, i.e. no tail. Otherwise out-of-range vectors are clamped
// onto the last vector of the row (all loads are unconditional) and masked out of the sums with a 0/1 factor;
// only the stores are predicated.
template <typename T, bool RMS, int VEC, int NV, int G, int WPR, bool FULL, int AFF>
__global__ __launch_bounds__((block_size<G, WPR>())) void ln_fwd_kernel(
    const T* __restrict__ X, const T* __restrict__ W, const T* __restrict__ Bv, T* __restrict__ Y,
    float* __restrict__ mean_out, float* __restrict__ rstd_out, int64_t M, int N, float eps, float inv_n) {
  constexpr int BLOCK = block_size<G, WPR>();
  constexpr int TPR = G * WPR;  // threads per row
  constexpr int ROWS = BLOCK / TPR;
  static_assert(BLOCK % TPR == 0, "block must hold whole rows");
  using V = Vec<T, VEC>;

  __shared__ float red[WPR > 1 ? 2 * 2 * WPR : 1];

  const int tid = threadIdx.x;
  const int c = tid % TPR;
  int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS + tid / TPR;
  const bool row_ok = row < M;
  if (!row_ok) row = M - 1;  // keep every lane of the wave active for the DPP reductions

  // element index of vector v for this thread, clamped into the row, and its 0/1 validity
  auto vidx = [&](int v) {
    const int idx = (v * TPR + c) * VEC;
    return FULL ? idx : min(idx, N - VEC);
  };
  auto vmask = [&](int v) -> float {
    if constexpr (FULL) {
      return 1.f;
    } else {
      return (v * TPR + c) * VEC < N ? 1.f : 0.f;
    }
  };

  const T* xr = X + row * static_cast<int64_t>(N);
  V xv[NV];
  V wv[NV];
  V bv[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) xv[v] = *reinterpret_cast<const V*>(xr + vidx(v));
  constexpr bool has_w = AFF >= 1, has_b = !RMS && AFF == 2;
  auto load_wb = [&]() {
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      if constexpr (has_w) wv[v] = *reinterpret_cast<const V*>(W + vidx(v));
      if constexpr (has_b) bv[v] = *reinterpret_cast<const V*>(Bv + vidx(v));
    }
  };
  // Short rows: gamma/beta in flight during the reductions. Wide rows: load them after the reductions so the
  // 2*NV extra vector registers do not cost occupancy.
  constexpr bool PRELOAD = NV * VEC * sizeof(T) <= 32;
  if constexpr (PRELOAD) load_wb();

  auto reduce1 = [&](float s) -> float {
    s = group_sum<G>(s);
    if constexpr (WPR > 1) {
      const int wave = tid >> 6;
      if ((tid & 63) == 0) red[wave] = s;
      __syncthreads();
      float t = red[0];
#pragma unroll
      for (int w = 1; w < WPR; ++w) t += red[w];
      return t;
    } else {
      return s;
    }
  };

  // one barrier for two values (the second LDS slot pair keeps the two reductions from racing)
  auto reduce2 = [&](float a, float b, float& ra, float& rb) {
    const float g1 = group_sum<G>(a), g2 = group_sum<G>(b);
    if constexpr (WPR > 1) {
      const int wave = tid >> 6;
      if ((tid & 63) == 0) {
        red[2 * WPR + wave] = g1;
        red[3 * WPR + wave] = g2;
      }
      __syncthreads();
      ra = red[2 * WPR];
      rb = red[3 * WPR];
#pragma unroll
      for (int w = 1; w < WPR; ++w) {
        ra += red[2 * WPR + w];
        rb += red[3 * WPR + w];
      }
    } else {
      ra = g1;
      rb = g2;
    }
  };

  float mean = 0.f, var = 1.f;
  if constexpr (RMS) {
    float s = 0.f;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float m = vmask(v);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float x = up(xv[v].e[e]);
        s += FULL ? x * x : x * x * m;
      }
    }
    var = reduce1(s) * inv_n;
  } else {
    // corrected two-pass: mean0 = sum(x)/N, then ONE reduction of (sum d, sum d^2) with d = x - mean0.
    // mean = mean0 + sum(d)/N repairs the rounding of mean0 (constant rows come out exactly), and
    // var = E[d^2] - E[d]^2 has no large-mean cancellation because d is already centred.
    float s = 0.f;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float m = vmask(v);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float x = up(xv[v].e[e]);
        s += FULL ? x : x * m;
      }
    }
    const float mean0 = reduce1(s) * inv_n;
    float c1 = 0.f, c2 = 0.f;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float m = vmask(v);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float d = up(xv[v].e[e]) - mean0;
        c1 += FULL ? d : d * m;
        c2 += FULL ? d * d : d * d * m;
      }
    }
    float t1, t2;
    reduce2(c1, c2, t1, t2);
    const float corr = t1 * inv_n;
    mean = mean0 + corr;
    var = fmaxf(t2 * inv_n - corr * corr, 0.f);
  }
  const float rstd = __builtin_amdgcn_rsqf(var + eps);

  if (row_ok && c == 0) {
    if constexpr (!RMS) mean_out[row] = mean;
    rstd_out[row] = rstd;
  }
  if (!row_ok) return;
  if constexpr (!PRELOAD) load_wb();

  T* yr = Y + row * static_cast<int64_t>(N);
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int idx = (v * TPR + c) * VEC;
    V o;
#pragma unroll
    for (int e = 0; e < VEC; ++e) {
      const float x = up(xv[v].e[e]);
      float t = RMS ? x * rstd : (x - mean) * rstd;
      if constexpr (has_w) {
        const float w = up(wv[v].e[e]);
        if constexpr (has_b) {
          t = fmaf(t, w, up(bv[v].e[e]));
        } else {
          t = t * w;
        }
      }
      o.e[e] = down<T>(t);
    }
    if (FULL || idx < N) *reinterpret_cast<V*>(yr + idx) = o;
  }
}

// ---- backward --------------------------------------------------------------------------------------------
constexpr int kBwdMaxCols = 64;  // columns per thread (NV * VEC): register-resident dgamma/dbeta accumulators
constexpr int kBwdPrefetchMaxCols = 16;
constexpr int kBwdMaxLds = 48 * 1024;  // bytes of LDS used to merge the row groups of a block

template <int NV, int VEC, int TPR, int ROWS>
constexpr int bwd_lds_bytes() {
  return ROWS > 1 ? ROWS * TPR * NV * VEC * static_cast<int>(sizeof(float)) : 0;
}

// Always accumulates dgamma / dbeta partials and expects gamma (the dX-only cases stay on the existing kernels).
template <typename T, bool RMS, int VEC, int NV, int G, int WPR, bool FULL>
__global__ __launch_bounds__((block_size<G, WPR>())) void ln_bwd_kernel(
    const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ mean,
    const float* __restrict__ rstd, const T* __restrict__ W, T* __restrict__ dX, float* __restrict__ part_g,
    float* __restrict__ part_b, int64_t M, int N, float inv_n) {
  constexpr int BLOCK = block_size<G, WPR>();
  constexpr int TPR = G * WPR;
  constexpr int ROWS = BLOCK / TPR;
  constexpr int CPT = NV * VEC;  // columns per thread
  static_assert(BLOCK % TPR == 0, "bad layout");
  using V = Vec<T, VEC>;

  __shared__ float red[WPR > 1 ? 2 * 2 * WPR : 1];  // [parity][slot][wave]
  __shared__ float lds[ROWS > 1 ? ROWS * TPR * CPT : 1];

  const int tid = threadIdx.x;
  const int c = tid % TPR;
  const int grp = tid / TPR;

  auto vidx = [&](int v) {
    const int idx = (v * TPR + c) * VEC;
    return FULL ? idx : min(idx, N - VEC);
  };
  auto vmask = [&](int v) -> float {
    if constexpr (FULL) {
      return 1.f;
    } else {
      return (v * TPR + c) * VEC < N ? 1.f : 0.f;
    }
  };

  V wv[NV];
#pragma unroll
  for (int v = 0; v < NV; ++v) wv[v] = *reinterpret_cast<const V*>(W + vidx(v));

  float dg[CPT];
  float db[RMS ? 1 : CPT];
#pragma unroll
  for (int k = 0; k < CPT; ++k) {
    dg[k] = 0.f;
    if constexpr (!RMS) db[k] = 0.f;
  }

  const int64_t gstride = static_cast<int64_t>(gridDim.x) * ROWS;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * ROWS;
  const int64_t row0 = first + grp;
  const int64_t iters = first < M ? (M - first + gstride - 1) / gstride : 0;

  V xv[NV], yv[NV], xn[NV], yn[NV];
  float mu = 0.f, rs = 0.f, mun = 0.f, rsn = 0.f;
  auto load_row = [&](int64_t row, V* xo, V* yo, float& m_o, float& r_o) {
    const int64_t rl = row < M ? row : M - 1;
    const T* xr = X + rl * static_cast<int64_t>(N);
    const T* yr = dY + rl * static_cast<int64_t>(N);
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const int i = vidx(v);
      xo[v] = *reinterpret_cast<const V*>(xr + i);
      yo[v] = *reinterpret_cast<const V*>(yr + i);
    }
    r_o = rstd[rl];
    if constexpr (!RMS) m_o = mean[rl];
  };
  // Prefetching the next row doubles the x / dY registers; skip it when the block is register-starved.
  constexpr bool PREFETCH = CPT <= kBwdPrefetchMaxCols && BLOCK < 1024;
  if constexpr (PREFETCH) {
    if (iters > 0) load_row(row0, xv, yv, mu, rs);
  }

  for (int64_t it = 0; it < iters; ++it) {
    const int64_t row = row0 + it * gstride;
    const bool rv = row < M;
    if constexpr (PREFETCH) {
      if (it + 1 < iters) load_row(row + gstride, xn, yn, mun, rsn);
    } else {
      load_row(row, xv, yv, mu, rs);
    }

    // pass 1: sums (and the dgamma / dbeta partials)
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      const float m = vmask(v);
#pragma unroll
      for (int e = 0; e < VEC; ++e) {
        const float x = up(xv[v].e[e]);
        float dy = up(yv[v].e[e]);
        if constexpr (ROWS > 1) dy = rv ? dy : 0.f;  // rows past M contribute nothing to dgamma/dbeta
        const float xh = RMS ? x * rs : (x - mu) * rs;
        float g = dy * up(wv[v].e[e]);
        if constexpr (!FULL) g *= m;
        s1 += g;
        s2 = fmaf(g, xh, s2);
        dg[v * VEC + e] = fmaf(dy, xh, dg[v * VEC + e]);
        if constexpr (!RMS) db[v * VEC + e] += dy;
      }
    }
    // row reduction (parity-double-buffered LDS so no second barrier is needed)
    float t1 = 0.f, t2 = 0.f;
    {
      const float g1 = RMS ? 0.f : group_sum<G>(s1);
      const float g2 = group_sum<G>(s2);
      if constexpr (WPR > 1) {
        const int wave = tid >> 6;
        float* r = red + (it & 1) * 2 * WPR;
        if ((tid & 63) == 0) {
          r[wave] = g1;
          r[WPR + wave] = g2;
        }
        __syncthreads();
        t1 = r[0];
        t2 = r[WPR];
#pragma unroll
        for (int w = 1; w < WPR; ++w) {
          t1 += r[w];
          t2 += r[WPR + w];
        }
      } else {
        t1 = g1;
        t2 = g2;
      }
    }
    const float c1 = t1 * inv_n, c2 = t2 * inv_n;

    // pass 2: dx
    if (rv) {
      T* dxr = dX + row * static_cast<int64_t>(N);
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        V o;
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          const float x = up(xv[v].e[e]);
          const float dy = up(yv[v].e[e]);
          const float xh = RMS ? x * rs : (x - mu) * rs;
          const float g = dy * up(wv[v].e[e]);
          const float t = RMS ? fmaf(-xh, c2, g) : fmaf(-xh, c2, g - c1);
          o.e[e] = down<T>(t * rs);
        }
        const int idx = (v * TPR + c) * VEC;
        if (FULL || idx < N) *reinterpret_cast<V*>(dxr + idx) = o;
      }
    }
    if constexpr (PREFETCH) {
      if (it + 1 < iters) {
#pragma unroll
        for (int v = 0; v < NV; ++v) {
          xv[v] = xn[v];
          yv[v] = yn[v];
        }
        mu = mun;
        rs = rsn;
      }
    }
  }

  // merge the row groups of this block and write one workspace row
  auto write_out = [&](float* acc, float* dst) {
    if constexpr (ROWS > 1) {
      __syncthreads();
#pragma unroll
      for (int k = 0; k < CPT; ++k) lds[(grp * CPT + k) * TPR + c] = acc[k];
      __syncthreads();
      if (grp == 0) {
#pragma unroll
        for (int k = 0; k < CPT; ++k) {
          float s = lds[k * TPR + c];
#pragma unroll
          for (int g = 1; g < ROWS; ++g) s += lds[(g * CPT + k) * TPR + c];
          acc[k] = s;
        }
      }
    }
    if (grp == 0) {
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int base = (v * TPR + c) * VEC;
#pragma unroll
        for (int e = 0; e < VEC; ++e)
          if (FULL || base + e < N) dst[base + e] = acc[v * VEC + e];
      }
    }
  };
  write_out(dg, part_g + static_cast<int64_t>(blockIdx.x) * N);
  if constexpr (!RMS) write_out(db, part_b + static_cast<int64_t>(blockIdx.x) * N);
}

// second stage: out[j] = sum_p part[p][j]
// block = 8 column groups (4 columns each, float4 loads) x 32 row slices; every thread issues its row loads
// back to back (unrolled) so the loads of the whole workspace column are in flight together. N % 4 == 0.
template <typename T, bool HAS_B>
__global__ __launch_bounds__(256) void ln_bwd_reduce_kernel(
    const float* __restrict__ part_g, const float* __restrict__ part_b, int P, int N, T* __restrict__ dW,
    T* __restrict__ dB) {
  __shared__ float4 buf[2][32][8];
  const int cx = threadIdx.x & 7, sy = threadIdx.x >> 3;
  const int col = (blockIdx.x * 8 + cx) * 4;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
  if (col < N) {
#pragma unroll 8
    for (int p = sy; p < P; p += 32) {
      const float4 v = *reinterpret_cast<const float4*>(part_g + static_cast<int64_t>(p) * N + col);
      a.x += v.x;
      a.y += v.y;
      a.z += v.z;
      a.w += v.w;
      if constexpr (HAS_B) {
        const float4 u = *reinterpret_cast<const float4*>(part_b + static_cast<int64_t>(p) * N + col);
        b.x += u.x;
        b.y += u.y;
        b.z += u.z;
        b.w += u.w;
      }
    }
  }
  buf[0][sy][cx] = a;
  buf[1][sy][cx] = b;
  __syncthreads();
  if (sy == 0 && col < N) {
#pragma unroll
    for (int s = 1; s < 32; ++s) {
      const float4 va = buf[0][s][cx], vb = buf[1][s][cx];
      a.x += va.x;
      a.y += va.y;
      a.z += va.z;
      a.w += va.w;
      b.x += vb.x;
      b.y += vb.y;
      b.z += vb.z;
      b.w += vb.w;
    }
    if (dW) {
      dW[col] = down<T>(a.x);
      dW[col + 1] = down<T>(a.y);
      dW[col + 2] = down<T>(a.z);
      dW[col + 3] = down<T>(a.w);
    }
    if constexpr (HAS_B) {
      if (dB) {
        dB[col] = down<T>(b.x);
        dB[col + 1] = down<T>(b.y);
        dB[col + 2] = down<T>(b.z);
        dB[col + 3] = down<T>(b.w);
      }
    }
  }
}

// ---- host side -----------------------------------------------------------------------------------------
// A launch configuration: G lanes per row group, WPR waves per row, NV vectors per thread.
struct Config {
  int G, WPR, NV;
};

inline bool aligned16(const void* p) {
  return p == nullptr || (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

// Instantiations that spill registers to scratch are rejected (checked once per instantiation): a spilling
// kernel is several times slower than the existing ones. The limit is 128 VGPRs for a 1024-thread block and 256
// for 512 or fewer, so the right configuration depends on the dtype and the number of accumulators.
template <typename Kernel>
inline bool no_spill(Kernel* fn) {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(fn)) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attr.localSizeBytes == 0;
}

// Number of vector columns for G == 64: NV values the table below instantiates.
constexpr bool nv_ok(int nv, int vec) {
  switch (nv) {
    case 1: case 2: case 3: case 4: case 6: case 8:
      return true;
    case 12: case 16:
      return vec <= 4;  // 32-bit element types only (CPT would exceed kBwdMaxCols otherwise)
    default:
      return false;
  }
}

inline int round_up_nv(int nv, int vec) {
  for (int c : {1, 2, 3, 4, 6, 8, 12, 16})
    if (nv <= c && nv_ok(c, vec)) return c;
  return -1;
}

// Largest accepted ratio of launched vector slots to real vectors per row (tail padding). With few rows the
// padded lanes cannot be hidden behind other blocks, so the backward is stricter for M < 1024.
constexpr double kMaxPadFwd = 1.25, kMaxPadBwd = 1.25, kMaxPadBwdSmallM = 1.1;
constexpr int64_t kPadGuardMinVec = 256;  // narrower rows fit in a wave or two; padding there is cheap

// ---- forward dispatch
template <typename T, bool RMS, int VEC, int NV, int G, int WPR, bool FULL, int AFF>
inline bool fwd_launch_one(const FwdArgs& k) {
  constexpr int BLOCK = block_size<G, WPR>();
  constexpr int ROWS = BLOCK / (G * WPR);
  static const bool fits = no_spill(&ln_fwd_kernel<T, RMS, VEC, NV, G, WPR, FULL, AFF>);
  if (!fits) return false;
  const int64_t blocks = (k.M + ROWS - 1) / ROWS;
  // ROCm rejects launches where gridDim.x * blockDim.x exceeds uint32_t max
  if (static_cast<uint64_t>(blocks) * BLOCK > std::numeric_limits<uint32_t>::max()) return false;
  ln_fwd_kernel<T, RMS, VEC, NV, G, WPR, FULL, AFF>
      <<<dim3(static_cast<unsigned>(blocks)), dim3(BLOCK), 0, k.stream>>>(
          static_cast<const T*>(k.X), static_cast<const T*>(k.gamma), static_cast<const T*>(k.beta),
          static_cast<T*>(k.Y), k.mean, k.rstd, k.M, static_cast<int>(k.N), k.eps, 1.0f / static_cast<float>(k.N));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

template <typename T, bool RMS, int VEC, int NV, int G, int WPR>
inline bool fwd_launch_full(const FwdArgs& k, bool full) {
  if (k.gamma == nullptr) {
    if (k.beta != nullptr) return false;
    return full ? fwd_launch_one<T, RMS, VEC, NV, G, WPR, true, 0>(k)
                : fwd_launch_one<T, RMS, VEC, NV, G, WPR, false, 0>(k);
  }
  if (RMS || k.beta == nullptr) {
    return full ? fwd_launch_one<T, RMS, VEC, NV, G, WPR, true, 1>(k)
                : fwd_launch_one<T, RMS, VEC, NV, G, WPR, false, 1>(k);
  }
  if constexpr (!RMS) {
    return full ? fwd_launch_one<T, RMS, VEC, NV, G, WPR, true, 2>(k)
                : fwd_launch_one<T, RMS, VEC, NV, G, WPR, false, 2>(k);
  }
  return false;
}

template <typename T, bool RMS, int VEC, int G, int WPR>
inline bool fwd_launch_nv(const FwdArgs& k, int nv, bool full) {
#define LN_ROCM_CASE(n)                                                         \
  case n:                                                                       \
    if constexpr (nv_ok(n, VEC)) return fwd_launch_full<T, RMS, VEC, n, G, WPR>(k, full); \
    else return false;
  switch (nv) {
    LN_ROCM_CASE(1) LN_ROCM_CASE(2) LN_ROCM_CASE(3) LN_ROCM_CASE(4) LN_ROCM_CASE(6) LN_ROCM_CASE(8)
    LN_ROCM_CASE(12) LN_ROCM_CASE(16)
  }
#undef LN_ROCM_CASE
  return false;
}

template <typename T, bool RMS, int VEC>
inline bool fwd_launch_config(const FwdArgs& k, const Config& cfg) {
  const int nvec = static_cast<int>(k.N / VEC);
  const int tpr = cfg.G == 64 ? 64 * cfg.WPR : cfg.G;
  if (cfg.NV * tpr < nvec) return false;
  const bool full = nvec == cfg.NV * tpr;
  if (cfg.G == 8) return cfg.NV == 1 && fwd_launch_full<T, RMS, VEC, 1, 8, 1>(k, full);
  if (cfg.G == 16) {
    if (cfg.NV == 1) return fwd_launch_full<T, RMS, VEC, 1, 16, 1>(k, full);
    if (cfg.NV == 2) return fwd_launch_full<T, RMS, VEC, 2, 16, 1>(k, full);
    return false;
  }
  switch (cfg.WPR) {
    case 1: return fwd_launch_nv<T, RMS, VEC, 64, 1>(k, cfg.NV, full);
    case 2: return fwd_launch_nv<T, RMS, VEC, 64, 2>(k, cfg.NV, full);
    case 4: return fwd_launch_nv<T, RMS, VEC, 64, 4>(k, cfg.NV, full);
    case 8: return fwd_launch_nv<T, RMS, VEC, 64, 8>(k, cfg.NV, full);
    case 16: return fwd_launch_nv<T, RMS, VEC, 64, 16>(k, cfg.NV, full);
  }
  return false;
}

// Candidate configurations in preference order (fitted on MI355X). nvec is the number of 16-byte vectors per row.
inline std::vector<Config> fwd_candidates(int64_t M, int64_t nvec, int vec) {
  std::vector<Config> out;
  if (nvec <= 8) out.push_back({8, 1, 1});
  if (nvec <= 16) out.push_back({16, 1, 1});
  if (nvec <= 32 && M >= 2048) out.push_back({16, 1, 2});
  // Wide rows: several waves cooperate; fewer rows -> fewer vectors per thread (more threads per row).
  const double target = M <= 256 ? 1.0 : (M <= 1024 ? 2.0 : 4.0);
  struct Scored {
    double score;
    Config cfg;
  };
  std::vector<Scored> sc;
  for (int w : {1, 2, 4, 8, 16}) {
    const int64_t tpr = 64 * w;
    const int nv = round_up_nv(static_cast<int>((nvec + tpr - 1) / tpr), vec);
    if (nv < 0 || (nvec > kPadGuardMinVec && nv * tpr > kMaxPadFwd * nvec)) continue;  // idle tail lanes
    sc.push_back({std::fabs(std::log(static_cast<double>(nv) / target)) + ((nvec % tpr == 0) ? 0.0 : 0.4), {64, w, nv}});
  }
  std::stable_sort(sc.begin(), sc.end(), [](const Scored& a, const Scored& b) { return a.score < b.score; });
  for (const auto& e : sc) out.push_back(e.cfg);
  return out;
}

template <typename T>
bool forward(const FwdArgs& k) {
  constexpr int VEC = 16 / sizeof(T);
  if (k.N % VEC != 0 || k.N > (1 << 24) || !aligned16(k.X) || !aligned16(k.gamma) || !aligned16(k.beta) ||
      !aligned16(k.Y) || k.M <= 0 || k.N <= 0)
    return false;
  for (const Config& cfg : fwd_candidates(k.M, k.N / VEC, VEC)) {
    const bool ok = k.rms ? fwd_launch_config<T, true, VEC>(k, cfg) : fwd_launch_config<T, false, VEC>(k, cfg);
    if (ok) return true;
  }
  return false;
}

// ---- backward dispatch
// `alloc(n)` returns a float buffer of n elements that stays valid until the launched kernels have run.
template <typename T, bool RMS, int VEC, int NV, int G, int WPR, bool FULL>
inline bool bwd_launch_one(const BwdArgs& k, int bpc, const WsAlloc& alloc) {
  constexpr int BLOCK = block_size<G, WPR>();
  constexpr int ROWS = BLOCK / (G * WPR);
  if constexpr (bwd_lds_bytes<NV, VEC, G * WPR, ROWS>() > kBwdMaxLds || NV * VEC > kBwdMaxCols) {
    return false;
  } else {
    static const bool fits = no_spill(&ln_bwd_kernel<T, RMS, VEC, NV, G, WPR, FULL>);
    if (!fits) return false;
    const int64_t need = (k.M + ROWS - 1) / ROWS;
    const int64_t grid = std::max<int64_t>(1, std::min<int64_t>(need, static_cast<int64_t>(k.sm_count) * bpc));
    float* ws = alloc(grid * k.N * (RMS ? 1 : 2));
    float* pg = ws;
    float* pb = RMS ? nullptr : ws + grid * k.N;
    ln_bwd_kernel<T, RMS, VEC, NV, G, WPR, FULL><<<dim3(static_cast<unsigned>(grid)), dim3(BLOCK), 0, k.stream>>>(
        static_cast<const T*>(k.dY), static_cast<const T*>(k.X), k.mean, k.rstd, static_cast<const T*>(k.gamma),
        static_cast<T*>(k.dX), pg, pb, k.M, static_cast<int>(k.N), 1.0f / static_cast<float>(k.N));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ln_bwd_reduce_kernel<T, !RMS><<<dim3(static_cast<unsigned>((k.N + 31) / 32)), dim3(256), 0, k.stream>>>(
        pg, pb, static_cast<int>(grid), static_cast<int>(k.N), static_cast<T*>(k.dgamma), static_cast<T*>(k.dbeta));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }
}

template <typename T, bool RMS, int VEC, int NV, int G, int WPR>
inline bool bwd_launch_full(const BwdArgs& k, int bpc, bool full, const WsAlloc& alloc) {
  return full ? bwd_launch_one<T, RMS, VEC, NV, G, WPR, true>(k, bpc, alloc)
              : bwd_launch_one<T, RMS, VEC, NV, G, WPR, false>(k, bpc, alloc);
}

template <typename T, bool RMS, int VEC, int G, int WPR>
inline bool bwd_launch_nv(const BwdArgs& k, int bpc, int nv, bool full, const WsAlloc& alloc) {
#define LN_ROCM_CASE(n)                                                                       \
  case n:                                                                                     \
    if constexpr (nv_ok(n, VEC)) return bwd_launch_full<T, RMS, VEC, n, G, WPR>(k, bpc, full, alloc); \
    else return false;
  switch (nv) {
    LN_ROCM_CASE(1) LN_ROCM_CASE(2) LN_ROCM_CASE(3) LN_ROCM_CASE(4) LN_ROCM_CASE(6) LN_ROCM_CASE(8)
    LN_ROCM_CASE(12) LN_ROCM_CASE(16)
  }
#undef LN_ROCM_CASE
  return false;
}

struct BwdConfig {
  Config cfg;
  int bpc;  // persistent blocks per CU
};

template <typename T, bool RMS, int VEC>
inline bool bwd_launch_config(const BwdArgs& k, const BwdConfig& bc, const WsAlloc& alloc) {
  const Config& cfg = bc.cfg;
  const int nvec = static_cast<int>(k.N / VEC);
  const int tpr = cfg.G == 64 ? 64 * cfg.WPR : cfg.G;
  if (cfg.NV * tpr < nvec) return false;
  const bool full = nvec == cfg.NV * tpr;
  if (cfg.G == 8) return cfg.NV == 1 && bwd_launch_full<T, RMS, VEC, 1, 8, 1>(k, bc.bpc, full, alloc);
  if (cfg.G == 16) {
    if (cfg.NV == 1) return bwd_launch_full<T, RMS, VEC, 1, 16, 1>(k, bc.bpc, full, alloc);
    if (cfg.NV == 2) return bwd_launch_full<T, RMS, VEC, 2, 16, 1>(k, bc.bpc, full, alloc);
    return false;
  }
  switch (cfg.WPR) {
    case 1: return bwd_launch_nv<T, RMS, VEC, 64, 1>(k, bc.bpc, cfg.NV, full, alloc);
    case 2: return bwd_launch_nv<T, RMS, VEC, 64, 2>(k, bc.bpc, cfg.NV, full, alloc);
    case 4: return bwd_launch_nv<T, RMS, VEC, 64, 4>(k, bc.bpc, cfg.NV, full, alloc);
    case 8: return bwd_launch_nv<T, RMS, VEC, 64, 8>(k, bc.bpc, cfg.NV, full, alloc);
    case 16: return bwd_launch_nv<T, RMS, VEC, 64, 16>(k, bc.bpc, cfg.NV, full, alloc);
  }
  return false;
}

// Candidate configurations in preference order. Fitted against a grid over N = 64..16384, M = 256..16384,
// bf16/f16/f32 on MI355X: geometric mean 1.03x of the best configuration per shape, worst 1.17x.
inline std::vector<BwdConfig> bwd_candidates(int64_t M, int64_t nvec, int vec) {
  std::vector<BwdConfig> out;
  const int bpc = M >= 8192 ? 2 : 1;
  if (nvec <= 8) out.push_back({{8, 1, 1}, 4});
  if (nvec <= 16) out.push_back({{16, 1, 1}, M >= 8192 ? 1 : 2});
  if (nvec <= 32) out.push_back({{16, 1, 2}, M >= 8192 ? 1 : 2});
  const double target = nvec <= 512 ? 6.0 : 3.0;
  struct Scored {
    double score;
    Config cfg;
  };
  std::vector<Scored> sc;
  for (int w : {1, 2, 4, 8, 16}) {
    const int64_t tpr = 64 * w;
    const int nv = round_up_nv(static_cast<int>((nvec + tpr - 1) / tpr), vec);
    if (nv < 0 || (nvec > kPadGuardMinVec && nv * tpr > (M < 1024 ? kMaxPadBwdSmallM : kMaxPadBwd) * nvec)) continue;
    sc.push_back({std::fabs(std::log(static_cast<double>(nv) / target)) + ((nvec % tpr == 0) ? 0.0 : 0.8), {64, w, nv}});
  }
  std::stable_sort(sc.begin(), sc.end(), [](const Scored& a, const Scored& b) { return a.score < b.score; });
  for (const auto& e : sc) out.push_back({e.cfg, bpc});
  return out;
}

template <typename T>
bool backward(const BwdArgs& k, const WsAlloc& alloc) {
  constexpr int VEC = 16 / sizeof(T);
  // dX is required and gamma must be present for the dgamma / dbeta partials; the reduce kernel wants N % 4 == 0
  if (k.dX == nullptr || k.gamma == nullptr || (k.dgamma == nullptr && k.dbeta == nullptr)) return false;
  if (k.N % VEC != 0 || k.N % 4 != 0 || k.M <= 0 || k.N <= 0) return false;
  if (!aligned16(k.dY) || !aligned16(k.X) || !aligned16(k.dX) || !aligned16(k.gamma)) return false;
  for (const BwdConfig& bc : bwd_candidates(k.M, k.N / VEC, VEC)) {
    const bool ok = k.rms ? bwd_launch_config<T, true, VEC>(k, bc, alloc) : bwd_launch_config<T, false, VEC>(k, bc, alloc);
    if (ok) return true;
  }
  return false;
}

}  // namespace detail

template <typename T>
bool forward(const FwdArgs& k) {
  return detail::forward<T>(k);
}
template <typename T>
bool backward(const BwdArgs& k, const WsAlloc& alloc) {
  return detail::backward<T>(k, alloc);
}

}  // namespace at::native::rocm_ln

#endif  // USE_ROCM
