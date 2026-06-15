#include <ATen/native/mps/kernels/Distance.h>
#include <c10/metal/common.h>
#include <metal_stdlib>

using namespace metal;

// P_KIND: 0 = p==1, 1 = p==inf, 2 = generic. The TG precomputes
// `grad / cdist^(p-1)` per (b, i, j) into `s_reducer`, shared across c-threads.
template <typename T, int P_KIND, uint TG_C>
kernel void cdist_backward(
    constant T* x1 [[buffer(0)]],
    constant T* x2 [[buffer(1)]],
    constant T* grad [[buffer(2)]],
    constant float* cdist [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant CdistBwdParams& params [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 ltid3 [[thread_position_in_threadgroup]]) {
  const uint ltid = ltid3.x;

  const long D = params.D;
  const long P = params.P;
  const long R = params.R;
  const float pm1 = params.p_minus_1;

  const long c = static_cast<long>(tid.x);
  const long i = static_cast<long>(tid.y);
  const long b = static_cast<long>(tid.z);
  // Padding threads (c >= D) still participate in the cooperative load.
  const bool active = c < D;

  threadgroup float s_reducer[TG_C];
  // Only the p == inf exact-match test needs cdist in TG memory.
  threadgroup float s_cdist[(P_KIND == 1) ? TG_C : 1];

  const ulong x1_off =
      static_cast<ulong>(b) * static_cast<ulong>(P) * static_cast<ulong>(D) +
      static_cast<ulong>(i) * static_cast<ulong>(D) + static_cast<ulong>(c);
  const ulong x2_row =
      static_cast<ulong>(b) * static_cast<ulong>(R) * static_cast<ulong>(D);
  const ulong gr_row =
      static_cast<ulong>(b) * static_cast<ulong>(P) * static_cast<ulong>(R) +
      static_cast<ulong>(i) * static_cast<ulong>(R);

  const float x1ic = active ? static_cast<float>(x1[x1_off]) : 0.0f;
  float acc = 0.0f;

  for (long j_base = 0; j_base < R; j_base += TG_C) {
    const long tile_size = min(static_cast<long>(TG_C), R - j_base);

    // Tile size = TG_C: each thread loads at most one j per tile.
    const long t = static_cast<long>(ltid);
    if (t < tile_size) {
      const ulong g_idx = gr_row + static_cast<ulong>(j_base + t);
      const float g_val = static_cast<float>(grad[g_idx]);
      if IF_CONSTEXPR (P_KIND == 0) {
        s_reducer[t] = g_val;
      } else if IF_CONSTEXPR (P_KIND == 1) {
        s_reducer[t] = g_val;
        s_cdist[t] = cdist[g_idx];
      } else {
        const float c_val = cdist[g_idx];
        s_reducer[t] =
            (c_val == 0.0f) ? 0.0f : g_val / ::metal::precise::pow(c_val, pm1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      const ulong x2_off = x2_row +
          static_cast<ulong>(j_base) * static_cast<ulong>(D) +
          static_cast<ulong>(c);
      for (long t_off = 0; t_off < tile_size; ++t_off) {
        const float x2jc = static_cast<float>(
            x2[x2_off + static_cast<ulong>(t_off) * static_cast<ulong>(D)]);
        const float dij = x1ic - x2jc;
        // dij == 0 contributes 0 (sign(0) = 0); also dodges pow(0, neg) when
        // p<1.
        if (dij == 0.0f) {
          continue;
        }
        const float sij = (dij > 0.0f) ? 1.0f : -1.0f;
        const float r = s_reducer[t_off];
        if IF_CONSTEXPR (P_KIND == 0) {
          acc += r * sij;
        } else if IF_CONSTEXPR (P_KIND == 1) {
          if (::metal::precise::abs(dij) == s_cdist[t_off]) {
            acc += r * sij;
          }
        } else {
          acc +=
              r * sij * ::metal::precise::pow(::metal::precise::abs(dij), pm1);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (active) {
    out[x1_off] = static_cast<T>(acc);
  }
}

#define REGISTER_CDIST_BACKWARD(T, P_KIND, TG_C)                        \
  template [[host_name("cdist_backward_" #T "_p" #P_KIND "_tg" #TG_C)]] \
  kernel void cdist_backward<T, P_KIND, TG_C>(                          \
      constant T * x1 [[buffer(0)]],                                    \
      constant T * x2 [[buffer(1)]],                                    \
      constant T * grad [[buffer(2)]],                                  \
      constant float* cdist [[buffer(3)]],                              \
      device T* out [[buffer(4)]],                                      \
      constant CdistBwdParams& params [[buffer(5)]],                    \
      uint3 tid [[thread_position_in_grid]],                            \
      uint3 ltid3 [[thread_position_in_threadgroup]]);

#define REGISTER_CDIST_BACKWARD_FOR_TYPE(T) \
  REGISTER_CDIST_BACKWARD(T, 0, 32)         \
  REGISTER_CDIST_BACKWARD(T, 0, 128)        \
  REGISTER_CDIST_BACKWARD(T, 1, 32)         \
  REGISTER_CDIST_BACKWARD(T, 1, 128)        \
  REGISTER_CDIST_BACKWARD(T, 2, 32)         \
  REGISTER_CDIST_BACKWARD(T, 2, 128)

REGISTER_CDIST_BACKWARD_FOR_TYPE(float)
REGISTER_CDIST_BACKWARD_FOR_TYPE(half)
REGISTER_CDIST_BACKWARD_FOR_TYPE(bfloat)
