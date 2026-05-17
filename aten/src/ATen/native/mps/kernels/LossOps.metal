// LossOps.metal
// Metal compute kernels replacing MPSGraph ops in LossOps.mm.
// Eliminates shape-dependent graph cache growth for all loss operations.
//
// Reduction codes match ATen: 0=None, 1=Mean, 2=Sum
// Kernel naming convention:
//   <op>_fwd_none_<T>  : None reduction, writes typed T* output
//   <op>_fwd_reduce_<T>: Mean/Sum, writes float* partial sums (one per TG)
//   <op>_bwd_<T>       : backward, grad_out scalar (reduce) or per-elem (none)

#include <metal_stdlib>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
using namespace metal;

// ──────────────────────────────────────────────────────────────────────────
// Param structs  (binary layout must match C++ structs in LossOps.mm)
// ──────────────────────────────────────────────────────────────────────────

struct PointwiseLossParams {
  uint32_t N;
  float scale; // 1/N (Mean) or 1.0 (Sum/None)
  uint32_t reduction;
};

struct SmoothHuberParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  float beta;
  uint32_t is_huber; // 0=SmoothL1, 1=HuberLoss
};

struct BCEParams {
  uint32_t N;
  float scale;
  uint32_t reduction;
  uint32_t has_weight;
};

struct NLLParams {
  uint32_t N;
  uint32_t C;
  int32_t ignore_index;
  uint32_t reduction;
  uint32_t has_weight;
};

// ──────────────────────────────────────────────────────────────────────────
// Threadgroup tree-reduce helper
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
inline T tg_sum(T val, threadgroup T* smem, uint lid, uint tgsz) {
  // SIMD-group reduce: hardware, zero barriers within warp
  float f = simd_sum(float(val));
  if ((lid & 31u) == 0u)
    smem[lid >> 5u] = T(f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    float acc = 0.f;
    for (uint g = 0u; g < (tgsz >> 5u); ++g)
      acc += float(smem[g]);
    smem[0] = T(acc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return smem[0];
}

// ──────────────────────────────────────────────────────────────────────────
// Phase-2: merge per-threadgroup float partials into loss[0]
// Always dispatched with a single 256-thread threadgroup.
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
kernel void loss_reduce_partials_typed(
    device const float* partial [[buffer(0)]],
    device T* loss [[buffer(1)]],
    constant uint32_t& nparts [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  for (uint i = lid; i < nparts; i += tgsz)
    acc += partial[i];
  acc = tg_sum(acc, smem, lid, tgsz);
  if (lid == 0)
    loss[0] = T(acc);
}

#define INST_REDUCE_PARTIALS(T)                      \
  template [[host_name("loss_reduce_partials_" #T)]] \
  kernel void loss_reduce_partials_typed<T>(         \
      device const float*, device T*, constant uint32_t&, uint, uint)
INST_REDUCE_PARTIALS(float);
INST_REDUCE_PARTIALS(half);
INST_REDUCE_PARTIALS(bfloat);

// ============================================================================
// MSE Loss
// ============================================================================

template <typename T>
kernel void mse_loss_fwd_none(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint32_t& N [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint base = gid * 4u;
  if (base + 4u <= N) {
    using T4 = vec<T, 4>;
    T4 i4 = reinterpret_cast<device const T4*>(input)[gid];
    T4 t4 = reinterpret_cast<device const T4*>(target)[gid];
    vec<float, 4> d = float4(i4) - float4(t4);
    reinterpret_cast<device T4*>(out)[gid] = T4(d * d);
  } else {
    for (uint i = base; i < N; i++) {
      float d = float(input[i]) - float(target[i]);
      out[i] = T(d * d);
    }
  }
}

template <typename T>
kernel void mse_loss_fwd_reduce(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device float* partial [[buffer(2)]], // (n_tg,) float
    constant PointwiseLossParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  using T4 = vec<T, 4>;
  uint aligned_N = (p.N / 4u) * 4u;
  uint base = gid * 4u;
  uint vec_stride = tpg * 4u;
  while (base + 4u <= aligned_N) {
    T4 i4 = reinterpret_cast<device const T4*>(input)[base / 4u];
    T4 t4 = reinterpret_cast<device const T4*>(target)[base / 4u];
    float4 d4 = float4(i4) - float4(t4);
    acc += d4.x * d4.x + d4.y * d4.y + d4.z * d4.z + d4.w * d4.w;
    base += vec_stride;
  }
  for (uint i = aligned_N + gid; i < p.N; i += tpg) {
    float d = float(input[i]) - float(target[i]);
    acc += d * d;
  }
  acc = tg_sum(acc, smem, lid, tgsz);
  if (lid == 0)
    partial[tgid] = acc * p.scale;
}

template <typename T>
kernel void mse_loss_bwd(
    device const T* grad_out [[buffer(0)]], // scalar (reduce) or size-N (none)
    device const T* input [[buffer(1)]],
    device const T* target [[buffer(2)]],
    device T* grad_in [[buffer(3)]],
    constant PointwiseLossParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  float g_scalar =
      (p.reduction != 0) ? float(grad_out[0]) * p.scale * 2.f : 0.f;
  uint base = gid * 4u;
  if (base + 4u <= p.N) {
    T4 i4 = reinterpret_cast<device const T4*>(input)[gid];
    T4 t4 = reinterpret_cast<device const T4*>(target)[gid];
    float4 d4 = float4(i4) - float4(t4);
    float4 g4 = (p.reduction == 0)
        ? float4(reinterpret_cast<device const T4*>(grad_out)[gid]) * 2.f * d4
        : g_scalar * d4;
    reinterpret_cast<device T4*>(grad_in)[gid] = T4(g4);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float g =
          (p.reduction == 0) ? float(grad_out[i]) * 2.f * d : g_scalar * d;
      grad_in[i] = T(g);
    }
  }
}

#define INSTANTIATE_MSE(T)                                                    \
  template [[host_name("mse_loss_fwd_none_" #T)]]                             \
  kernel void mse_loss_fwd_none<T>(                                           \
      device const T*, device const T*, device T*, constant uint32_t&, uint); \
  template [[host_name("mse_loss_fwd_reduce_" #T)]]                           \
  kernel void mse_loss_fwd_reduce<T>(                                         \
      device const T*,                                                        \
      device const T*,                                                        \
      device float*,                                                          \
      constant PointwiseLossParams&,                                          \
      uint,                                                                   \
      uint,                                                                   \
      uint,                                                                   \
      uint,                                                                   \
      uint);                                                                  \
  template [[host_name("mse_loss_bwd_" #T)]]                                  \
  kernel void mse_loss_bwd<T>(                                                \
      device const T*,                                                        \
      device const T*,                                                        \
      device const T*,                                                        \
      device T*,                                                              \
      constant PointwiseLossParams&,                                          \
      uint);

INSTANTIATE_MSE(float)
INSTANTIATE_MSE(half)
INSTANTIATE_MSE(bfloat)

// ============================================================================
// SmoothL1 / Huber Loss  (is_huber flag selects formula)
// ============================================================================

template <typename T>
kernel void smooth_huber_fwd_none(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  T t_beta = T(p.beta);
  auto sh_elem = [&](T d) -> T {
    T ad = select(-d, d, d >= T(0));
    T l_q = p.is_huber ? T(0.5) * d * d : T(0.5) * d * d / t_beta;
    T l_l = p.is_huber ? t_beta * (ad - T(0.5) * t_beta) : ad - T(0.5) * t_beta;
    return select(l_l, l_q, ad < t_beta);
  };
  uint base = gid * 4u;
  if (base + 4u <= p.N) {
    using T4 = vec<T, 4>;
    T4 i4 = reinterpret_cast<device const T4*>(input)[gid];
    T4 t4 = reinterpret_cast<device const T4*>(target)[gid];
    T4 res;
    res[0] = sh_elem(i4[0] - t4[0]);
    res[1] = sh_elem(i4[1] - t4[1]);
    res[2] = sh_elem(i4[2] - t4[2]);
    res[3] = sh_elem(i4[3] - t4[3]);
    reinterpret_cast<device T4*>(out)[gid] = res;
  } else {
    for (uint i = base; i < p.N; i++) {
      out[i] = sh_elem(input[i] - target[i]);
    }
  }
}

template <typename T>
kernel void smooth_huber_fwd_reduce(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  using T4 = vec<T, 4>;
  auto loss_elem = [&](float d) -> float {
    float ad = abs(d);
    return p.is_huber
        ? (ad < p.beta ? 0.5f * d * d : p.beta * (ad - 0.5f * p.beta))
        : (ad < p.beta ? 0.5f * d * d / p.beta : ad - 0.5f * p.beta);
  };
  uint aligned_N = (p.N / 4u) * 4u;
  uint base = gid * 4u;
  uint vec_stride = tpg * 4u;
  while (base + 4u <= aligned_N) {
    T4 i4 = reinterpret_cast<device const T4*>(input)[base / 4u];
    T4 t4 = reinterpret_cast<device const T4*>(target)[base / 4u];
    float4 d4 = float4(i4) - float4(t4);
    acc += loss_elem(d4.x) + loss_elem(d4.y) + loss_elem(d4.z) + loss_elem(d4.w);
    base += vec_stride;
  }
  for (uint i = aligned_N + gid; i < p.N; i += tpg) {
    float d = float(input[i]) - float(target[i]);
    acc += loss_elem(d);
  }
  acc = tg_sum(acc, smem, lid, tgsz);
  if (lid == 0)
    partial[tgid] = acc * p.scale;
}

template <typename T>
kernel void smooth_huber_bwd(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* target [[buffer(2)]],
    device T* grad_in [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  auto dg_elem = [&](float d) -> float {
    float ad = abs(d);
    return p.is_huber ? (ad < p.beta ? d : p.beta * sign(d))
                      : (ad < p.beta ? d / p.beta : sign(d));
  };
  auto dg_vec4 = [&](float4 d4) -> float4 {
    return float4(
        dg_elem(d4[0]), dg_elem(d4[1]), dg_elem(d4[2]), dg_elem(d4[3]));
  };
  float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
  uint base = gid * 16u;
  if (base + 16u <= p.N) {
    uint t = gid * 4u;
    T4 i0 = reinterpret_cast<device const T4*>(input)[t + 0u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[t + 1u];
    T4 i2 = reinterpret_cast<device const T4*>(input)[t + 2u];
    T4 i3 = reinterpret_cast<device const T4*>(input)[t + 3u];
    T4 a0 = reinterpret_cast<device const T4*>(target)[t + 0u];
    T4 a1 = reinterpret_cast<device const T4*>(target)[t + 1u];
    T4 a2 = reinterpret_cast<device const T4*>(target)[t + 2u];
    T4 a3 = reinterpret_cast<device const T4*>(target)[t + 3u];
    float4 d0 = float4(i0) - float4(a0);
    float4 d1 = float4(i1) - float4(a1);
    float4 d2 = float4(i2) - float4(a2);
    float4 d3 = float4(i3) - float4(a3);
    float4 g0, g1, g2, g3;
    float4 dg0 = dg_vec4(d0), dg1 = dg_vec4(d1), dg2 = dg_vec4(d2),
           dg3 = dg_vec4(d3);
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 0u]) * dg0;
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 1u]) * dg1;
      g2 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 2u]) * dg2;
      g3 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 3u]) * dg3;
    } else {
      g0 = g_scalar * dg0;
      g1 = g_scalar * dg1;
      g2 = g_scalar * dg2;
      g3 = g_scalar * dg3;
    }
    reinterpret_cast<device T4*>(grad_in)[t + 0u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[t + 1u] = T4(g1);
    reinterpret_cast<device T4*>(grad_in)[t + 2u] = T4(g2);
    reinterpret_cast<device T4*>(grad_in)[t + 3u] = T4(g3);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float dg = dg_elem(d);
      float g = (p.reduction == 0) ? float(grad_out[i]) * dg : g_scalar * dg;
      grad_in[i] = T(g);
    }
  }
}

#define INSTANTIATE_SMOOTH_HUBER(T)                     \
  template [[host_name("smooth_huber_fwd_none_" #T)]]   \
  kernel void smooth_huber_fwd_none<T>(                 \
      device const T*,                                  \
      device const T*,                                  \
      device T*,                                        \
      constant SmoothHuberParams&,                      \
      uint);                                            \
  template [[host_name("smooth_huber_fwd_reduce_" #T)]] \
  kernel void smooth_huber_fwd_reduce<T>(               \
      device const T*,                                  \
      device const T*,                                  \
      device float*,                                    \
      constant SmoothHuberParams&,                      \
      uint,                                             \
      uint,                                             \
      uint,                                             \
      uint,                                             \
      uint);                                            \
  template [[host_name("smooth_huber_bwd_" #T)]]        \
  kernel void smooth_huber_bwd<T>(                      \
      device const T*,                                  \
      device const T*,                                  \
      device const T*,                                  \
      device T*,                                        \
      constant SmoothHuberParams&,                      \
      uint);

INSTANTIATE_SMOOTH_HUBER(float)
INSTANTIATE_SMOOTH_HUBER(half)
INSTANTIATE_SMOOTH_HUBER(bfloat)

// ============================================================================
// BCE Loss  (binary cross-entropy; input clamped to (eps, 1-eps))
// ============================================================================

template <typename T>
kernel void bce_loss_fwd_none(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant BCEParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  auto bce_elem = [&](float x_raw, float y, float w) -> float {
    float x = clamp(x_raw, 1e-7f, 1.f - 1e-7f);
    float l = -y * log(x) - (1.f - y) * log(1.f - x);
    return p.has_weight ? l * w : l;
  };
  uint base = gid * 4u;
  if (base + 4u <= p.N) {
    using T4 = vec<T, 4>;
    T4 i4 = reinterpret_cast<device const T4*>(input)[gid];
    T4 t4 = reinterpret_cast<device const T4*>(target)[gid];
    T4 w4 = reinterpret_cast<device const T4*>(weight)[gid];
    T4 res;
    res[0] = T(bce_elem(float(i4[0]), float(t4[0]), float(w4[0])));
    res[1] = T(bce_elem(float(i4[1]), float(t4[1]), float(w4[1])));
    res[2] = T(bce_elem(float(i4[2]), float(t4[2]), float(w4[2])));
    res[3] = T(bce_elem(float(i4[3]), float(t4[3]), float(w4[3])));
    reinterpret_cast<device T4*>(out)[gid] = res;
  } else {
    for (uint i = base; i < p.N; i++) {
      float x = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
      float y = float(target[i]);
      float w = p.has_weight ? float(weight[i]) : 1.f;
      out[i] = T(bce_elem(x, y, w));
    }
  }
}

template <typename T>
kernel void bce_loss_fwd_reduce(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant BCEParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256];
  float acc = 0.f;
  using T4 = vec<T, 4>;
  auto bce_elem = [&](float x_in, float y) -> float {
    float x = clamp(x_in, 1e-7f, 1.f - 1e-7f);
    return -y * log(x) - (1.f - y) * log(1.f - x);
  };
  uint aligned_N = (p.N / 4u) * 4u;
  uint base = gid * 4u;
  uint vec_stride = tpg * 4u;
  while (base + 4u <= aligned_N) {
    T4 i4 = reinterpret_cast<device const T4*>(input)[base / 4u];
    T4 t4 = reinterpret_cast<device const T4*>(target)[base / 4u];
    float4 x4 = float4(i4);
    float4 y4 = float4(t4);
    float4 l4 = float4(
        bce_elem(x4.x, y4.x),
        bce_elem(x4.y, y4.y),
        bce_elem(x4.z, y4.z),
        bce_elem(x4.w, y4.w));
    if (p.has_weight) {
      T4 w4 = reinterpret_cast<device const T4*>(weight)[base / 4u];
      float4 wf4 = float4(w4);
      l4 *= wf4;
    }
    acc += l4.x + l4.y + l4.z + l4.w;
    base += vec_stride;
  }
  for (uint i = aligned_N + gid; i < p.N; i += tpg) {
    float x = clamp(float(input[i]), 1e-7f, 1.f - 1e-7f);
    float y = float(target[i]);
    float l = -y * log(x) - (1.f - y) * log(1.f - x);
    acc += p.has_weight ? l * float(weight[i]) : l;
  }
  acc = tg_sum(acc, smem, lid, tgsz);
  if (lid == 0)
    partial[tgid] = acc * p.scale;
}

// REVERTED to v2 (the high-water mark): single kernel, 16-elem hand-unrolled
// batched-loads, runtime if (p.reduction == 0) branch. Each subsequent
// iteration (v3 32-elem unified, v4/v5 split, v6 vec<T,8>) was a net
// regression vs this form. Further changes will be informed by Metal frame
// capture data, not hypothesis.
template <typename T>
kernel void bce_loss_bwd(
    device const T* grad_out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    device const T* target [[buffer(2)]],
    device const T* weight [[buffer(3)]],
    device T* grad_in [[buffer(4)]],
    constant BCEParams& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  auto dx_vec4 = [](float4 x_raw, float4 y) -> float4 {
    // Match CPU semantics: numerator uses raw x (so grad=0 at x=y=0 or x=y=1),
    // denominator uses clamped x for numerical stability.
    float4 x = clamp(x_raw, 1e-7f, 1.f - 1e-7f);
    return (x_raw - y) / (x * (1.f - x));
  };
  float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
  uint base = gid * 16u;
  if (base + 16u <= p.N) {
    uint t = gid * 4u;
    T4 i0 = reinterpret_cast<device const T4*>(input)[t + 0u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[t + 1u];
    T4 i2 = reinterpret_cast<device const T4*>(input)[t + 2u];
    T4 i3 = reinterpret_cast<device const T4*>(input)[t + 3u];
    T4 a0 = reinterpret_cast<device const T4*>(target)[t + 0u];
    T4 a1 = reinterpret_cast<device const T4*>(target)[t + 1u];
    T4 a2 = reinterpret_cast<device const T4*>(target)[t + 2u];
    T4 a3 = reinterpret_cast<device const T4*>(target)[t + 3u];
    float4 dx0 = dx_vec4(float4(i0), float4(a0));
    float4 dx1 = dx_vec4(float4(i1), float4(a1));
    float4 dx2 = dx_vec4(float4(i2), float4(a2));
    float4 dx3 = dx_vec4(float4(i3), float4(a3));
    if (p.has_weight) {
      dx0 *= float4(reinterpret_cast<device const T4*>(weight)[t + 0u]);
      dx1 *= float4(reinterpret_cast<device const T4*>(weight)[t + 1u]);
      dx2 *= float4(reinterpret_cast<device const T4*>(weight)[t + 2u]);
      dx3 *= float4(reinterpret_cast<device const T4*>(weight)[t + 3u]);
    }
    float4 g0, g1, g2, g3;
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 0u]) * dx0;
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 1u]) * dx1;
      g2 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 2u]) * dx2;
      g3 = float4(reinterpret_cast<device const T4*>(grad_out)[t + 3u]) * dx3;
    } else {
      g0 = g_scalar * dx0;
      g1 = g_scalar * dx1;
      g2 = g_scalar * dx2;
      g3 = g_scalar * dx3;
    }
    reinterpret_cast<device T4*>(grad_in)[t + 0u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[t + 1u] = T4(g1);
    reinterpret_cast<device T4*>(grad_in)[t + 2u] = T4(g2);
    reinterpret_cast<device T4*>(grad_in)[t + 3u] = T4(g3);
  } else {
    for (uint i = base; i < p.N; i++) {
      float x_raw = float(input[i]);
      float x = clamp(x_raw, 1e-7f, 1.f - 1e-7f);
      float dx = (x_raw - float(target[i])) / (x * (1.f - x));
      if (p.has_weight)
        dx *= float(weight[i]);
      float g = (p.reduction == 0) ? float(grad_out[i]) * dx : g_scalar * dx;
      grad_in[i] = T(g);
    }
  }
}

#define INSTANTIATE_BCE(T)                          \
  template [[host_name("bce_loss_fwd_none_" #T)]]   \
  kernel void bce_loss_fwd_none<T>(                 \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device T*,                                    \
      constant BCEParams&,                          \
      uint);                                        \
  template [[host_name("bce_loss_fwd_reduce_" #T)]] \
  kernel void bce_loss_fwd_reduce<T>(               \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device float*,                                \
      constant BCEParams&,                          \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint);                                        \
  template [[host_name("bce_loss_bwd_" #T)]]        \
  kernel void bce_loss_bwd<T>(                      \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device const T*,                              \
      device T*,                                    \
      constant BCEParams&,                          \
      uint);

INSTANTIATE_BCE(float)
INSTANTIATE_BCE(half)
INSTANTIATE_BCE(bfloat)

// ============================================================================
// NLL Loss 1-D  (input (N,C) log-probs, target (N,) int class indices)
// C++ handles final scale (Mean denominator) after phase-2 reduction.
// Caller must pre-zero grad_in before dispatching nll_loss_bwd.
// ============================================================================

template <typename T>
kernel void nll_loss_fwd_none(
    device const T* log_prob [[buffer(0)]], // (N, C)
    device const int* target [[buffer(1)]], // (N,)
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]], // (N,)
    constant NLLParams& p [[buffer(4)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint n = gid; n < p.N; n += tpg) {
    int t = target[n];
    if (t == p.ignore_index) {
      out[n] = T(0);
      continue;
    }
    if (t < 0 || t >= int(p.C)) {
      TORCH_REPORT_ERROR(error_buf, "nll_loss: Target ", t, " is out of bounds [0, ", int(p.C), ")");
      out[n] = T(0);
      continue;
    }
    float l = -float(log_prob[n * p.C + uint32_t(t)]);
    out[n] = T(p.has_weight ? l * float(weight[t]) : l);
  }
}

template <typename T>
kernel void nll_loss_fwd_reduce(
    device const T* log_prob [[buffer(0)]],
    device const int* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device float* partial [[buffer(3)]], // (n_tg,) loss sums
    device float* wpartial [[buffer(4)]], // (n_tg,) weight sums
    constant NLLParams& p [[buffer(5)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup float smem[256], wsmem[256];
  float acc = 0.f, wacc = 0.f;
  for (uint n = gid; n < p.N; n += tpg) {
    int t = target[n];
    if (t == p.ignore_index)
      continue;
    if (t < 0 || t >= int(p.C)) {
      TORCH_REPORT_ERROR(error_buf, "nll_loss: Target ", t, " is out of bounds [0, ", int(p.C), ")");
      continue;
    }
    float w = p.has_weight ? float(weight[t]) : 1.f;
    acc += -float(log_prob[n * p.C + uint32_t(t)]) * w;
    wacc += w;
  }
  acc = tg_sum(acc, smem, lid, tgsz);
  wacc = tg_sum(wacc, wsmem, lid, tgsz);
  if (lid == 0) {
    partial[tgid] = acc;
    wpartial[tgid] = wacc;
  }
}

// Backward: writes -grad_out_scaled to grad_in[n, target[n]].
// Caller zeros grad_in before dispatch; each thread handles one n.
template <typename T>
kernel void nll_loss_bwd(
    device const T* grad_out [[buffer(0)]], // scalar (reduce) or (N,)
    device const int* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* grad_in [[buffer(3)]], // (N, C) pre-zeroed
    device const T* total_w [[buffer(4)]], // scalar weight sum (Mean)
    constant NLLParams& p [[buffer(5)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint n = gid; n < p.N; n += tpg) {
    int t = target[n];
    if (t == p.ignore_index)
      continue;
    if (t < 0 || t >= int(p.C)) {
      TORCH_REPORT_ERROR(error_buf, "nll_loss: Target ", t, " is out of bounds [0, ", int(p.C), ")");
      continue;
    }
    float w = p.has_weight ? float(weight[t]) : 1.f;
    float scale;
    if (p.reduction == 0) {
      scale = -float(grad_out[n]) * w;
    } else if (p.reduction == 1) {
      float denom = (float(total_w[0]) != 0.f) ? float(total_w[0]) : float(p.N);
      scale = -float(grad_out[0]) * w / denom;
    } else {
      scale = -float(grad_out[0]) * w;
    }
    grad_in[n * p.C + uint32_t(t)] = T(scale);
  }
}

#define INSTANTIATE_NLL(T)                          \
  template [[host_name("nll_loss_fwd_none_" #T)]]   \
  kernel void nll_loss_fwd_none<T>(                 \
      device const T*,                              \
      device const int*,                            \
      device const T*,                              \
      device T*,                                    \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint);                                        \
  template [[host_name("nll_loss_fwd_reduce_" #T)]] \
  kernel void nll_loss_fwd_reduce<T>(               \
      device const T*,                              \
      device const int*,                            \
      device const T*,                              \
      device float*,                                \
      device float*,                                \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint,                                         \
      uint);                                        \
  template [[host_name("nll_loss_bwd_" #T)]]        \
  kernel void nll_loss_bwd<T>(                      \
      device const T*,                              \
      device const int*,                            \
      device const T*,                              \
      device T*,                                    \
      device const T*,                              \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint);

INSTANTIATE_NLL(float)
INSTANTIATE_NLL(half)
INSTANTIATE_NLL(bfloat)

// Phase-3 for NLL Mean reduction: divide typed output[0] by typed
// total_weight[0]. Dispatched with a single thread after the two
// encode_reduce_partials calls.
template <typename T>
kernel void nll_finalize_mean(
    device T* loss [[buffer(0)]],
    device const T* total_weight [[buffer(1)]]) {
  float tw = float(total_weight[0]);
  loss[0] = (tw == 0.f) ? T(NAN) : T(float(loss[0]) / tw);
}

#define INST_NLL_FINALIZE(T)                      \
  template [[host_name("nll_finalize_mean_" #T)]] \
  kernel void nll_finalize_mean<T>(device T*, device const T*)
INST_NLL_FINALIZE(float);
INST_NLL_FINALIZE(half);
INST_NLL_FINALIZE(bfloat);

// ============================================================================
// Fused Cross-Entropy  (online Milakov-Gimelshein, one threadgroup per sample)
// lse_buf (N,) saved forward for backward; C++ reduces partial[] to scalar.
// ============================================================================

template <typename T>
kernel void cross_entropy_fwd(
    device const T* logits [[buffer(0)]], // (N, C)
    device const int* target [[buffer(1)]], // (N,)
    device float* partial [[buffer(2)]], // (N,) per-sample loss
    device float* lse_buf [[buffer(3)]], // (N,) log-sum-exp
    constant NLLParams& p [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  uint n = tgid;
  if (n >= p.N)
    return;
  device const T* row = logits + n * p.C;

  threadgroup float smem_m[256];
  threadgroup float smem_s[256];

  // Per-thread online LSE (single pass, numerically stable)
  float lm = -INFINITY, ls = 0.f;
  for (uint c = lid; c < p.C; c += tgsz) {
    float x = float(row[c]);
    float nm = max(lm, x);
    ls = ls * exp(lm - nm) + exp(x - nm);
    lm = nm;
  }

  // Tree-reduce (m, s) pairs
  smem_m[lid] = lm;
  smem_s[lid] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tgsz >> 1; s > 0; s >>= 1) {
    if (lid < s) {
      float ma = smem_m[lid], mb = smem_m[lid + s];
      float sa = smem_s[lid], sb = smem_s[lid + s];
      float nm = max(ma, mb);
      smem_m[lid] = nm;
      smem_s[lid] = sa * exp(ma - nm) + sb * exp(mb - nm);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float lse = log(smem_s[0]) + smem_m[0];

  if (lid == 0) {
    int t = target[n];
    float tv = (t != p.ignore_index) ? float(row[t]) : 0.f;
    partial[n] = (t != p.ignore_index) ? lse - tv : 0.f;
    lse_buf[n] = lse;
  }
}

template <typename T>
kernel void cross_entropy_bwd(
    device const T* grad_out [[buffer(0)]], // scalar or (N,)
    device const T* logits [[buffer(1)]], // (N, C)
    device const int* target [[buffer(2)]], // (N,)
    device const float* lse_buf [[buffer(3)]], // (N,)
    device T* grad_in [[buffer(4)]], // (N, C)
    constant NLLParams& p [[buffer(5)]],
    constant float& inv_scale [[buffer(6)]], // 1/N (Mean) or 1.0 (Sum/None)
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint idx = gid; idx < p.N * p.C; idx += tpg) {
    uint n = idx / p.C;
    uint c = idx % p.C;
    int t = target[n];
    float sm = exp(float(logits[idx]) - lse_buf[n]);
    float ind = (t != p.ignore_index && c == uint32_t(t)) ? 1.f : 0.f;
    float gout = (p.reduction == 0) ? float(grad_out[n]) : float(grad_out[0]);
    grad_in[idx] = T(gout * inv_scale * (sm - ind));
  }
}

#define INSTANTIATE_CE(T)                         \
  template [[host_name("cross_entropy_fwd_" #T)]] \
  kernel void cross_entropy_fwd<T>(               \
      device const T*,                            \
      device const int*,                          \
      device float*,                              \
      device float*,                              \
      constant NLLParams&,                        \
      uint,                                       \
      uint,                                       \
      uint);                                      \
  template [[host_name("cross_entropy_bwd_" #T)]] \
  kernel void cross_entropy_bwd<T>(               \
      device const T*,                            \
      device const T*,                            \
      device const int*,                          \
      device const float*,                        \
      device T*,                                  \
      constant NLLParams&,                        \
      constant float&,                            \
      uint,                                       \
      uint);

INSTANTIATE_CE(float)
INSTANTIATE_CE(half)
INSTANTIATE_CE(bfloat)

// ============================================================================
// FUSED FWD + SAVED-GRAD KERNELS (kernel fusion for fwd+bwd autograd path)
// ============================================================================
// Forward writes loss AND a saved gradient factor; backward reads only
// grad_out + saved_dg, avoiding the second round-trip to load x, y.
// Wins only kick in at reduction=none + N >= 1M; the C++ dispatcher gates
// these via kFusionMinNumel and falls through to the standard custom kernel
// otherwise (which is already optimized for mean/sum via fwd_reduce).

template <typename T>
kernel void huber_fwd_sg(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    device T* saved_dg [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const float fbeta = p.beta;
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  if (base + VEC <= p.N) {
    T4 i0 = reinterpret_cast<device const T4*>(input)[gid * 2u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[gid * 2u + 1u];
    T4 t0 = reinterpret_cast<device const T4*>(target)[gid * 2u];
    T4 t1 = reinterpret_cast<device const T4*>(target)[gid * 2u + 1u];
    float4 d0 = float4(i0) - float4(t0);
    float4 d1 = float4(i1) - float4(t1);
    float4 dg0 = clamp(d0, float4(-fbeta), float4(fbeta));
    float4 dg1 = clamp(d1, float4(-fbeta), float4(fbeta));
    float4 ad0 = abs(d0), ad1 = abs(d1);
    float4 q0 = min(ad0, float4(fbeta)), q1 = min(ad1, float4(fbeta));
    float4 lin0 = ad0 - q0, lin1 = ad1 - q1;
    float4 l0 = 0.5f * q0 * q0 + float4(fbeta) * lin0;
    float4 l1 = 0.5f * q1 * q1 + float4(fbeta) * lin1;
    reinterpret_cast<device T4*>(out)[gid * 2u] = T4(l0);
    reinterpret_cast<device T4*>(out)[gid * 2u + 1u] = T4(l1);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u] = T4(dg0);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u + 1u] = T4(dg1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float dg = clamp(d, -fbeta, fbeta);
      float ad = abs(d);
      float q = min(ad, fbeta);
      float lin = ad - q;
      out[i] = T(0.5f * q * q + fbeta * lin);
      saved_dg[i] = T(dg);
    }
  }
}

template <typename T>
kernel void smooth_l1_fwd_sg(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    device T* saved_dg [[buffer(3)]],
    constant SmoothHuberParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const float fbeta = p.beta;
  const float inv_beta = 1.0f / fbeta;
  const float inv_2beta = 0.5f * inv_beta;
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  if (base + VEC <= p.N) {
    T4 i0 = reinterpret_cast<device const T4*>(input)[gid * 2u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[gid * 2u + 1u];
    T4 t0 = reinterpret_cast<device const T4*>(target)[gid * 2u];
    T4 t1 = reinterpret_cast<device const T4*>(target)[gid * 2u + 1u];
    float4 d0 = float4(i0) - float4(t0);
    float4 d1 = float4(i1) - float4(t1);
    float4 dg0 = clamp(d0, float4(-fbeta), float4(fbeta)) * float4(inv_beta);
    float4 dg1 = clamp(d1, float4(-fbeta), float4(fbeta)) * float4(inv_beta);
    float4 ad0 = abs(d0), ad1 = abs(d1);
    float4 q0 = min(ad0, float4(fbeta)), q1 = min(ad1, float4(fbeta));
    float4 l0 = q0 * q0 * float4(inv_2beta) + (ad0 - q0);
    float4 l1 = q1 * q1 * float4(inv_2beta) + (ad1 - q1);
    reinterpret_cast<device T4*>(out)[gid * 2u] = T4(l0);
    reinterpret_cast<device T4*>(out)[gid * 2u + 1u] = T4(l1);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u] = T4(dg0);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u + 1u] = T4(dg1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(input[i]) - float(target[i]);
      float dg = clamp(d, -fbeta, fbeta) * inv_beta;
      float ad = abs(d);
      float q = min(ad, fbeta);
      out[i] = T(q * q * inv_2beta + (ad - q));
      saved_dg[i] = T(dg);
    }
  }
}

template <typename T>
kernel void huber_or_sl1_bwd_sg(
    device const T* grad_out [[buffer(0)]],
    device const T* saved_dg [[buffer(1)]],
    device T* grad_in [[buffer(2)]],
    constant SmoothHuberParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  float g_scalar = (p.reduction != 0) ? float(grad_out[0]) * p.scale : 0.f;
  if (base + VEC <= p.N) {
    T4 dg0 = reinterpret_cast<device const T4*>(saved_dg)[gid * 2u];
    T4 dg1 = reinterpret_cast<device const T4*>(saved_dg)[gid * 2u + 1u];
    float4 g0, g1;
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[gid * 2u]) *
          float4(dg0);
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[gid * 2u + 1u]) *
          float4(dg1);
    } else {
      g0 = g_scalar * float4(dg0);
      g1 = g_scalar * float4(dg1);
    }
    reinterpret_cast<device T4*>(grad_in)[gid * 2u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[gid * 2u + 1u] = T4(g1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float dg = float(saved_dg[i]);
      float gout = (p.reduction == 0) ? float(grad_out[i]) : g_scalar;
      grad_in[i] = T(gout * dg);
    }
  }
}

template <typename T>
kernel void mse_fwd_sg(
    device const T* input [[buffer(0)]],
    device const T* target [[buffer(1)]],
    device T* out [[buffer(2)]],
    device T* saved_dg [[buffer(3)]],
    constant uint32_t& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  if (base + VEC <= N) {
    T4 i0 = reinterpret_cast<device const T4*>(input)[gid * 2u];
    T4 i1 = reinterpret_cast<device const T4*>(input)[gid * 2u + 1u];
    T4 t0 = reinterpret_cast<device const T4*>(target)[gid * 2u];
    T4 t1 = reinterpret_cast<device const T4*>(target)[gid * 2u + 1u];
    float4 d0 = float4(i0) - float4(t0);
    float4 d1 = float4(i1) - float4(t1);
    reinterpret_cast<device T4*>(out)[gid * 2u] = T4(d0 * d0);
    reinterpret_cast<device T4*>(out)[gid * 2u + 1u] = T4(d1 * d1);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u] = T4(d0);
    reinterpret_cast<device T4*>(saved_dg)[gid * 2u + 1u] = T4(d1);
  } else {
    for (uint i = base; i < N; i++) {
      float d = float(input[i]) - float(target[i]);
      out[i] = T(d * d);
      saved_dg[i] = T(d);
    }
  }
}

template <typename T>
kernel void mse_bwd_sg(
    device const T* grad_out [[buffer(0)]],
    device const T* saved_dg [[buffer(1)]],
    device T* grad_in [[buffer(2)]],
    constant PointwiseLossParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  using T4 = vec<T, 4>;
  constexpr uint VEC = 8u;
  const uint base = gid * VEC;
  float g_scalar =
      (p.reduction != 0) ? float(grad_out[0]) * p.scale * 2.f : 0.f;
  if (base + VEC <= p.N) {
    T4 d0 = reinterpret_cast<device const T4*>(saved_dg)[gid * 2u];
    T4 d1 = reinterpret_cast<device const T4*>(saved_dg)[gid * 2u + 1u];
    float4 g0, g1;
    if (p.reduction == 0) {
      g0 = float4(reinterpret_cast<device const T4*>(grad_out)[gid * 2u]) *
          2.f * float4(d0);
      g1 = float4(reinterpret_cast<device const T4*>(grad_out)[gid * 2u + 1u]) *
          2.f * float4(d1);
    } else {
      g0 = g_scalar * float4(d0);
      g1 = g_scalar * float4(d1);
    }
    reinterpret_cast<device T4*>(grad_in)[gid * 2u] = T4(g0);
    reinterpret_cast<device T4*>(grad_in)[gid * 2u + 1u] = T4(g1);
  } else {
    for (uint i = base; i < p.N; i++) {
      float d = float(saved_dg[i]);
      float g =
          (p.reduction == 0) ? float(grad_out[i]) * 2.f * d : g_scalar * d;
      grad_in[i] = T(g);
    }
  }
}

template [[host_name("huber_fwd_sg_half")]]
kernel void huber_fwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    device half*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_fwd_sg_bfloat")]]
kernel void huber_fwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    device bfloat*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_fwd_sg_float")]]
kernel void huber_fwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    device float*,
    constant SmoothHuberParams&,
    uint);

template [[host_name("smooth_l1_fwd_sg_half")]]
kernel void smooth_l1_fwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    device half*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("smooth_l1_fwd_sg_bfloat")]]
kernel void smooth_l1_fwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    device bfloat*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("smooth_l1_fwd_sg_float")]]
kernel void smooth_l1_fwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    device float*,
    constant SmoothHuberParams&,
    uint);

template [[host_name("huber_or_sl1_bwd_sg_half")]]
kernel void huber_or_sl1_bwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_or_sl1_bwd_sg_bfloat")]]
kernel void huber_or_sl1_bwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    constant SmoothHuberParams&,
    uint);
template [[host_name("huber_or_sl1_bwd_sg_float")]]
kernel void huber_or_sl1_bwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    constant SmoothHuberParams&,
    uint);

template [[host_name("mse_fwd_sg_half")]]
kernel void mse_fwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    device half*,
    constant uint32_t&,
    uint);
template [[host_name("mse_fwd_sg_bfloat")]]
kernel void mse_fwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    device bfloat*,
    constant uint32_t&,
    uint);
template [[host_name("mse_fwd_sg_float")]]
kernel void mse_fwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    device float*,
    constant uint32_t&,
    uint);

template [[host_name("mse_bwd_sg_half")]]
kernel void mse_bwd_sg<half>(
    device const half*,
    device const half*,
    device half*,
    constant PointwiseLossParams&,
    uint);
template [[host_name("mse_bwd_sg_bfloat")]]
kernel void mse_bwd_sg<bfloat>(
    device const bfloat*,
    device const bfloat*,
    device bfloat*,
    constant PointwiseLossParams&,
    uint);
template [[host_name("mse_bwd_sg_float")]]
kernel void mse_bwd_sg<float>(
    device const float*,
    device const float*,
    device float*,
    constant PointwiseLossParams&,
    uint);
