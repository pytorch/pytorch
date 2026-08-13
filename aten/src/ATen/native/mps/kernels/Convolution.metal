// Templated conv3d kernels. conv3d_mpp bakes popular geometries into bundled
// entry points; catalog misses use conv3d_simd.
#include <ATen/native/mps/kernels/Convolution.h>
#include <c10/metal/utils.h>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// Tiled NCDHW -> NDHWC transpose; PyTorch's generic strided copy runs well
// below memory bandwidth for this permutation.
template <typename T, int TC, int TX, int NTH, bool VECR, bool VECW>
kernel void nchw_to_nhwc(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant int2& dims [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup T tile[TC][TX + 1];
  const int C = dims.x, X = dims.y;
  const int c0 = int(tgid.y) * TC;
  const int x0 = int(tgid.x) * TX;
  device const T* s = src + (int64_t)tgid.z * C * X;
  device T* d = dst + (int64_t)tgid.z * C * X;

  if IF_CONSTEXPR (VECR) {
    for (int i = tid; i < TC * (TX / 2); i += NTH) {
      const int x = (i % (TX / 2)) * 2, c = i / (TX / 2);
      const int gc = c0 + c, gx = x0 + x;
      if (gc >= C || gx >= X) {
        continue;
      }
      const int64_t off = (int64_t)gc * X + gx;
      if (gx + 1 < X) {
        vec<T, 2> v = *(device const vec<T, 2>*)(s + off);
        tile[c][x] = v.x;
        tile[c][x + 1] = v.y;
      } else {
        tile[c][x] = s[off];
      }
    }
  } else {
    for (int i = tid; i < TC * TX; i += NTH) {
      const int x = i % TX, c = i / TX;
      const int gc = c0 + c, gx = x0 + x;
      if (gc < C && gx < X) {
        tile[c][x] = s[(int64_t)gc * X + gx];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if IF_CONSTEXPR (VECW) {
    for (int i = tid; i < (TC / 2) * TX; i += NTH) {
      const int c = (i % (TC / 2)) * 2, x = i / (TC / 2);
      const int gc = c0 + c, gx = x0 + x;
      if (gx >= X || gc >= C) {
        continue;
      }
      const int64_t off = (int64_t)gx * C + gc;
      if (gc + 1 < C) {
        *(device vec<T, 2>*)(d + off) = vec<T, 2>(tile[c][x], tile[c + 1][x]);
      } else {
        d[off] = tile[c][x];
      }
    }
  } else {
    for (int i = tid; i < TC * TX; i += NTH) {
      const int c = i % TC, x = i / TC;
      const int gc = c0 + c, gx = x0 + x;
      if (gc < C && gx < X) {
        d[(int64_t)gx * C + gc] = tile[c][x];
      }
    }
  }
}

// Implicit-GEMM conv3d on simdgroup_matrix per (batch, out-depth) plane; k
// walks DHWIO order, geometry is runtime state, fp32 accumulation. IDX = long
// widens the flat spatial index when a plane exceeds int32.
template <typename T, typename IDX, int BM, int BN, int BK, int WM, int WN>
kernel void conv3d_simd(
    device T* act [[buffer(0)]],
    device T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv3DParams& gP [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constant Conv2DParams& p = gP.conv2d;
  constexpr int TGP = WM * WN * 32;
  constexpr int WT_M = BM / WM, WT_N = BN / WN;
  constexpr int TM = WT_M / 8, TN = WT_N / 8;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  static_assert(BM % (8 * WM) == 0 && BN % (8 * WN) == 0 && BK % 8 == 0, "");
  static_assert(TGP % BK == 0 && TGP % BN == 0, "cooperative load mappings");

  threadgroup T As[BM * LDA];
  threadgroup T Bs[BK * LDB];

  const int tid = int(sgid) * 32 + int(lane);
  const int n_tiles = p.C_out_per_group / BN + (p.C_out_per_group % BN != 0);
  const int g = int(tgid.x) / n_tiles;
  const int n_block = (int(tgid.x) % n_tiles) * BN;
  const IDX m_block = IDX(tgid.y) * BM;
  const int dd = int(tgid.z) % gP.outD;
  const int nb = int(tgid.z) / gP.outD;
  const int c0 = g * p.C_in_per_group;
  const int o0 = g * p.C_out_per_group;

  const IDX M = IDX(p.outH) * p.outW;
  const int K = gP.kD * p.kH * p.kW * p.C_in_per_group;
  const int64_t act_nb = (int64_t)nb * gP.D * p.H * p.W * p.C_in;

  simdgroup_matrix<float, 8, 8> Cfrag[TM][TN];
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      Cfrag[i][j] = simdgroup_matrix<float, 8, 8>(0);
    }
  }
  const int warp_m = (int(sgid) / WN) * WT_M;
  const int warp_n = (int(sgid) % WN) * WT_N;

  // fixed k-column / n-column per thread so the k -> (c, kw, kh, kd)
  // decomposition happens once per K-tile, not once per element
  const int a_k = tid % BK;
  const int a_r0 = tid / BK;
  const int b_n = tid % BN;
  const int b_r0 = tid / BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int k = k0 + a_k;
    int di = 0, dh = 0, dw = 0, cc = 0;
    bool k_ok = k < K;
    if (k_ok) {
      cc = k % p.C_in_per_group;
      int r = k / p.C_in_per_group;
      const int kw = r % p.kW;
      r /= p.kW;
      const int kh = r % p.kH;
      const int kd = r / p.kH;
      di = dd * gP.sD - gP.padD + kd * gP.dD;
      dh = kh * p.dH - p.padH;
      dw = kw * p.dW - p.padW;
      k_ok = di >= 0 && di < gP.D;
    }
    for (int rl = a_r0; rl < BM; rl += TGP / BK) {
      const IDX m = m_block + rl;
      T v = T(0);
      if (k_ok && m < M) {
        const int hi = int(m / p.outW) * p.sH + dh;
        const int wi = int(m % p.outW) * p.sW + dw;
        if (hi >= 0 && hi < p.H && wi >= 0 && wi < p.W) {
          v =
              act[act_nb + (((int64_t)di * p.H + hi) * p.W + wi) * p.C_in + c0 +
                  cc];
        }
      }
      As[rl * LDA + a_k] = v;
    }
    for (int rl = b_r0; rl < BK; rl += TGP / BN) {
      const int kb = k0 + rl;
      const int n = n_block + b_n;
      Bs[rl * LDB + b_n] = (kb < K && n < p.C_out_per_group)
          ? wts[(int64_t)kb * p.C_out + o0 + n]
          : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<T, 8, 8> Afrag[TM];
      simdgroup_matrix<T, 8, 8> Bfrag[TN];
      for (int i = 0; i < TM; ++i) {
        simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA + kk], LDA);
      }
      for (int j = 0; j < TN; ++j) {
        simdgroup_load(Bfrag[j], &Bs[kk * LDB + warp_n + j * 8], LDB);
      }
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
      }
    }
  }

  // per-lane (row, col) ownership within an 8x8 fragment; thread_elements()
  // holds (row, col) and (row, col + 1)
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      const IDX m = m_block + warp_m + i * 8 + fm;
      if (m >= M) {
        continue;
      }
      const int ho = int(m / p.outW), wo = int(m % p.outW);
      for (int e = 0; e < 2; ++e) {
        const int n = n_block + warp_n + j * 8 + fn + e;
        if (n >= p.C_out_per_group) {
          continue;
        }
        float v = Cfrag[i][j].thread_elements()[e];
        const int o = o0 + n;
        if (p.has_bias) {
          v += (float)bias[o];
        }
        int64_t idx;
        if (gP.out_ncdhw) {
          idx = ((((int64_t)nb * p.C_out + o) * gP.outD + dd) * p.outH + ho) *
                  p.outW +
              wo;
        } else {
          idx = ((((int64_t)nb * gP.outD + dd) * p.outH + ho) * p.outW + wo) *
                  p.C_out +
              o;
        }
        dst[idx] = (T)v;
      }
    }
  }
}

#define INSTANTIATE_CONV3D_SIMD(PFX, DT, IDX, BM, BN, BK, WM, WN)            \
  template                                                                   \
      [[host_name(PFX "_" #DT "_" #BM "_" #BN "_" #WM "_" #WN)]] kernel void \
      conv3d_simd<DT, IDX, BM, BN, BK, WM, WN>(                              \
          device DT*,                                                        \
          device DT*,                                                        \
          device DT*,                                                        \
          constant Conv3DParams&,                                            \
          device const DT*,                                                  \
          uint3,                                                             \
          uint,                                                              \
          uint);

// int covers planes within int32; one long variant handles the rest
#define INSTANTIATE_CONV3D_SIMD_ALL(DT)                             \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd", DT, int, 64, 64, 16, 2, 2) \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd", DT, int, 32, 64, 16, 1, 2) \
  INSTANTIATE_CONV3D_SIMD("conv3d_simd_long", DT, long, 64, 64, 16, 2, 2)

INSTANTIATE_CONV3D_SIMD_ALL(float)
INSTANTIATE_CONV3D_SIMD_ALL(half)
INSTANTIATE_CONV3D_SIMD_ALL(bfloat)

#define INSTANTIATE_NCHW_TO_NHWC(DT, TC, TX, VECR, VECW)             \
  template [[host_name("nchw_to_nhwc_" #DT "_" #TC "_" #TX "_" #VECR \
                       "_" #VECW)]] kernel void                      \
  nchw_to_nhwc<DT, TC, TX, 256, VECR, VECW>(                         \
      device const DT*, device DT*, constant int2&, uint3, uint);

#define INSTANTIATE_NCHW_TO_NHWC_ALL(DT)              \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, false, false)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, false, true)   \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, true, false)   \
  INSTANTIATE_NCHW_TO_NHWC(DT, 16, 64, true, true)    \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, false, false)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, false, true)   \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, true, false)   \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 32, true, true)    \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 128, false, false) \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 128, false, true)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 128, true, false)  \
  INSTANTIATE_NCHW_TO_NHWC(DT, 32, 128, true, true)

INSTANTIATE_NCHW_TO_NHWC_ALL(float)
INSTANTIATE_NCHW_TO_NHWC_ALL(half)
INSTANTIATE_NCHW_TO_NHWC_ALL(bfloat)

// Depthwise conv1d directly on NCL: rows are per-channel independent, so the
// NDHWC transpose the matmul kernels need would be pure overhead here.
template <typename T>
kernel void conv1d_dw(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv1dDwParams& params [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 position [[thread_position_in_grid]]) {
  // the host puts the output's stride-1 axis on x
  const int output_index = int(params.swap_grid ? position.y : position.x);
  const int channel = int(params.swap_grid ? position.x : position.y);
  const int batch = int(position.z);
  float accumulator = params.has_bias ? (float)bias[channel] : 0.0f;
  device const T* input_batch = input +
      (int64_t)batch * params.input_channels * params.input_length +
      (int64_t)(channel / params.channel_multiplier) * params.in_channel_stride;
  const int input_start = output_index * params.stride - params.padding;
  for (int kernel_index = 0; kernel_index < params.kernel_size;
       ++kernel_index) {
    const int input_index = input_start + kernel_index * params.dilation;
    if (input_index >= 0 && input_index < params.input_length) {
      accumulator +=
          (float)input_batch[(int64_t)input_index * params.in_pos_stride] *
          (float)weight[(int64_t)channel * params.kernel_size + kernel_index];
    }
  }
  output
      [(int64_t)batch * params.input_channels * params.channel_multiplier *
           params.output_length +
       (int64_t)channel * params.out_channel_stride +
       (int64_t)output_index * params.out_pos_stride] = (T)accumulator;
}

template <typename T, bool CHECK_BOUNDS>
kernel void conv1d_dw_vec(
    device const T* input [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant Conv1dDwParams& params [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 position [[thread_position_in_grid]]) {
  const int output_start = int(position.x) * conv1d_dw_outputs_per_thread;
  const int channel = int(position.y);
  const int batch = int(position.z);
  const float initial = params.has_bias ? (float)bias[channel] : 0.0f;
  float accumulators[conv1d_dw_outputs_per_thread];
#pragma unroll
  for (int lane = 0; lane < conv1d_dw_outputs_per_thread; ++lane) {
    accumulators[lane] = initial;
  }
  device const T* input_row = input +
      ((int64_t)batch * params.input_channels +
       channel / params.channel_multiplier) *
          params.input_length;
  for (int kernel_index = 0; kernel_index < params.kernel_size;
       ++kernel_index) {
    const int input_start =
        output_start - params.padding + kernel_index * params.dilation;
    const float weight_value =
        (float)weight[(int64_t)channel * params.kernel_size + kernel_index];
#pragma unroll
    for (int lane = 0; lane < conv1d_dw_outputs_per_thread; ++lane) {
      const int input_index = input_start + lane;
      if IF_CONSTEXPR (!CHECK_BOUNDS) {
        accumulators[lane] += (float)input_row[input_index] * weight_value;
      } else if (input_index >= 0 && input_index < params.input_length) {
        accumulators[lane] += (float)input_row[input_index] * weight_value;
      }
    }
  }
  const int64_t output_row =
      ((int64_t)batch * params.input_channels * params.channel_multiplier +
       channel) *
      params.output_length;
#pragma unroll
  for (int lane = 0; lane < conv1d_dw_outputs_per_thread; ++lane) {
    const int output_index = output_start + lane;
    if IF_CONSTEXPR (!CHECK_BOUNDS) {
      output[output_row + output_index] = (T)accumulators[lane];
    } else if (output_index < params.output_length) {
      output[output_row + output_index] = (T)accumulators[lane];
    }
  }
}

// DHWIO copy of an OIDHW weight view; ATen's generic permute copy runs well
// below memory bandwidth for this permutation. Each thread streams one kW row.
template <typename T>
kernel void conv_weight_to_dhwio(
    device const T* source [[buffer(0)]],
    device T* destination [[buffer(1)]],
    constant ConvWeightPermuteParams& params [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  const int output_channel = position.x;
  const int input_channel = position.y;
  const int kernel_plane_index = position.z;
  const int kernel_height_index = kernel_plane_index % params.kernel_height;
  const int kernel_depth_index = kernel_plane_index / params.kernel_height;
  device const T* source_row = source +
      output_channel * params.output_channel_stride +
      input_channel * params.input_channel_stride +
      kernel_depth_index * params.depth_stride +
      kernel_height_index * params.height_stride;
  device T* destination_row = destination +
      (kernel_plane_index * params.kernel_width *
           params.input_channels_per_group +
       input_channel) *
          params.output_channels +
      output_channel;
  for (int kernel_width_index = 0; kernel_width_index < params.kernel_width;
       ++kernel_width_index) {
    destination_row
        [kernel_width_index * params.input_channels_per_group *
         params.output_channels] =
            source_row[kernel_width_index * params.width_stride];
  }
}

#define INSTANTIATE_CONV_WEIGHT_TO_DHWIO(DT)                      \
  template [[host_name("conv_weight_to_dhwio_" #DT)]] kernel void \
  conv_weight_to_dhwio<DT>(                                       \
      device const DT*, device DT*, constant ConvWeightPermuteParams&, uint3);

INSTANTIATE_CONV_WEIGHT_TO_DHWIO(float)
INSTANTIATE_CONV_WEIGHT_TO_DHWIO(half)
INSTANTIATE_CONV_WEIGHT_TO_DHWIO(bfloat)

// (kW, O, I) copy of an OIkW weight view: per-tap slabs with input channels
// contiguous, the A-operand form of conv1d_matmul.
template <typename T>
kernel void conv_weight_to_koc(
    device const T* source [[buffer(0)]],
    device T* destination [[buffer(1)]],
    constant ConvWeightPermuteParams& params [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  const int input_channel = int(position.x);
  const int output_channel = int(position.y);
  device const T* source_row = source +
      (int64_t)output_channel * params.output_channel_stride +
      (int64_t)input_channel * params.input_channel_stride;
  device T* destination_row = destination +
      (int64_t)output_channel * params.input_channels_per_group + input_channel;
  const int64_t tap_stride =
      (int64_t)params.input_channels_per_group * params.output_channels;
  for (int kernel_width_index = 0; kernel_width_index < params.kernel_width;
       ++kernel_width_index) {
    destination_row[kernel_width_index * tap_stride] =
        source_row[(int64_t)kernel_width_index * params.width_stride];
  }
}

static_assert(
    conv1d_dw_outputs_per_thread == 8,
    "CONV1D_DW_OUTPUTS_PER_THREAD_STR must match conv1d_dw_outputs_per_thread");

#define INSTANTIATE_CONV1D_DW(DT)                                       \
  template [[host_name("conv1d_dw_" #DT)]] kernel void conv1d_dw<DT>(   \
      device const DT*,                                                 \
      device const DT*,                                                 \
      device DT*,                                                       \
      constant Conv1dDwParams&,                                         \
      device const DT*,                                                 \
      uint3);                                                           \
  template [[host_name("conv1d_dw_vec" CONV1D_DW_OUTPUTS_PER_THREAD_STR \
                       "_" #DT)]] kernel void                           \
  conv1d_dw_vec<DT, true>(                                              \
      device const DT*,                                                 \
      device const DT*,                                                 \
      device DT*,                                                       \
      constant Conv1dDwParams&,                                         \
      device const DT*,                                                 \
      uint3);                                                           \
  template [[host_name("conv1d_dw_vec" CONV1D_DW_OUTPUTS_PER_THREAD_STR \
                       "_valid_" #DT)]] kernel void                     \
  conv1d_dw_vec<DT, false>(                                             \
      device const DT*,                                                 \
      device const DT*,                                                 \
      device DT*,                                                       \
      constant Conv1dDwParams&,                                         \
      device const DT*,                                                 \
      uint3);                                                           \
  template [[host_name("conv_weight_to_koc_" #DT)]] kernel void         \
  conv_weight_to_koc<DT>(                                               \
      device const DT*, device DT*, constant ConvWeightPermuteParams&, uint3);

INSTANTIATE_CONV1D_DW(float)
INSTANTIATE_CONV1D_DW(half)
INSTANTIATE_CONV1D_DW(bfloat)

// conv1d_matmul on simdgroup_matrix for hosts without
// MetalPerformancePrimitives: identical dispatch contract (koc weights,
// region table, NCL/NLC operands), fp32 accumulation.
template <typename T, int BM, bool NLC, bool HAS_BIAS, bool GROUPED>
kernel void conv1d_matmul_simd(
    device const T* src [[buffer(0)]],
    device const T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv1dMatmulParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int WM = BM >= 64 ? 2 : 1;
  constexpr int WN = 2;
  constexpr int TGP = WM * WN * 32;
  constexpr int WT_M = BM / WM, WT_N = BN / WN;
  constexpr int TM = WT_M / 8, TN = WT_N / 8;
  constexpr int LDA = BK + 8, LDB = BN + 8;

  threadgroup T As[BM * LDA];
  threadgroup T Bs[BK * LDB];

  const int Cg = p.C_in / p.groups;
  const int Og = p.C_out / p.groups;
  int o_off, o_end, c0;
  if IF_CONSTEXPR (GROUPED) {
    const int o_tiles = Og / BM + (Og % BM != 0);
    const int g = int(tgid.x) / o_tiles;
    o_off = (int(tgid.x) % o_tiles) * BM + g * Og;
    o_end = g * Og + Og;
    c0 = g * Cg;
  } else {
    o_off = int(tgid.x) * BM;
    o_end = p.C_out;
    c0 = 0;
  }
  const int zi = int(tgid.y);
  int r = 0;
  for (int i = 1; i < p.region_count; ++i) {
    if (zi >= p.regions[i].tile0) {
      r = i;
    }
  }
  constant const Conv1dMatmulRegion& reg = p.regions[r];
  const int m_off = (zi - reg.tile0) * BN;
  const int nb = int(tgid.z);
  const int tid = int(sgid) * 32 + int(lane);

  device const T* srcn = src + (int64_t)nb * p.C_in * p.L;
  device T* dstn = dst + (int64_t)nb * p.C_out * p.outW_total +
      (NLC ? reg.out_col0 * p.C_out : reg.out_col0);

  simdgroup_matrix<float, 8, 8> Cfrag[TM][TN];
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      Cfrag[i][j] = simdgroup_matrix<float, 8, 8>(0);
    }
  }
  const int warp_m = (int(sgid) / WN) * WT_M;
  const int warp_n = (int(sgid) % WN) * WT_N;

  for (int j = 0; j < reg.taps; ++j) {
    const int64_t col = reg.in_col0 + (int64_t)j * p.dilation;
    device const T* wj = wts + (int64_t)(reg.w_tap0 + j) * Cg * p.C_out;
    device const T* xj =
        NLC ? srcn + col * p.C_in + c0 : srcn + (int64_t)c0 * p.L + col;
    for (int k0 = 0; k0 < Cg; k0 += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = tid; i < BM * BK; i += TGP) {
        const int m = i / BK, kk = i % BK;
        const int o = o_off + m, c = k0 + kk;
        As[m * LDA + kk] = o < o_end && c < Cg ? wj[(int64_t)o * Cg + c] : T(0);
      }
      for (int i = tid; i < BK * BN; i += TGP) {
        const int kk = i / BN, n = i % BN;
        const int c = k0 + kk, m = m_off + n;
        T v = T(0);
        if (c < Cg && m < reg.out_cols) {
          v = NLC ? xj[(int64_t)m * p.C_in + c] : xj[(int64_t)c * p.L + m];
        }
        Bs[kk * LDB + n] = v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_matrix<T, 8, 8> Afrag[TM];
        simdgroup_matrix<T, 8, 8> Bfrag[TN];
        for (int i = 0; i < TM; ++i) {
          simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA + kk], LDA);
        }
        for (int t = 0; t < TN; ++t) {
          simdgroup_load(Bfrag[t], &Bs[kk * LDB + warp_n + t * 8], LDB);
        }
        for (int i = 0; i < TM; ++i) {
          for (int t = 0; t < TN; ++t) {
            simdgroup_multiply_accumulate(
                Cfrag[i][t], Afrag[i], Bfrag[t], Cfrag[i][t]);
          }
        }
      }
    }
  }

  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;

  for (int i = 0; i < TM; ++i) {
    const int o = o_off + warp_m + i * 8 + fm;
    if (o >= o_end) {
      continue;
    }
    for (int t = 0; t < TN; ++t) {
      for (int e = 0; e < 2; ++e) {
        const int m = m_off + warp_n + t * 8 + fn + e;
        if (m >= reg.out_cols) {
          continue;
        }
        float v = Cfrag[i][t].thread_elements()[e];
        if IF_CONSTEXPR (HAS_BIAS) {
          v += (float)bias[o];
        }
        if IF_CONSTEXPR (NLC) {
          dstn[(int64_t)m * p.C_out + o] = (T)v;
        } else {
          dstn[(int64_t)o * p.outW_total + m] = (T)v;
        }
      }
    }
  }
}

#define INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(                           \
    DT, BM, LNAME, NLC, BNAME, HAS_BIAS, GNAME, GROUPED)              \
  template [[host_name("conv1d_matmul_simd_" #DT "_bm" #BM "_" #LNAME \
                       "_" #BNAME "_" #GNAME)]] kernel void           \
  conv1d_matmul_simd<DT, BM, NLC, HAS_BIAS, GROUPED>(                 \
      device const DT*,                                               \
      device const DT*,                                               \
      device DT*,                                                     \
      constant Conv1dMatmulParams&,                                   \
      device const DT*,                                               \
      uint3,                                                          \
      uint,                                                           \
      uint);

// clang-format off
#define INSTANTIATE_CONV1D_MATMUL_SIMD_VARIANTS(DT, BM)                                  \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, ncl, false, bias, true, ungrouped, false)   \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, ncl, false, nobias, false, ungrouped, false) \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, nlc, true, bias, true, ungrouped, false)     \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, nlc, true, nobias, false, ungrouped, false)  \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, ncl, false, bias, true, grouped, true)       \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, ncl, false, nobias, false, grouped, true)    \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, nlc, true, bias, true, grouped, true)        \
  INSTANTIATE_CONV1D_MATMUL_SIMD_ONE(DT, BM, nlc, true, nobias, false, grouped, true)

#define INSTANTIATE_CONV1D_MATMUL_SIMD(DT)         \
  INSTANTIATE_CONV1D_MATMUL_SIMD_VARIANTS(DT, 32)  \
  INSTANTIATE_CONV1D_MATMUL_SIMD_VARIANTS(DT, 64)
// clang-format on

INSTANTIATE_CONV1D_MATMUL_SIMD(float)
INSTANTIATE_CONV1D_MATMUL_SIMD(half)
INSTANTIATE_CONV1D_MATMUL_SIMD(bfloat)

#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_cooperative_tensor>
#include <metal_simdgroup>

using namespace mpp::tensor_ops;

template <int X, int Y, int Z>
struct StaticInt3 {
  static constexpr constant int x = X;
  static constexpr constant int y = Y;
  static constexpr constant int z = Z;
};

// One 2D convolution per (batch, output-depth) slice; the KD loop accumulates
// kernel-depth taps into one cooperative tensor, skipping depth padding.
template <
    typename T,
    typename Block,
    int NSG,
    typename Kernel,
    typename Stride,
    typename Dilation,
    typename Source,
    bool RELAXED,
    bool HAS_BIAS,
    bool OUT_NCDHW,
    bool GROUPED>
kernel void conv3d_mpp(
    device T* act [[buffer(0)]],
    device T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv3DParams& gP [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  constant Conv2DParams& p = gP.conv2d;
  constexpr int BO = Block::x, BW = Block::y, BH = Block::z;
  constexpr int KW = Kernel::x, KH = Kernel::y, KD = Kernel::z;
  constexpr int SX = Stride::x, SY = Stride::y, SZ = Stride::z;
  constexpr int DX = Dilation::x, DY = Dilation::y, DZ = Dilation::z;
  constexpr int SRCC = Source::x, SRCW = Source::y, SRCH = Source::z;
  const int h_tiles = p.outH / BH + (p.outH % BH != 0);
  int o_off, o_end, c0;
  if constexpr (GROUPED) {
    const int o_tiles = p.C_out_per_group / BO + (p.C_out_per_group % BO != 0);
    const int g = int(tgid.x) / o_tiles;
    o_off = (int(tgid.x) % o_tiles) * BO + g * p.C_out_per_group;
    o_end = g * p.C_out_per_group + p.C_out_per_group;
    c0 = g * p.C_in_per_group;
  } else {
    o_off = int(tgid.x) * BO;
    o_end = p.C_out;
    c0 = 0;
  }
  const int wo_off = int(tgid.y) * BW;
  int zi = int(tgid.z);
  const int ho_off = (zi % h_tiles) * BH;
  zi /= h_tiles;
  const int dd = zi % gP.outD;
  const int nb = zi / gP.outD;

  // DHWIO weights viewed innermost-first; kernel-depth slice kd starts at
  // row kd * KH of the flattened (KH * KD) height axis.
  tensor<device T, dextents<int32_t, 4>, tensor_inline> tWt(
      wts, dextents<int32_t, 4>(p.C_out, p.C_in_per_group, KW, KH * KD));

  constexpr auto desc = convolution2d_descriptor(
      int4(BO, BW, BH, 1),
      int4(SRCC, SRCW, SRCH, 1),
      int2(KW, KH),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(SX, SY),
      int2(DX, DY),
      1,
      RELAXED,
      convolution2d_descriptor::mode::multiply_accumulate);
  convolution2d<desc, execution_simdgroups<NSG>> op;
  // MPP anchors the window at ceil((K - 1) * dilation / 2) (measured; for
  // even kernels with dilation > 1 this is NOT (K / 2) * dilation).
  op.set_offsets(int2(
      wo_off * SX - p.padW + ((KW - 1) * DX + 1) / 2,
      ho_off * SY - p.padH + ((KH - 1) * DY + 1) / 2));

  const int64_t plane = (int64_t)p.H * p.W * p.C_in;
  device T* actn = act + (int64_t)nb * gP.D * plane;
  auto act_view = [&](device T* ptr) {
    if constexpr (GROUPED) {
      return tensor<device T, dextents<int32_t, 4>, tensor_inline>(
          ptr,
          dextents<int32_t, 4>(c0 + p.C_in_per_group, p.W, p.H, 1),
          array<int32_t, 4>{1, p.C_in, p.C_in * p.W, p.C_in * p.W * p.H});
    } else {
      return tensor<device T, dextents<int32_t, 4>, tensor_inline>(
          ptr, dextents<int32_t, 4>(p.C_in, p.W, p.H, 1));
    }
  };
  auto tA0 = act_view(actn);
  auto mA0 = tA0.slice(c0, 0, 0, 0);
  auto mW0 = tWt.slice(o_off, 0, 0, 0);

  auto cT = op.template get_destination_cooperative_tensor<
      decltype(mA0),
      decltype(mW0),
      float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      cT[i] = 0.0f;
    }
  }

  for (int kd = 0; kd < KD; ++kd) {
    const int di = dd * SZ - gP.padD + kd * DZ;
    if (di < 0 || di >= gP.D) {
      continue;
    }
    auto tA = act_view(actn + di * plane);
    auto mA = tA.slice(c0, 0, 0, 0);
    auto mW = tWt.slice(o_off, 0, 0, kd * KH);
    op.run(mA, mW, cT);
  }

  if constexpr (GROUPED || OUT_NCDHW) {
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      if (!cT.is_valid_element(i)) {
        continue;
      }
      auto idx = cT.get_multidimensional_index(i);
      const int o = o_off + (int)idx[0];
      const int x = wo_off + (int)idx[1];
      const int y = ho_off + (int)idx[2];
      if (o >= o_end || x >= p.outW || y >= p.outH) {
        continue;
      }
      float v = cT[i];
      if constexpr (HAS_BIAS) {
        v += (float)bias[o];
      }
      int64_t di;
      if constexpr (OUT_NCDHW) {
        di = ((((int64_t)nb * p.C_out + o) * gP.outD + dd) * p.outH + y) *
                p.outW +
            x;
      } else {
        di = ((((int64_t)nb * gP.outD + dd) * p.outH + y) * p.outW + x) *
                p.C_out +
            o;
      }
      dst[di] = (T)v;
    }
  } else {
    device T* dstn =
        dst + ((int64_t)nb * gP.outD + dd) * p.outH * p.outW * p.C_out;
    tensor<device T, dextents<int32_t, 4>, tensor_inline> tD(
        dstn, dextents<int32_t, 4>(p.C_out, p.outW, p.outH, 1));
    auto mD = tD.slice(o_off, wo_off, ho_off, 0);
    auto cO = op.template get_destination_cooperative_tensor<
        decltype(mA0),
        decltype(mW0),
        T>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
      if (!cT.is_valid_element(i)) {
        continue;
      }
      float v = cT[i];
      if constexpr (HAS_BIAS) {
        // clamp: lanes past O are masked by the bounds-aware store below
        const int o = o_off + (int)cT.get_multidimensional_index(i)[0];
        v += (float)bias[min(o, p.C_out - 1)];
      }
      cO[i] = (T)v;
    }
    cO.store(mD);
  }
}

// Source-width shape hint baked into each specialization through the
// convolution2d_descriptor. The compiled kernel assumes the activation never
// exceeds it: a wider source miscomputes silently and -1 (dynamic, as used
// for channels) miscomputes too, so the host rejects any activation longer
// than the hint its entry point was built with. Oversizing measured free, so
// the conv1d block redefines this to the wide header constant; conv3d keeps
// 16384, which covers its plane extents.
#define CONV3D_MPP_SRCW 16384

#define INSTANTIATE_CONV3D_MPP(                                             \
    DT,                                                                     \
    DNAME,                                                                  \
    RELAXED,                                                                \
    BO,                                                                     \
    BW,                                                                     \
    BH,                                                                     \
    NSG,                                                                    \
    KD,                                                                     \
    KH,                                                                     \
    KW,                                                                     \
    SZ,                                                                     \
    SY,                                                                     \
    SX,                                                                     \
    DZ,                                                                     \
    DY,                                                                     \
    DX,                                                                     \
    CNAME,                                                                  \
    SRCC,                                                                   \
    BNAME,                                                                  \
    HAS_BIAS,                                                               \
    LNAME,                                                                  \
    OUT_NCDHW,                                                              \
    GNAME,                                                                  \
    GROUPED)                                                                \
  template                                                                  \
      [[host_name("conv3d_mpp_" #DNAME "_b" #BO "_w" #BW "_h" #BH "_s" #NSG \
                  "_k" #KD #KH #KW "_s" #SZ #SY #SX "_d" #DZ #DY #DX        \
                  "_c" #CNAME "_" #BNAME "_" #LNAME "_" #GNAME)]]           \
      kernel void conv3d_mpp<                                               \
          DT,                                                               \
          StaticInt3<BO, BW, BH>,                                           \
          NSG,                                                              \
          StaticInt3<KW, KH, KD>,                                           \
          StaticInt3<SX, SY, SZ>,                                           \
          StaticInt3<DX, DY, DZ>,                                           \
          StaticInt3<SRCC, CONV3D_MPP_SRCW, 16384>,                         \
          RELAXED,                                                          \
          HAS_BIAS,                                                         \
          OUT_NCDHW,                                                        \
          GROUPED>(                                                         \
          device DT*,                                                       \
          device DT*,                                                       \
          device DT*,                                                       \
          constant Conv3DParams&,                                           \
          device const DT*,                                                 \
          uint3);

#define INSTANTIATE_CONV3D_MPP_DTYPES( \
    BO,                                \
    BW,                                \
    BH,                                \
    NSG,                               \
    KD,                                \
    KH,                                \
    KW,                                \
    SZ,                                \
    SY,                                \
    SX,                                \
    DZ,                                \
    DY,                                \
    DX,                                \
    CNAME,                             \
    SRCC,                              \
    BNAME,                             \
    HAS_BIAS,                          \
    LNAME,                             \
    OUT_NCDHW,                         \
    GNAME,                             \
    GROUPED)                           \
  INSTANTIATE_CONV3D_MPP(              \
      float,                           \
      float,                           \
      false,                           \
      BO,                              \
      BW,                              \
      BH,                              \
      NSG,                             \
      KD,                              \
      KH,                              \
      KW,                              \
      SZ,                              \
      SY,                              \
      SX,                              \
      DZ,                              \
      DY,                              \
      DX,                              \
      CNAME,                           \
      SRCC,                            \
      BNAME,                           \
      HAS_BIAS,                        \
      LNAME,                           \
      OUT_NCDHW,                       \
      GNAME,                           \
      GROUPED)                         \
  INSTANTIATE_CONV3D_MPP(              \
      half,                            \
      half,                            \
      true,                            \
      BO,                              \
      BW,                              \
      BH,                              \
      NSG,                             \
      KD,                              \
      KH,                              \
      KW,                              \
      SZ,                              \
      SY,                              \
      SX,                              \
      DZ,                              \
      DY,                              \
      DX,                              \
      CNAME,                           \
      SRCC,                            \
      BNAME,                           \
      HAS_BIAS,                        \
      LNAME,                           \
      OUT_NCDHW,                       \
      GNAME,                           \
      GROUPED)                         \
  INSTANTIATE_CONV3D_MPP(              \
      bfloat,                          \
      bfloat,                          \
      true,                            \
      BO,                              \
      BW,                              \
      BH,                              \
      NSG,                             \
      KD,                              \
      KH,                              \
      KW,                              \
      SZ,                              \
      SY,                              \
      SX,                              \
      DZ,                              \
      DY,                              \
      DX,                              \
      CNAME,                           \
      SRCC,                            \
      BNAME,                           \
      HAS_BIAS,                        \
      LNAME,                           \
      OUT_NCDHW,                       \
      GNAME,                           \
      GROUPED)

#define INSTANTIATE_CONV3D_MPP_TILES( \
    KD,                               \
    KH,                               \
    KW,                               \
    SZ,                               \
    SY,                               \
    SX,                               \
    DZ,                               \
    DY,                               \
    DX,                               \
    CNAME,                            \
    SRCC,                             \
    BNAME,                            \
    HAS_BIAS,                         \
    LNAME,                            \
    OUT_NCDHW,                        \
    GNAME,                            \
    GROUPED)                          \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      32,                             \
      8,                              \
      8,                              \
      2,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      32,                             \
      16,                             \
      8,                              \
      4,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      64,                             \
      8,                              \
      8,                              \
      2,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      64,                             \
      16,                             \
      8,                              \
      4,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)                        \
  INSTANTIATE_CONV3D_MPP_DTYPES(      \
      128,                            \
      8,                              \
      8,                              \
      4,                              \
      KD,                             \
      KH,                             \
      KW,                             \
      SZ,                             \
      SY,                             \
      SX,                             \
      DZ,                             \
      DY,                             \
      DX,                             \
      CNAME,                          \
      SRCC,                           \
      BNAME,                          \
      HAS_BIAS,                       \
      LNAME,                          \
      OUT_NCDHW,                      \
      GNAME,                          \
      GROUPED)

#define INSTANTIATE_CONV3D_MPP_STANDARD_VARIANT(          \
    KD, KH, KW, SZ, SY, SX, CNAME, SRCC, BNAME, HAS_BIAS) \
  INSTANTIATE_CONV3D_MPP_TILES(                           \
      KD,                                                 \
      KH,                                                 \
      KW,                                                 \
      SZ,                                                 \
      SY,                                                 \
      SX,                                                 \
      1,                                                  \
      1,                                                  \
      1,                                                  \
      CNAME,                                              \
      SRCC,                                               \
      BNAME,                                              \
      HAS_BIAS,                                           \
      ncdhw,                                              \
      true,                                               \
      ungrouped,                                          \
      false)

#define INSTANTIATE_CONV3D_MPP_STANDARD(KD, KH, KW, SZ, SY, SX, CNAME, SRCC) \
  INSTANTIATE_CONV3D_MPP_STANDARD_VARIANT(                                   \
      KD, KH, KW, SZ, SY, SX, CNAME, SRCC, bias, true)                       \
  INSTANTIATE_CONV3D_MPP_STANDARD_VARIANT(                                   \
      KD, KH, KW, SZ, SY, SX, CNAME, SRCC, nobias, false)

#define INSTANTIATE_CONV3D_MPP_GROUPED(KD, KH, KW, SZ, SY, SX) \
  INSTANTIATE_CONV3D_MPP_TILES(                                \
      KD,                                                      \
      KH,                                                      \
      KW,                                                      \
      SZ,                                                      \
      SY,                                                      \
      SX,                                                      \
      1,                                                       \
      1,                                                       \
      1,                                                       \
      1,                                                       \
      1,                                                       \
      nobias,                                                  \
      false,                                                   \
      ncdhw,                                                   \
      true,                                                    \
      grouped,                                                 \
      true)

// conv1d rides conv3d with a unit depth axis and outH == 1; destination
// tiles are flat along W. Relaxed-mode MPP miscomputes BH == 1 tiles with
// BW / NSG < 16 (measured), so every candidate keeps BW >= 16 * NSG.
// clang-format off
#define INSTANTIATE_CONV1D_MPP_TILES(KW, SX, DX, CNAME, SRCC, BNAME, HAS_BIAS, LNAME, OUT_NCDHW) \
  INSTANTIATE_CONV3D_MPP_DTYPES(32, 32, 1, 2, 1, 1, KW, 1, 1, SX, 1, 1, DX,   \
      CNAME, SRCC, BNAME, HAS_BIAS, LNAME, OUT_NCDHW, ungrouped, false)        \
  INSTANTIATE_CONV3D_MPP_DTYPES(64, 64, 1, 2, 1, 1, KW, 1, 1, SX, 1, 1, DX,   \
      CNAME, SRCC, BNAME, HAS_BIAS, LNAME, OUT_NCDHW, ungrouped, false)        \
  INSTANTIATE_CONV3D_MPP_DTYPES(64, 128, 1, 4, 1, 1, KW, 1, 1, SX, 1, 1, DX,  \
      CNAME, SRCC, BNAME, HAS_BIAS, LNAME, OUT_NCDHW, ungrouped, false)        \
  INSTANTIATE_CONV3D_MPP_DTYPES(128, 64, 1, 4, 1, 1, KW, 1, 1, SX, 1, 1, DX,  \
      CNAME, SRCC, BNAME, HAS_BIAS, LNAME, OUT_NCDHW, ungrouped, false)

#define INSTANTIATE_CONV1D_MPP(KW, SX, DX, CNAME, SRCC)                          \
  INSTANTIATE_CONV1D_MPP_TILES(KW, SX, DX, CNAME, SRCC, bias, true, ncdhw, true) \
  INSTANTIATE_CONV1D_MPP_TILES(KW, SX, DX, CNAME, SRCC, nobias, false, ncdhw, true)

// Strided conv1d cannot ride the unit-stride conv1d_matmul path, so channels-last
// outputs (inherited from channels-last inputs) need ndhwc twins to stay on the
// direct kernel; s == 1 geometries never reach it with a channels-last output.
#define INSTANTIATE_CONV1D_MPP_STRIDED(KW, SX, CNAME, SRCC)                        \
  INSTANTIATE_CONV1D_MPP(KW, SX, 1, CNAME, SRCC)                                   \
  INSTANTIATE_CONV1D_MPP_TILES(KW, SX, 1, CNAME, SRCC, bias, true, ndhwc, false)   \
  INSTANTIATE_CONV1D_MPP_TILES(KW, SX, 1, CNAME, SRCC, nobias, false, ndhwc, false)
// clang-format on

#undef CONV3D_MPP_SRCW
#define CONV3D_MPP_SRCW conv1d_mpp_src_width_hint

INSTANTIATE_CONV1D_MPP_STRIDED(2, 2, dyn, -1)
INSTANTIATE_CONV1D_MPP(3, 1, 1, dyn, -1)
INSTANTIATE_CONV1D_MPP(3, 1, 2, dyn, -1)
INSTANTIATE_CONV1D_MPP(3, 1, 3, dyn, -1)
INSTANTIATE_CONV1D_MPP(3, 1, 4, dyn, -1)
INSTANTIATE_CONV1D_MPP(3, 1, 5, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(3, 2, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(3, 3, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(4, 2, dyn, -1)
INSTANTIATE_CONV1D_MPP(5, 1, 1, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(5, 2, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(6, 3, dyn, -1)
INSTANTIATE_CONV1D_MPP(7, 1, 1, dyn, -1)
INSTANTIATE_CONV1D_MPP(7, 1, 3, dyn, -1)
INSTANTIATE_CONV1D_MPP(7, 1, 5, dyn, -1)
INSTANTIATE_CONV1D_MPP(7, 1, 9, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(8, 4, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(10, 4, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(10, 5, dyn, -1)
INSTANTIATE_CONV1D_MPP(10, 5, 1, 1, 1)
INSTANTIATE_CONV1D_MPP(11, 1, 1, dyn, -1)
INSTANTIATE_CONV1D_MPP(11, 1, 3, dyn, -1)
INSTANTIATE_CONV1D_MPP(11, 1, 5, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(12, 6, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(16, 8, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(3, 5, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(5, 3, dyn, -1)
INSTANTIATE_CONV1D_MPP_STRIDED(7, 2, dyn, -1)

#undef CONV3D_MPP_SRCW
#define CONV3D_MPP_SRCW 16384

// Deduplicated direct Conv3d specs from the surveyed model shapes.
INSTANTIATE_CONV3D_MPP_GROUPED(1, 1, 2, 1, 1, 2)
INSTANTIATE_CONV3D_MPP_GROUPED(1, 2, 1, 1, 2, 1)
INSTANTIATE_CONV3D_MPP_GROUPED(2, 1, 1, 2, 1, 1)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 2, 2, 1, 2, 2, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 2, 2, 1, 2, 2, 36, 36)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 3, 3, 1, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 3, 3, 1, 1, 1, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(1, 3, 3, 1, 2, 2, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 1, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 1, 1, 1, 3, 3)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 1, 1, 1, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 1, 1, 2, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 3, 3)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 16, 16)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 48, 48)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 1, 1, 64, 64)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 1, 2, 2, dyn, -1)
INSTANTIATE_CONV3D_MPP_STANDARD(3, 3, 3, 2, 2, 2, dyn, -1)

// Unit-stride conv1d as `taps` accumulated shifted matmuls: for tap j,
// dst[o, m] += W_j[o, c] * src[c, in_col0 + m + j * dilation]. The activation
// is read in its native layout (NCL, or NLC via the transposed-B form), so no
// NHWC staging pass exists on this path, and kernel size and dilation stay
// runtime. Weights come in as the DHWIO copy: tap-j slab is (C_per_group,
// C_out) with output channels contiguous. The host splits padded edges into
// extra dispatches over clipped tap ranges so reads never leave [0, L).
template <
    typename T,
    int BM,
    int BN,
    int NSG,
    bool NLC,
    bool HAS_BIAS,
    bool GROUPED>
kernel void conv1d_matmul(
    device T* src [[buffer(0)]],
    device T* wts [[buffer(1)]],
    device T* dst [[buffer(2)]],
    constant Conv1dMatmulParams& p [[buffer(3)]],
    device const T* bias [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const int Cg = p.C_in / p.groups;
  const int Og = p.C_out / p.groups;
  int o_off, o_end, c0;
  if constexpr (GROUPED) {
    const int o_tiles = Og / BM + (Og % BM != 0);
    const int g = int(tgid.x) / o_tiles;
    o_off = (int(tgid.x) % o_tiles) * BM + g * Og;
    o_end = g * Og + Og;
    c0 = g * Cg;
  } else {
    o_off = int(tgid.x) * BM;
    o_end = p.C_out;
    c0 = 0;
  }
  const int zi = int(tgid.y);
  int r = 0;
  for (int i = 1; i < p.region_count; ++i) {
    if (zi >= p.regions[i].tile0) {
      r = i;
    }
  }
  constant const Conv1dMatmulRegion& reg = p.regions[r];
  const int m_off = (zi - reg.tile0) * BN;
  const int nb = int(tgid.z);

  constexpr auto desc = matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      NLC,
      !is_same_v<T, float>,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  device T* srcn = src + (int64_t)nb * p.C_in * p.L;
  device T* dstn = dst + (int64_t)nb * p.C_out * p.outW_total +
      (NLC ? reg.out_col0 * p.C_out : reg.out_col0);

  auto cT = op.template get_destination_cooperative_tensor<
      tensor<device T, dextents<int32_t, 2>, tensor_inline>,
      tensor<device T, dextents<int32_t, 2>, tensor_inline>,
      float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      cT[i] = 0.0f;
    }
  }

  for (int j = 0; j < reg.taps; ++j) {
    const int64_t col = reg.in_col0 + (int64_t)j * p.dilation;
    device T* wj = wts + (int64_t)(reg.w_tap0 + j) * Cg * p.C_out;
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tA(
        wj, dextents<int32_t, 2>(Cg, p.C_out), array<int32_t, 2>{1, Cg});
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tB(
        NLC ? srcn + col * p.C_in + c0 : srcn + (int64_t)c0 * p.L + col,
        NLC ? dextents<int32_t, 2>(Cg, reg.out_cols)
            : dextents<int32_t, 2>(reg.out_cols, Cg),
        NLC ? array<int32_t, 2>{1, p.C_in} : array<int32_t, 2>{1, p.L});
    auto mA = tA.slice(0, o_off);
    auto mB = NLC ? tB.slice(0, m_off) : tB.slice(m_off, 0);
    op.run(mA, mB, cT);
  }

  if constexpr (!NLC) {
    if (m_off + BN <= reg.out_cols && o_off + BM <= o_end) {
      tensor<device T, dextents<int32_t, 2>, tensor_inline> tD(
          dstn,
          dextents<int32_t, 2>(reg.out_cols, p.C_out),
          array<int32_t, 2>{1, p.outW_total});
      auto mD = tD.slice(m_off, o_off);
      auto cO = op.template get_destination_cooperative_tensor<
          tensor<device T, dextents<int32_t, 2>, tensor_inline>,
          tensor<device T, dextents<int32_t, 2>, tensor_inline>,
          T>();
      for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) {
          continue;
        }
        float v = cT[i];
        if constexpr (HAS_BIAS) {
          // clamp: lanes past O are masked by the bounds-aware store below
          const int o = o_off + (int)cT.get_multidimensional_index(i)[1];
          v += (float)bias[min(o, p.C_out - 1)];
        }
        cO[i] = (T)v;
      }
      cO.store(mD);
      return;
    }
  }
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (!cT.is_valid_element(i)) {
      continue;
    }
    auto idx = cT.get_multidimensional_index(i);
    const int m = m_off + (int)idx[0];
    const int o = o_off + (int)idx[1];
    if (m >= reg.out_cols || o >= o_end) {
      continue;
    }
    float v = cT[i];
    if constexpr (HAS_BIAS) {
      v += (float)bias[o];
    }
    if constexpr (NLC) {
      dstn[(int64_t)m * p.C_out + o] = (T)v;
    } else {
      dstn[(int64_t)o * p.outW_total + m] = (T)v;
    }
  }
}

#define INSTANTIATE_CONV1D_MATMUL_ONE(                                    \
    DT, BM, BN, NSG, LNAME, NLC, BNAME, HAS_BIAS, GNAME, GROUPED)         \
  template [[host_name("conv1d_matmul_" #DT "_bm" #BM "_bn" #BN "_s" #NSG \
                       "_" #LNAME "_" #BNAME "_" #GNAME)]] kernel void    \
  conv1d_matmul<DT, BM, BN, NSG, NLC, HAS_BIAS, GROUPED>(                 \
      device DT*,                                                         \
      device DT*,                                                         \
      device DT*,                                                         \
      constant Conv1dMatmulParams&,                                       \
      device const DT*,                                                   \
      uint3);

// clang-format off
#define INSTANTIATE_CONV1D_MATMUL_VARIANTS(DT, BM, BN, NSG)                            \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, ncl, false, bias, true, ungrouped, false)   \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, ncl, false, nobias, false, ungrouped, false) \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, nlc, true, bias, true, ungrouped, false)     \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, nlc, true, nobias, false, ungrouped, false)  \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, ncl, false, bias, true, grouped, true)       \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, ncl, false, nobias, false, grouped, true)    \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, nlc, true, bias, true, grouped, true)        \
  INSTANTIATE_CONV1D_MATMUL_ONE(DT, BM, BN, NSG, nlc, true, nobias, false, grouped, true)

#define INSTANTIATE_CONV1D_MATMUL(DT)              \
  INSTANTIATE_CONV1D_MATMUL_VARIANTS(DT, 32, 64, 2) \
  INSTANTIATE_CONV1D_MATMUL_VARIANTS(DT, 64, 64, 2) \
  INSTANTIATE_CONV1D_MATMUL_VARIANTS(DT, 128, 64, 4)
// clang-format on

INSTANTIATE_CONV1D_MATMUL(float)
INSTANTIATE_CONV1D_MATMUL(half)
INSTANTIATE_CONV1D_MATMUL(bfloat)

#endif // __METAL_VERSION__ >= 400 && MetalPerformancePrimitives

// Direct NCHW conv fallback for filter dims >= 256 (MPSGraph miscomputes); each
// thread does R consecutive ow outputs. I is int when all numels fit in int32.
template <typename T, typename I, int R>
kernel void conv2d(
    constant T* input [[buffer(0)]],
    constant T* weight [[buffer(1)]],
    constant T* bias [[buffer(2)]],
    device T* output [[buffer(3)]],
    constant Conv2DParams& p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  using acc_t = ::c10::metal::accum_t<T>;

  const uint owBlocks = (uint(p.outW) + R - 1) / R;
  uint rem = tid;
  const int ow0 = int(rem % owBlocks) * R;
  rem /= owBlocks;
  const int oh = int(rem % uint(p.outH));
  rem /= uint(p.outH);
  const int co = int(rem % uint(p.C_out));
  const int n = int(rem / uint(p.C_out));

  const int cin_base = (co / p.C_out_per_group) * p.C_in_per_group;
  const int ih0 = oh * p.sH - p.padH;
  const int iw_base = ow0 * p.sW - p.padW;
  // Full unit-stride block: loads need no per-lane bounds checks and merge.
  const bool full = (ow0 + R <= p.outW) && (p.sW == 1);

  const acc_t b = p.has_bias ? static_cast<acc_t>(bias[co]) : acc_t(0);
  acc_t acc[R];
  for (int r = 0; r < R; ++r) {
    acc[r] = b;
  }

  const I in_n_off = static_cast<I>(n) * p.C_in * p.H * p.W;
  const I w_co_off = static_cast<I>(co) * p.C_in_per_group * p.kH * p.kW;
  for (int ci = 0; ci < p.C_in_per_group; ++ci) {
    const I in_c_off = in_n_off + static_cast<I>(cin_base + ci) * p.H * p.W;
    const I w_c_off = w_co_off + static_cast<I>(ci) * p.kH * p.kW;
    for (int kh = 0; kh < p.kH; ++kh) {
      const int ih = ih0 + kh * p.dH;
      if (ih < 0 || ih >= p.H) {
        continue;
      }
      const I in_row = in_c_off + static_cast<I>(ih) * p.W;
      const I w_row = w_c_off + static_cast<I>(kh) * p.kW;
      for (int kw = 0; kw < p.kW; ++kw) {
        const int iw = iw_base + kw * p.dW;
        const acc_t wv = static_cast<acc_t>(weight[w_row + kw]);
        if (full && iw >= 0 && iw + R <= p.W) {
          for (int r = 0; r < R; ++r) {
            acc[r] += static_cast<acc_t>(input[in_row + iw + r]) * wv;
          }
        } else {
          for (int r = 0; r < R; ++r) {
            const int iwr = iw + r * p.sW;
            if (ow0 + r < p.outW && iwr >= 0 && iwr < p.W) {
              acc[r] += static_cast<acc_t>(input[in_row + iwr]) * wv;
            }
          }
        }
      }
    }
  }
  const I out_row = ((static_cast<I>(n) * p.C_out + co) * p.outH + oh) * p.outW;
  for (int r = 0; r < R; ++r) {
    if (ow0 + r < p.outW) {
      output[out_row + ow0 + r] = static_cast<T>(acc[r]);
    }
  }
}

#define REGISTER_CONV2D_OP(DTYPE, ITYPE, INAME, R)                       \
  template [[host_name("conv2d_r" #R "_" INAME "_" #DTYPE)]] kernel void \
  conv2d<DTYPE, ITYPE, R>(                                               \
      constant DTYPE * input [[buffer(0)]],                              \
      constant DTYPE * weight [[buffer(1)]],                             \
      constant DTYPE * bias [[buffer(2)]],                               \
      device DTYPE * output [[buffer(3)]],                               \
      constant Conv2DParams & p [[buffer(4)]],                           \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_CONV2D_DTYPE(DTYPE)         \
  REGISTER_CONV2D_OP(DTYPE, int, "i32", 1);  \
  REGISTER_CONV2D_OP(DTYPE, int, "i32", 4);  \
  REGISTER_CONV2D_OP(DTYPE, long, "i64", 1); \
  REGISTER_CONV2D_OP(DTYPE, long, "i64", 4);

REGISTER_CONV2D_DTYPE(float);
REGISTER_CONV2D_DTYPE(half);
REGISTER_CONV2D_DTYPE(bfloat);
