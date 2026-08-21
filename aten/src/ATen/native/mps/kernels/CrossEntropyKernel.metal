#include <ATen/native/mps/kernels/CrossEntropyKernel.h>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

// Fused 2D cross-entropy on native Metal. One threadgroup processes one row
// (one sample) of the [B, V] logits. The forward computes the cross-entropy
// loss per sample with an online log-sum-exp pass, never materializing the
// full [B, V] log_softmax intermediate. The backward recomputes softmax from
// the saved per-row log-sum-exp.
//
// Reference semantics (aten cross_entropy_loss):
//   nll      = -w[t] * (x[t] - lse)
//   smooth   = -sum_c w[c] * (x[c] - lse) = sum_w * lse - sum_c w[c] * x[c]
//   loss_row = (1 - eps) * nll + (eps / V) * smooth
// where w[c] == 1 when no class weight is supplied and sum_w = sum_c w[c].
// The cross-row reduction (mean/sum) and the mean normalizer are applied by
// the host in C++ so that autograd stays a plain differentiable composite.

static inline float4 ce_load_vec4(device const float* p) {
  return *reinterpret_cast<device const packed_float4*>(p);
}
static inline float4 ce_load_vec4(device const half* p) {
  return float4(*reinterpret_cast<device const packed_half4*>(p));
}
static inline float4 ce_load_vec4(device const bfloat* p) {
  return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}

static inline void ce_store_vec4(device float* p, float4 v) {
  *reinterpret_cast<device packed_float4*>(p) = v;
}
static inline void ce_store_vec4(device half* p, float4 v) {
  *reinterpret_cast<device packed_half4*>(p) = half4(v);
}
static inline void ce_store_vec4(device bfloat* p, float4 v) {
  p[0] = static_cast<bfloat>(v[0]);
  p[1] = static_cast<bfloat>(v[1]);
  p[2] = static_cast<bfloat>(v[2]);
  p[3] = static_cast<bfloat>(v[3]);
}

// Forward: cross-entropy loss per sample.
//   loss[B]  : per-row loss (weighted), fp32 for accumulation precision.
//   lse[B]   : per-row log-sum-exp, saved for backward.
//   wsum[B]  : per-row target weight (w[target], or 1; 0 when ignored). Used
//              by the host to build the mean normalizer.
template <typename T>
kernel void cross_entropy_forward(
    device const T* logits [[buffer(0)]],
    device const int64_t* target [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* sum_w_buf [[buffer(3)]],
    device float* loss [[buffer(4)]],
    device float* lse [[buffer(5)]],
    device float* wsum [[buffer(6)]],
    constant CrossEntropyParams& params [[buffer(7)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint V = params.vocab_size;
  int32_t ignore_idx = params.ignore_index;
  float smoothing = params.label_smoothing;
  bool has_weight = params.has_weight != 0;

  int64_t tgt = target[tg_id];
  device const T* row = logits + uint64_t(tg_id) * V;

  // ignore_index: this row contributes nothing.
  if (tgt == int64_t(ignore_idx)) {
    if (tid == 0) {
      loss[tg_id] = 0.0f;
      lse[tg_id] = 0.0f;
      wsum[tg_id] = 0.0f;
    }
    return;
  }

  // Target-out-of-range: report on the device error buffer (mirrors the
  // merged scatter/gather pattern) so the host raises on the next sync.
  // Never a CPU .item() round-trip.
  if (tgt < 0 || tgt >= int64_t(V)) {
    if (tid == 0) {
      TORCH_REPORT_ERROR(
          error_buf,
          "cross_entropy: target ",
          tgt,
          " is out of bounds for ",
          long(V),
          " classes");
      loss[tg_id] = 0.0f;
      lse[tg_id] = 0.0f;
      wsum[tg_id] = 0.0f;
    }
    return;
  }

  // Pass 1: online log-sum-exp; gather the target logit; accumulate the
  // weighted logit sum sum_c w[c]*x[c] for the label-smoothing term.
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  float local_wlogit_sum = 0.0f;
  bool found_nan = false;
  float target_logit = 0.0f;
  bool found_target = false;

  for (uint r = 0; r < V; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= V) {
      float4 v = ce_load_vec4(row + base);
      // fmax discards NaN, so track it explicitly; the flag poisons the sum
      // after the loop (CPU parity: one NaN logit -> NaN loss).
      found_nan = found_nan || metal::any(metal::isnan(v));
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      // All-(-inf) prefix chunks: exp(-inf - -inf) is NaN; the sum is 0 and
      // stays 0 until a finite value raises new_max.
      if (new_max > -INFINITY) {
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(v.x - new_max) +
            metal::precise::exp(v.y - new_max) +
            metal::precise::exp(v.z - new_max) +
            metal::precise::exp(v.w - new_max);
      }
      local_max = new_max;
      if (has_weight) {
        local_wlogit_sum += weight[base + 0] * v.x + weight[base + 1] * v.y +
            weight[base + 2] * v.z + weight[base + 3] * v.w;
      } else {
        local_wlogit_sum += v.x + v.y + v.z + v.w;
      }
      for (int i = 0; i < N_READS; i++) {
        if (int64_t(base + i) == tgt) {
          target_logit = v[i];
          found_target = true;
        }
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), V); i++) {
        float val = float(row[i]);
        found_nan = found_nan || metal::isnan(val);
        float new_max = fmax(local_max, val);
        if (new_max > -INFINITY) {
          local_sum = local_sum * metal::precise::exp(local_max - new_max) +
              metal::precise::exp(val - new_max);
        }
        local_max = new_max;
        local_wlogit_sum += has_weight ? weight[i] * val : val;
        if (int64_t(i) == tgt) {
          target_logit = val;
          found_target = true;
        }
      }
    }
  }

  // A NaN input must poison the row like CPU; applied once after the scan
  // (an in-loop poison could be skipped by the guarded update above). NaN
  // survives the 0-factor rescale below (0 * NaN == NaN) and simd_sum.
  if (found_nan) {
    local_sum = NAN;
  }
  // Reduce max across the threadgroup. Guard the all-idle simdgroup case:
  // exp(-inf - (-inf)) is NaN; use factor 0 when a simdgroup saw no data.
  float sg_max = simd_max(local_max);
  float rescale =
      (sg_max == -INFINITY) ? 0.0f : metal::precise::exp(local_max - sg_max);
  local_sum *= rescale;

  float sg_sum = simd_sum(local_sum);
  float sg_wlogit_sum = simd_sum(local_wlogit_sum);

  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];
  threadgroup float shared_wlogit_sum[simdgroup_size];
  threadgroup float tg_result[4]; // max, sum, _, wlogit_sum

  if (simd_lane_id == 0) {
    shared_max[simdgroup_id] = sg_max;
    shared_sum[simdgroup_id] = sg_sum;
    shared_wlogit_sum[simdgroup_id] = sg_wlogit_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_id == 0) {
    float m = (simd_lane_id < lsize / simdgroup_size) ? shared_max[simd_lane_id]
                                                      : -INFINITY;
    float global_max = simd_max(m);
    float s = (simd_lane_id < lsize / simdgroup_size)
        ? shared_sum[simd_lane_id] * metal::precise::exp(m - global_max)
        : 0.0f;
    float global_sum = simd_sum(s);
    float ls = (simd_lane_id < lsize / simdgroup_size)
        ? shared_wlogit_sum[simd_lane_id]
        : 0.0f;
    float global_wlogit_sum = simd_sum(ls);
    if (simd_lane_id == 0) {
      tg_result[0] = global_max;
      tg_result[1] = global_sum;
      tg_result[3] = global_wlogit_sum;
    }
  }

  // Broadcast the target logit: exactly one thread found it.
  threadgroup float shared_target[1];
  if (found_target) {
    shared_target[0] = target_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float row_max = tg_result[0];
    float total_sum = tg_result[1];
    float wlogit_sum = tg_result[3];
    float tgt_logit = shared_target[0];

    float log_sum_exp = row_max + metal::precise::log(total_sum);
    lse[tg_id] = log_sum_exp;

    float w_t = has_weight ? weight[tgt] : 1.0f;
    wsum[tg_id] = w_t;

    // NLL term: -w[t] * (x[t] - lse)
    float nll = -w_t * (tgt_logit - log_sum_exp);

    if (smoothing > 0.0f) {
      float sum_w = has_weight ? sum_w_buf[0] : float(V);
      // smooth_loss = sum_w * lse - sum_c w[c] * x[c]
      float smooth_loss = sum_w * log_sum_exp - wlogit_sum;
      loss[tg_id] =
          (1.0f - smoothing) * nll + (smoothing / float(V)) * smooth_loss;
    } else {
      loss[tg_id] = nll;
    }
  }
}

// Backward: gradient of the per-row cross-entropy loss w.r.t. logits.
//   grad_input[b, c] = grad_loss[b] *
//       ( (1 - eps) * w_t * ( sm[c] - one_hot(c == t) )
//         + (eps / V) * ( sum_w * sm[c] - w[c] ) )
// where sm = softmax(x) recomputed from the saved lse, w_t = w[t], and
// grad_loss[b] is the upstream gradient already scaled by the host (it folds
// in the mean normalizer / reduction factor). Rows with ignore_index get a
// zero gradient. The host passes grad_loss as fp32.
template <typename T>
kernel void cross_entropy_backward(
    device const float* grad_loss [[buffer(0)]],
    device const T* logits [[buffer(1)]],
    device const int64_t* target [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* sum_w_buf [[buffer(4)]],
    device const float* lse [[buffer(5)]],
    device T* grad_input [[buffer(6)]],
    constant CrossEntropyParams& params [[buffer(7)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;
  uint V = params.vocab_size;
  int32_t ignore_idx = params.ignore_index;
  float smoothing = params.label_smoothing;
  bool has_weight = params.has_weight != 0;

  int64_t tgt = target[tg_id];
  device const T* row_in = logits + uint64_t(tg_id) * V;
  device T* row_out = grad_input + uint64_t(tg_id) * V;

  // ignore_index (and the out-of-range rows the forward already flagged):
  // emit a zero gradient.
  if (tgt == int64_t(ignore_idx) || tgt < 0 || tgt >= int64_t(V)) {
    for (uint r = 0; r < V; r += lsize * N_READS) {
      uint base = r + tid * N_READS;
      if (base + N_READS <= V) {
        ce_store_vec4(row_out + base, float4(0.0f));
      } else {
        for (uint i = base; i < min(base + uint(N_READS), V); i++) {
          row_out[i] = static_cast<T>(0.0f);
        }
      }
    }
    return;
  }

  float row_lse = lse[tg_id];
  float go = grad_loss[tg_id];
  float w_t = has_weight ? weight[tgt] : 1.0f;
  float sum_w = has_weight ? sum_w_buf[0] : float(V);
  float eps_over_v = smoothing / float(V);

  for (uint r = 0; r < V; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= V) {
      float4 v = ce_load_vec4(row_in + base);
      float4 grad;
      for (int i = 0; i < N_READS; i++) {
        float sm = metal::precise::exp(v[i] - row_lse);
        float w_c = has_weight ? weight[base + i] : 1.0f;
        float indicator = (int64_t(base + i) == tgt) ? 1.0f : 0.0f;
        // d loss / d x[c]
        float g = (1.0f - smoothing) * w_t * (sm - indicator);
        if (smoothing > 0.0f) {
          g += eps_over_v * (sum_w * sm - w_c);
        }
        grad[i] = go * g;
      }
      ce_store_vec4(row_out + base, grad);
    } else {
      for (uint i = base; i < min(base + uint(N_READS), V); i++) {
        float val = float(row_in[i]);
        float sm = metal::precise::exp(val - row_lse);
        float w_c = has_weight ? weight[i] : 1.0f;
        float indicator = (int64_t(i) == tgt) ? 1.0f : 0.0f;
        float g = (1.0f - smoothing) * w_t * (sm - indicator);
        if (smoothing > 0.0f) {
          g += eps_over_v * (sum_w * sm - w_c);
        }
        row_out[i] = static_cast<T>(go * g);
      }
    }
  }
}

#define INSTANTIATE_CE(DTYPE)                                          \
  template [[host_name("cross_entropy_forward_" #DTYPE)]] kernel void  \
  cross_entropy_forward<DTYPE>(                                        \
      device const DTYPE* logits [[buffer(0)]],                        \
      device const int64_t* target [[buffer(1)]],                      \
      device const float* weight [[buffer(2)]],                        \
      device const float* sum_w_buf [[buffer(3)]],                     \
      device float* loss [[buffer(4)]],                                \
      device float* lse [[buffer(5)]],                                 \
      device float* wsum [[buffer(6)]],                                \
      constant CrossEntropyParams& params [[buffer(7)]],               \
      device c10::metal::ErrorMessages* error_buf [[buffer(8)]],       \
      uint tg_id [[threadgroup_position_in_grid]],                     \
      uint tid [[thread_position_in_threadgroup]],                     \
      uint lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                 \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);           \
                                                                       \
  template [[host_name("cross_entropy_backward_" #DTYPE)]] kernel void \
  cross_entropy_backward<DTYPE>(                                       \
      device const float* grad_loss [[buffer(0)]],                     \
      device const DTYPE* logits [[buffer(1)]],                        \
      device const int64_t* target [[buffer(2)]],                      \
      device const float* weight [[buffer(3)]],                        \
      device const float* sum_w_buf [[buffer(4)]],                     \
      device const float* lse [[buffer(5)]],                           \
      device DTYPE* grad_input [[buffer(6)]],                          \
      constant CrossEntropyParams& params [[buffer(7)]],               \
      uint tg_id [[threadgroup_position_in_grid]],                     \
      uint tid [[thread_position_in_threadgroup]],                     \
      uint lsize [[threads_per_threadgroup]]);

INSTANTIATE_CE(float)
INSTANTIATE_CE(half)
INSTANTIATE_CE(bfloat)
