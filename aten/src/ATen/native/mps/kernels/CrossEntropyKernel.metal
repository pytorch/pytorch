#include <ATen/native/mps/kernels/CrossEntropyKernel.h>
#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
using c10::metal::simdgroup_size;

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

// Forward: compute cross-entropy loss per sample.
// One threadgroup per batch row. Loops over vocab with online log-sum-exp.
// Outputs: loss[B] (unreduced), lse[B] (for backward recomputation).
template <typename T>
kernel void cross_entropy_forward(
    device const T* logits [[buffer(0)]],
    device const int64_t* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device float* lse [[buffer(3)]],
    constant CrossEntropyParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4;
  uint V = params.vocab_size;
  int32_t ignore_idx = params.ignore_index;
  float smoothing = params.label_smoothing;

  int64_t tgt = target[tg_id];
  device const T* row = logits + uint64_t(tg_id) * V;

  // Check ignore_index
  if (tgt == int64_t(ignore_idx)) {
    if (tid == 0) {
      loss[tg_id] = 0.0f;
      lse[tg_id] = 0.0f;
    }
    return;
  }

  // Pass 1: online log-sum-exp + find target logit + sum of all logits (for
  // label smoothing)
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  float local_logit_sum = 0.0f;
  float target_logit = 0.0f;
  bool found_target = false;

  for (uint r = 0; r < V; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= V) {
      float4 v = ce_load_vec4(row + base);
      float chunk_max = fmax(fmax(v.x, v.y), fmax(v.z, v.w));
      float new_max = fmax(local_max, chunk_max);
      local_sum = local_sum * metal::precise::exp(local_max - new_max) +
          metal::precise::exp(v.x - new_max) +
          metal::precise::exp(v.y - new_max) +
          metal::precise::exp(v.z - new_max) +
          metal::precise::exp(v.w - new_max);
      local_max = new_max;
      local_logit_sum += v.x + v.y + v.z + v.w;

      // Check if target falls in this chunk
      for (int i = 0; i < N_READS; i++) {
        if (int64_t(base + i) == tgt) {
          target_logit = v[i];
          found_target = true;
        }
      }
    } else {
      for (uint i = base; i < min(base + uint(N_READS), V); i++) {
        float val = float(row[i]);
        float new_max = fmax(local_max, val);
        local_sum = local_sum * metal::precise::exp(local_max - new_max) +
            metal::precise::exp(val - new_max);
        local_max = new_max;
        local_logit_sum += val;
        if (int64_t(i) == tgt) {
          target_logit = val;
          found_target = true;
        }
      }
    }
  }

  // Reduce max across threadgroup.
  // Guard: when all threads in a simdgroup are idle, local_max and sg_max
  // are both -inf. exp(-inf - (-inf)) = exp(NaN) = NaN, which poisons the
  // sum. Use factor=0 when sg_max is -inf (no valid data in this simdgroup).
  float sg_max = simd_max(local_max);
  float rescale = (sg_max == -INFINITY) ? 0.0f
      : metal::precise::exp(local_max - sg_max);
  local_sum *= rescale;

  float sg_sum = simd_sum(local_sum);
  float sg_logit_sum = simd_sum(local_logit_sum);

  threadgroup float shared_max[simdgroup_size];
  threadgroup float shared_sum[simdgroup_size];
  threadgroup float shared_logit_sum[simdgroup_size];
  threadgroup float tg_result[4]; // max, sum, target_logit, logit_sum

  if (simd_lane_id == 0) {
    shared_max[simdgroup_id] = sg_max;
    shared_sum[simdgroup_id] = sg_sum;
    shared_logit_sum[simdgroup_id] = sg_logit_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_id == 0) {
    float m = (simd_lane_id < lsize / simdgroup_size)
        ? shared_max[simd_lane_id]
        : -INFINITY;
    float global_max = simd_max(m);
    float s = (simd_lane_id < lsize / simdgroup_size)
        ? shared_sum[simd_lane_id] * metal::precise::exp(m - global_max)
        : 0.0f;
    float global_sum = simd_sum(s);
    float ls = (simd_lane_id < lsize / simdgroup_size)
        ? shared_logit_sum[simd_lane_id]
        : 0.0f;
    float global_logit_sum = simd_sum(ls);
    if (simd_lane_id == 0) {
      tg_result[0] = global_max;
      tg_result[1] = global_sum;
      tg_result[3] = global_logit_sum;
    }
  }

  // Broadcast target_logit: exactly one thread found it
  threadgroup float shared_target[1];
  if (found_target) {
    shared_target[0] = target_logit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float row_max = tg_result[0];
    float total_sum = tg_result[1];
    float logit_sum = tg_result[3];
    float tgt_logit = shared_target[0];

    float log_sum_exp = row_max + metal::precise::log(total_sum);
    lse[tg_id] = log_sum_exp;

    // NLL component: -log_softmax(target)
    float nll = -(tgt_logit - log_sum_exp);

    if (smoothing > 0.0f) {
      // Label smoothing: (1-eps)*nll + eps*(lse - mean_logit)
      float smooth_loss = log_sum_exp - logit_sum / float(V);
      loss[tg_id] = (1.0f - smoothing) * nll + smoothing * smooth_loss;
    } else {
      loss[tg_id] = nll;
    }
  }
}

// Backward: compute gradient of cross-entropy w.r.t. logits.
// grad_input[b,c] = grad_out[b] * (softmax(logit[b,c]) - (c == target[b]))
// Recomputes softmax on-the-fly from saved lse.
template <typename T>
kernel void cross_entropy_backward(
    device const float* grad_output [[buffer(0)]],
    device const T* logits [[buffer(1)]],
    device const int64_t* target [[buffer(2)]],
    device const float* lse [[buffer(3)]],
    device T* grad_input [[buffer(4)]],
    constant CrossEntropyParams& params [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  constexpr int N_READS = 4;
  uint V = params.vocab_size;
  int32_t ignore_idx = params.ignore_index;
  float smoothing = params.label_smoothing;

  int64_t tgt = target[tg_id];
  float grad_out = grad_output[tg_id];
  device const T* row_in = logits + uint64_t(tg_id) * V;
  device T* row_out = grad_input + uint64_t(tg_id) * V;

  // If ignored, zero the gradient
  if (tgt == int64_t(ignore_idx)) {
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
  float inv_V = 1.0f / float(V);

  for (uint r = 0; r < V; r += lsize * N_READS) {
    uint base = r + tid * N_READS;
    if (base + N_READS <= V) {
      float4 v = ce_load_vec4(row_in + base);
      float4 sm = float4(
          metal::precise::exp(v.x - row_lse),
          metal::precise::exp(v.y - row_lse),
          metal::precise::exp(v.z - row_lse),
          metal::precise::exp(v.w - row_lse));

      float4 grad;
      if (smoothing > 0.0f) {
        // d/dx of (1-eps)*nll + eps*smooth_loss
        // = (1-eps)*(sm - one_hot) + eps*(sm - 1/V)
        // = sm - (1-eps)*one_hot - eps/V
        for (int i = 0; i < N_READS; i++) {
          float indicator = (int64_t(base + i) == tgt) ? 1.0f : 0.0f;
          grad[i] =
              grad_out * (sm[i] - (1.0f - smoothing) * indicator - smoothing * inv_V);
        }
      } else {
        for (int i = 0; i < N_READS; i++) {
          float indicator = (int64_t(base + i) == tgt) ? 1.0f : 0.0f;
          grad[i] = grad_out * (sm[i] - indicator);
        }
      }
      ce_store_vec4(row_out + base, grad);
    } else {
      for (uint i = base; i < min(base + uint(N_READS), V); i++) {
        float val = float(row_in[i]);
        float sm = metal::precise::exp(val - row_lse);
        float indicator = (int64_t(i) == tgt) ? 1.0f : 0.0f;
        float g;
        if (smoothing > 0.0f) {
          g = grad_out * (sm - (1.0f - smoothing) * indicator - smoothing * inv_V);
        } else {
          g = grad_out * (sm - indicator);
        }
        row_out[i] = static_cast<T>(g);
      }
    }
  }
}

#define INSTANTIATE_CE(DTYPE)                                                  \
  template [[host_name("cross_entropy_forward_" #DTYPE)]] kernel void         \
  cross_entropy_forward<DTYPE>(                                               \
      device const DTYPE* logits [[buffer(0)]],                               \
      device const int64_t* target [[buffer(1)]],                             \
      device float* loss [[buffer(2)]],                                       \
      device float* lse [[buffer(3)]],                                        \
      constant CrossEntropyParams& params [[buffer(4)]],                      \
      uint tg_id [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_position_in_threadgroup]],                            \
      uint lsize [[threads_per_threadgroup]],                                 \
      uint simd_lane_id [[thread_index_in_simdgroup]],                        \
      uint simdgroup_id [[simdgroup_index_in_threadgroup]]);                   \
                                                                              \
  template [[host_name("cross_entropy_backward_" #DTYPE)]] kernel void        \
  cross_entropy_backward<DTYPE>(                                              \
      device const float* grad_output [[buffer(0)]],                          \
      device const DTYPE* logits [[buffer(1)]],                               \
      device const int64_t* target [[buffer(2)]],                             \
      device const float* lse [[buffer(3)]],                                  \
      device DTYPE* grad_input [[buffer(4)]],                                 \
      constant CrossEntropyParams& params [[buffer(5)]],                      \
      uint tg_id [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_position_in_threadgroup]],                            \
      uint lsize [[threads_per_threadgroup]]);

INSTANTIATE_CE(float)
INSTANTIATE_CE(half)
INSTANTIATE_CE(bfloat)
