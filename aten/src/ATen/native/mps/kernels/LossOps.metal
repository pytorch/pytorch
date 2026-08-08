#include <ATen/native/mps/kernels/LossOps.h>
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

// Augmented target lookup: l'[idx] is BLANK for even idx, l[idx/2] for odd.
template <typename T_target, typename T_index>
inline T_index get_target_prime(
    constant T_target* targets,
    T_index stride,
    T_index idx,
    T_index BLANK) {
  return (idx % 2 == 0) ? BLANK
                        : static_cast<T_index>(targets[stride * (idx / 2)]);
}

// Calculate `logsumexp(A, ...) = m + logsumexp(A - m, ...)`,
// where `m = max(A, ...)`.
template <typename T, typename... Ts>
static inline T logsumexp(T first, Ts... rest) {
  constexpr T neginf = -numeric_limits<T>::infinity();
  T args[] = {first, T(rest)...};
  constexpr int N = 1 + sizeof...(Ts);

  T m = args[0];
#pragma unroll
  for (int i = 1; i < N; i++) {
    m = max(m, args[i]);
  }
  if (m == neginf) {
    return neginf;
  }

  T s = 0;
#pragma unroll
  for (int i = 0; i < N; i++) {
    s += precise::exp(args[i] - m);
  }
  return precise::log(s) + m;
}

template <typename T, typename T_target, typename T_index, bool beta = false>
static void calc_log_alpha_beta(
    device T* log_alpha,
    constant T* log_probs,
    constant T_target* targets,
    constant T_index*,
    constant T_index*,
    constant T_index*,
    constant CTCLossParams<T_index>& params,
    uint tid,
    uint tptg,
    T_index input_length,
    T_index target_length) {
  using T_op = opmath_t<T>;
  constexpr T neginf = -numeric_limits<T>::infinity();
  constexpr T_op neginf_op = -numeric_limits<T_op>::infinity();

  T_index S_max = 2 * params.max_target_length + 1;
  T_index S = 2 * target_length + 1;

  // Initialize first time step for all the target tokens assigned to this
  // thread.
  for (T_index s = tid; s < S_max; s += tptg) {
    T la;
    auto log_alpha_time_offset =
        beta ? (params.log_alpha_time_stride * (input_length - 1)) : 0;
    auto log_probs_time_offset =
        beta ? (params.log_probs_time_stride * (input_length - 1)) : 0;
    auto targets_time_offset =
        beta ? (params.tg_target_stride * (target_length - 1)) : 0;
    switch (beta ? (S - s - 1) : s) {
      case 0:
        la = log_probs
            [params.log_probs_token_stride * params.BLANK +
             log_probs_time_offset];
        break;
      case 1:
        la = (target_length == 0) ? neginf
                                  : log_probs
                                        [params.log_probs_token_stride *
                                             targets[targets_time_offset] +
                                         log_probs_time_offset];
        break;
      default:
        la = neginf;
    }
    log_alpha[params.log_alpha_target_stride * s + log_alpha_time_offset] = la;
  }

  // Iterate over the rest of the time steps, for each of the target tokens
  // assigned to this thread.
  for (T_index block_s = 0; block_s < S_max; block_s += tptg) {
    T_index s = block_s + tid;
    T_index target_token;
    bool use_C;
    bool use_B = beta ? ((s + 1) < S) : (s >= 1);
    auto s_prev = beta ? (s + 1) : (s - 1);
    auto s_prev_prev = beta ? (s + 2) : (s - 2);

    if (s < S && target_length > 0) {
      target_token =
          get_target_prime(targets, params.tg_target_stride, s, params.BLANK);
      if IF_CONSTEXPR (beta) {
        use_C = ((s + 2) < S) &&
            (get_target_prime(
                 targets, params.tg_target_stride, s + 2, params.BLANK) !=
             target_token);
      } else {
        use_C = (s >= 2) &&
            (get_target_prime(
                 targets, params.tg_target_stride, s - 2, params.BLANK) !=
             target_token);
      }
    } else {
      target_token = params.BLANK;
      use_C = false;
    }

    for (T_index t_base = 1; t_base < params.max_input_length; t_base++) {
      auto t = beta ? (params.max_input_length - (t_base + 1)) : t_base;
      auto t_prev = beta ? (t + 1) : (t - 1);
      threadgroup_barrier(mem_flags::mem_device);
      if (beta && t == input_length - 1)
        continue;
      if (t < input_length && s < S) {
        // A = log(alpha[t_prev, s])
        auto A = static_cast<T_op>(log_alpha
                                       [params.log_alpha_time_stride * t_prev +
                                        params.log_alpha_target_stride * s]);
        // B = log(alpha[t_prev, s_prev]), or -inf if s_prev is out of bounds
        auto B = use_B
            ? static_cast<T_op>(log_alpha
                                    [params.log_alpha_time_stride * t_prev +
                                     params.log_alpha_target_stride * s_prev])
            : neginf_op;
        // C = log(alpha[t_prev, s_prev_prev]), or -inf if !use_C
        auto C = use_C ? static_cast<T_op>(
                             log_alpha
                                 [params.log_alpha_time_stride * t_prev +
                                  params.log_alpha_target_stride * s_prev_prev])
                       : neginf_op;
        auto y = static_cast<T_op>(
            log_probs
                [t * params.log_probs_time_stride +
                 params.log_probs_token_stride * target_token]);
        log_alpha
            [params.log_alpha_time_stride * t +
             params.log_alpha_target_stride * s] =
                static_cast<T>(logsumexp(A, B, C) + y);

      } else if (s < S_max) {
        log_alpha
            [params.log_alpha_time_stride * t +
             params.log_alpha_target_stride * s] = neginf;
      }
    }
  }
}

template <typename T, typename T_target, typename T_index>
[[max_total_threads_per_threadgroup(1024)]]
kernel void ctc_loss(
    device T* loss [[buffer(0)]],
    device T* log_alpha [[buffer(1)]],
    constant T* log_probs [[buffer(2)]],
    constant T_target* targets [[buffer(3)]],
    constant T_index* input_lengths [[buffer(4)]],
    constant T_index* target_lengths [[buffer(5)]],
    constant T_index* target_batch_offsets [[buffer(6)]],
    constant CTCLossParams<T_index>& params [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {
  using T_op = opmath_t<T>;
  constexpr T_op neginf_op = -numeric_limits<T_op>::infinity();

  auto batch = static_cast<T_index>(tgid);
  T_index input_length = input_lengths[batch];
  T_index target_length = target_lengths[batch];

  if (input_length == 0) {
    if (tid == 0)
      loss[batch] = (target_length == 0) ? T(0) : T(INFINITY);
    return;
  }

  targets += target_batch_offsets[batch];
  log_alpha += batch * params.log_alpha_batch_stride;
  log_probs += batch * params.log_probs_batch_stride;

  calc_log_alpha_beta(
      log_alpha,
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      target_batch_offsets,
      params,
      tid,
      tptg,
      input_length,
      target_length);

  threadgroup_barrier(mem_flags::mem_device);

  if (tid == 0) {
    auto l1 = static_cast<T_op>(
        log_alpha
            [params.log_alpha_time_stride * (input_length - 1) +
             params.log_alpha_target_stride * (target_length * 2)]);
    auto l2 = (target_length > 0)
        ? static_cast<T_op>(
              log_alpha
                  [params.log_alpha_time_stride * (input_length - 1) +
                   params.log_alpha_target_stride * (target_length * 2 - 1)])
        : neginf_op;
    loss[batch] = static_cast<T>(-logsumexp(l1, l2));
  }
}

template <typename T, typename T_target, typename T_index>
[[max_total_threads_per_threadgroup(1024)]]
kernel void ctc_loss_backward_log_beta(
    device T* log_beta [[buffer(0)]],
    constant T* log_probs [[buffer(1)]],
    constant T_target* targets [[buffer(2)]],
    constant T_index* input_lengths [[buffer(3)]],
    constant T_index* target_lengths [[buffer(4)]],
    constant T_index* target_batch_offsets [[buffer(5)]],
    constant CTCLossParams<T_index>& params [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]]) {
  auto batch = static_cast<T_index>(tgid);
  T_index input_length = input_lengths[batch];
  T_index target_length = target_lengths[batch];

  if (input_length == 0) {
    return;
  }

  targets += target_batch_offsets[batch];
  log_beta += batch * params.log_alpha_batch_stride;
  log_probs += batch * params.log_probs_batch_stride;

  calc_log_alpha_beta<T, T_target, T_index, /*beta=*/true>(
      log_beta,
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      target_batch_offsets,
      params,
      tid,
      tptg,
      input_length,
      target_length);
}

// logsumexp-reduce (log_alpha + log_beta) into gradient then apply final
// conversion. Dispatched with one thread per (batch, t) pair.
template <typename T, typename T_target, typename T_index>
kernel void ctc_loss_backward_collect(
    device T* grad [[buffer(0)]],
    constant T* grad_out [[buffer(1)]],
    constant T* log_alpha [[buffer(2)]],
    constant T* log_beta [[buffer(3)]],
    constant T* log_probs [[buffer(4)]],
    constant T_target* targets [[buffer(5)]],
    constant T_index* input_lengths [[buffer(6)]],
    constant T_index* target_lengths [[buffer(7)]],
    constant T* loss [[buffer(8)]],
    constant T_index* target_batch_offsets [[buffer(9)]],
    constant CTCLossBackwardCollectParams<T_index>& params [[buffer(10)]],
    uint2 tid [[thread_position_in_grid]]) {
  using T_op = opmath_t<T>;

  T_index t = static_cast<T_index>(tid.x);
  T_index batch = static_cast<T_index>(tid.y);

  if (t >= params.max_input_length)
    return;

  T_index input_length = input_lengths[batch];
  T_index target_length = target_lengths[batch];
  T_index S_max = 2 * params.max_target_length + 1;
  T_index S = 2 * target_length + 1;

  targets += target_batch_offsets[batch];
  T_index la_batch_offset = batch * params.log_alpha_beta_batch_stride +
      params.log_alpha_beta_time_stride * t;
  log_alpha += la_batch_offset;
  log_beta += la_batch_offset;
  log_probs +=
      batch * params.log_probs_batch_stride + t * params.log_probs_time_stride;
  grad += batch * params.grad_batch_stride + t * params.grad_time_stride;

  T loss_val = loss[batch];
  T grad_out_val = grad_out[batch * params.grad_out_batch_stride];

  // logsumexp-reduce `(log_alpha + log_beta)[t, s]` into `grad[t, target'[s]]`
  for (T_index s = 0; s < S_max; s++) {
    if (s < S) {
      T_index current_target_prime =
          get_target_prime(targets, params.tg_target_stride, s, params.BLANK);
      auto la =
          static_cast<T_op>(log_alpha[params.log_alpha_beta_target_stride * s]);
      auto lb =
          static_cast<T_op>(log_beta[params.log_alpha_beta_target_stride * s]);
      device T& lcab = grad[params.grad_token_stride * current_target_prime];
      lcab = static_cast<T>(logsumexp(static_cast<T_op>(lcab), la + lb));
    }
  }

  // Apply gradient formula for each label
  for (T_index c = 0; c < params.num_labels; c++) {
    device T& res = grad[params.grad_token_stride * c];
    if (t < input_length &&
        (!params.zero_infinity || loss_val != T(INFINITY))) {
      T lp = log_probs[params.log_probs_token_stride * c];
      res = static_cast<T>(
          (precise::exp(static_cast<T_op>(lp)) -
           precise::exp(
               static_cast<T_op>(res) + static_cast<T_op>(loss_val) -
               static_cast<T_op>(lp))) *
          static_cast<T_op>(grad_out_val));
    } else {
      res = T(0);
    }
  }
}

#define INSTANTIATE_CTC_LOSS(T, T_target, T_index)                  \
  template [[host_name("ctc_loss_" #T "_" #T_target "_" #T_index)]] \
  kernel void ctc_loss<T, T_target, T_index>(                       \
      device T*,                                                    \
      device T*,                                                    \
      constant T*,                                                  \
      constant T_target*,                                           \
      constant T_index*,                                            \
      constant T_index*,                                            \
      constant T_index*,                                            \
      constant CTCLossParams<T_index>&,                             \
      uint,                                                         \
      uint,                                                         \
      uint);

#define INSTANTIATE_CTC_LOSS_BACKWARD_LOG_BETA(T, T_target, T_index)  \
  template [[host_name("ctc_loss_backward_log_beta_" #T "_" #T_target \
                       "_" #T_index)]]                                \
  kernel void ctc_loss_backward_log_beta<T, T_target, T_index>(       \
      device T*,                                                      \
      constant T*,                                                    \
      constant T_target*,                                             \
      constant T_index*,                                              \
      constant T_index*,                                              \
      constant T_index*,                                              \
      constant CTCLossParams<T_index>&,                               \
      uint,                                                           \
      uint,                                                           \
      uint);

#define INSTANTIATE_CTC_LOSS_BACKWARD_COLLECT(T, T_target, T_index)  \
  template [[host_name("ctc_loss_backward_collect_" #T "_" #T_target \
                       "_" #T_index)]]                               \
  kernel void ctc_loss_backward_collect<T, T_target, T_index>(       \
      device T*,                                                     \
      constant T*,                                                   \
      constant T*,                                                   \
      constant T*,                                                   \
      constant T*,                                                   \
      constant T_target*,                                            \
      constant T_index*,                                             \
      constant T_index*,                                             \
      constant T*,                                                   \
      constant T_index*,                                             \
      constant CTCLossBackwardCollectParams<T_index>&,               \
      uint2);

#define INSTANTIATE_CTC_LOSS_INDEX_TYPES(T, T_target)           \
  INSTANTIATE_CTC_LOSS(T, T_target, int32_t);                   \
  INSTANTIATE_CTC_LOSS(T, T_target, int64_t);                   \
  INSTANTIATE_CTC_LOSS_BACKWARD_LOG_BETA(T, T_target, int32_t); \
  INSTANTIATE_CTC_LOSS_BACKWARD_LOG_BETA(T, T_target, int64_t); \
  INSTANTIATE_CTC_LOSS_BACKWARD_COLLECT(T, T_target, int32_t);  \
  INSTANTIATE_CTC_LOSS_BACKWARD_COLLECT(T, T_target, int64_t);

#define INSTANTIATE_CTC_LOSS_TARGET_TYPES(T) \
  INSTANTIATE_CTC_LOSS_INDEX_TYPES(T, int);  \
  INSTANTIATE_CTC_LOSS_INDEX_TYPES(T, long);

INSTANTIATE_CTC_LOSS_TARGET_TYPES(float);
INSTANTIATE_CTC_LOSS_TARGET_TYPES(bfloat);
INSTANTIATE_CTC_LOSS_TARGET_TYPES(half);

// ============================================================================
// Phase-2: merge per-threadgroup float partials into loss[0]
// Always dispatched with a single 256-thread threadgroup.
// ============================================================================

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
  acc = c10::metal::threadgroup_sum<float>(smem, acc, lid, tgsz);
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
// NLL Loss 1-D  (input (N,C) log-probs, target (N,) int class indices)
// C++ handles final scale (Mean denominator) after phase-2 reduction.
// Caller must pre-zero grad_in before dispatching nll_loss_bwd.
// ============================================================================

template <typename T>
kernel void nll_loss_fwd_none(
    device const T* log_prob [[buffer(0)]], // (N, C)
    device const long* target [[buffer(1)]], // (N,)
    device const T* weight [[buffer(2)]],
    device T* out [[buffer(3)]], // (N,)
    constant NLLParams& p [[buffer(4)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint n = gid; n < p.N; n += tpg) {
    long t = target[n];
    if (t == p.ignore_index) {
      out[n] = T(0);
      continue;
    }
    if (t < 0 || t >= long(p.C)) {
      TORCH_REPORT_ERROR(
          error_buf,
          "nll_loss: Target ",
          t,
          " is out of bounds [0, ",
          uint(p.C),
          ")");
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
    device const long* target [[buffer(1)]],
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
    long t = target[n];
    if (t == p.ignore_index)
      continue;
    if (t < 0 || t >= long(p.C)) {
      TORCH_REPORT_ERROR(
          error_buf,
          "nll_loss: Target ",
          t,
          " is out of bounds [0, ",
          uint(p.C),
          ")");
      continue;
    }
    float w = p.has_weight ? float(weight[t]) : 1.f;
    acc += -float(log_prob[n * p.C + uint32_t(t)]) * w;
    wacc += w;
  }
  acc = c10::metal::threadgroup_sum<float>(smem, acc, lid, tgsz);
  wacc = c10::metal::threadgroup_sum<float>(wsmem, wacc, lid, tgsz);
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
    device const long* target [[buffer(1)]],
    device const T* weight [[buffer(2)]],
    device T* grad_in [[buffer(3)]], // (N, C) pre-zeroed
    device const T* total_w [[buffer(4)]], // scalar weight sum (Mean)
    constant NLLParams& p [[buffer(5)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]]) {
  for (uint n = gid; n < p.N; n += tpg) {
    long t = target[n];
    if (t == p.ignore_index)
      continue;
    if (t < 0 || t >= long(p.C)) {
      TORCH_REPORT_ERROR(
          error_buf,
          "nll_loss: Target ",
          t,
          " is out of bounds [0, ",
          uint(p.C),
          ")");
      continue;
    }
    float w = p.has_weight ? float(weight[t]) : 1.f;
    float scale;
    if (p.reduction == 0) {
      scale = -float(grad_out[n]) * w;
    } else if (p.reduction == 1) {
      // Mean: divide by the summed weight. total_w == 0 is reachable only in
      // the degenerate all-zero-weight case, where 0/0 -> NaN matches CPU.
      scale = -float(grad_out[0]) * w / float(total_w[0]);
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
      device const long*,                           \
      device const T*,                              \
      device T*,                                    \
      constant NLLParams&,                          \
      device c10::metal::ErrorMessages*,            \
      uint,                                         \
      uint);                                        \
  template [[host_name("nll_loss_fwd_reduce_" #T)]] \
  kernel void nll_loss_fwd_reduce<T>(               \
      device const T*,                              \
      device const long*,                           \
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
      device const long*,                           \
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
