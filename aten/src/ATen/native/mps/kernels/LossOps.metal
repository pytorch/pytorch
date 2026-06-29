#include <ATen/native/mps/kernels/LossOps.h>
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
