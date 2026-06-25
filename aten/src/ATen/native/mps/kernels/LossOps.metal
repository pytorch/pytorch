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
  constexpr T neginf = -numeric_limits<T>::infinity();
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

  T_index S_max = 2 * params.max_target_length + 1;
  T_index S = 2 * target_length + 1;

  // Initialize first time step for all the target tokens assigned to this
  // thread.
  for (T_index s = tid; s < S_max; s += tptg) {
    T la;
    switch (s) {
      case 0:
        la = log_probs[params.log_probs_token_stride * params.BLANK];
        break;
      case 1:
        la = (target_length == 0)
            ? neginf
            : log_probs[params.log_probs_token_stride * targets[0]];
        break;
      default:
        la = neginf;
    }
    if (s < S_max) {
      log_alpha[params.log_alpha_target_stride * s] = la;
    }
  }

  // Iterate over the rest of the time steps, for each of the target tokens
  // assigned to this thread.
  for (T_index block_s = 0; block_s < S_max; block_s += tptg) {
    T_index s = block_s + tid;
    T_index target_token;
    bool use_C;

    if (s < S && target_length > 0) {
      target_token =
          get_target_prime(targets, params.tg_target_stride, s, params.BLANK);
      use_C = (s >= 2) &&
          (get_target_prime(
               targets, params.tg_target_stride, s - 2, params.BLANK) !=
           target_token);
    } else {
      target_token = params.BLANK;
      use_C = false;
    }

    for (T_index t = 1; t < params.max_input_length; t++) {
      threadgroup_barrier(mem_flags::mem_device);
      if (t < input_length && s < S) {
        // A = log(alpha[t-1, s])
        auto A = static_cast<T_op>(log_alpha
                                       [params.log_alpha_time_stride * (t - 1) +
                                        params.log_alpha_target_stride * s]);
        // B = log(alpha[t-1, s-1]), or 0 if s-1 is out of bounds
        auto B = (s >= 1)
            ? static_cast<T_op>(log_alpha
                                    [params.log_alpha_time_stride * (t - 1) +
                                     params.log_alpha_target_stride * (s - 1)])
            : neginf_op;
        // C = log(alpha[t-1, s-2]), or 0 if !use_C
        auto C = use_C
            ? static_cast<T_op>(log_alpha
                                    [params.log_alpha_time_stride * (t - 1) +
                                     params.log_alpha_target_stride * (s - 2)])
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

#define INSTANTIATE_CTC_LOSS_INDEX_TYPES(T, T_target) \
  INSTANTIATE_CTC_LOSS(T, T_target, int32_t);         \
  INSTANTIATE_CTC_LOSS(T, T_target, int64_t);

#define INSTANTIATE_CTC_LOSS_TARGET_TYPES(T) \
  INSTANTIATE_CTC_LOSS_INDEX_TYPES(T, int);  \
  INSTANTIATE_CTC_LOSS_INDEX_TYPES(T, long);

INSTANTIATE_CTC_LOSS_TARGET_TYPES(float);
INSTANTIATE_CTC_LOSS_TARGET_TYPES(bfloat);
INSTANTIATE_CTC_LOSS_TARGET_TYPES(half);
