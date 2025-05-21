// Referenced in https://github.com/pytorch/pytorch/issues/153976
// Metal version:
// Copyright (c) 2025 André de Souza Pinto
// baed on CUDA version
// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license
// This is the GPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al.: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// We use the equations from above link, but note that [1] has 1-based indexing and we (of course) use 0-based.
// Graves et al. call the probabilities y, we use log_probs (also calling them inputs)
// ALPHA few optimizations (similar to those here, but also some I didn't take) are described in
// 2. Minmin Sun: http://on-demand.gputechconf.com/gtc/2016/presentation/s6383-minmin-sun-speech-recognition.pdf

// #include <metal_array>
// #include <metal_simdgroup>
// #include <metal_atomic>
#include <metal_stdlib>

using namespace metal;


// Translate targets to augmented targets l'
// If l is l_0, l_1, ..., then l' = blank l_0 blank l_1 blank ... blank
// ref: Graves 2006, section 4.1

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1])
// so if l is l_0 l_1 ... l_(tl-1) then this looks up idx in
// l' = BLANK l_0 BLANK l_1 BLANK ... BLANK l_(tl-1) BLANK
// - note that no bound-checking is done
// - it is important to only call it with idx == 0 if the target length is 0
// - __restrict__ impact to be measured, see
//   https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
template <typename target_t>
inline int64_t get_target_prime(
  device const target_t* target,
    int64_t tg_batch_offset,
    int64_t stride,
    int64_t idx,
    int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[tg_batch_offset + stride * (idx / 2)];
  }
}

// this kernel is a relatively straightforward implementation of the alpha calculation in the forward backward algorithm (section 4.1).
// ALPHA (minor) twist is that we are using log-calculations to enhance numerical stability (log_probs and log_alpha).
// In total it would be more efficient to compute the beta in the same kernel (e.g. cudnn does this). While the beta are not
// needed for the loss itself (just the grad), we can return log_alpha+log_beta (so same space as currently) and the overhead
// is small and the use-case for loss without grad is relatively limited.
// We parallelize by batch and target sequence. Empirically, it is faster to loop over the input (log probs) sequence  and do
// target in parallel, even if it means more frequent __syncthreads.
// In contrast to the cuDNN implementation, we allow large target lengths. For this we need that all previous `s` have been
// computed when we start a new block_s. This is why we have our own for loop here.
template<typename scalar_t, typename target_t>
kernel void ctc_loss_log_alpha_mps_kernel(
  device scalar_t* log_alpha_data         [[buffer(0)]],
  device const scalar_t* log_probs_data  [[buffer(1)]],
  device const int64_t* input_lengths     [[buffer(2)]],
  constant int64_t& max_input_length      [[buffer(3)]],
  device const target_t* targets_data      [[buffer(4)]],
  device const int64_t* target_lengths    [[buffer(5)]],
  constant int64_t& max_target_length     [[buffer(6)]],
  device scalar_t* neg_log_likelihood_data [[buffer(7)]],
  constant int64_t& lp_input_stride       [[buffer(8)]],
  constant int64_t& lp_batch_stride       [[buffer(9)]],
  constant int64_t& lp_char_stride        [[buffer(10)]],
  constant int64_t& la_batch_stride       [[buffer(11)]],
  constant int64_t& la_input_stride       [[buffer(12)]],
  constant int64_t& la_target_stride      [[buffer(13)]],
  device const int64_t* tg_batch_offsets  [[buffer(14)]],
  constant int64_t& tg_target_stride      [[buffer(15)]],
  constant int64_t& batch_size            [[buffer(16)]],
  constant int64_t& BLANK                 [[buffer(17)]],
  // Emular os nomes CUDA
  uint2 threadIdx [[thread_position_in_threadgroup]],         // == threadIdx in CUDA
  uint2 blockIdx  [[threadgroup_position_in_grid]],           // == blockIdx in CUDA
  uint2 blockDim  [[threads_per_threadgroup]],                // == blockDim in CUDA
  uint2 gridDim   [[threadgroups_per_grid]]                   // == gridDim in CUDA
) {
  constexpr scalar_t neginf = -INFINITY;

  // bookkeeping
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t la_batch_offset = b*la_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (b >= batch_size)
    return;

  if (input_length == 0) {
    if (threadIdx.x == 0) {
      scalar_t log_likelihood = target_length == 0 ? 0 : neginf;
      neg_log_likelihood_data[b] = -log_likelihood;
    }
    return;
  }

  // first row (t=0), the three equations for alpha_1 above eq (6)
  for (int64_t block_s = 0; block_s < 2*max_target_length+1; block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t la;
    switch (s) {
    case 0:
      la = log_probs_data[lp_batch_offset + lp_char_stride * BLANK];
      break;
    case 1:
      la = target_length == 0 ? neginf
                              : log_probs_data
                                    [lp_batch_offset +
                                      lp_char_stride *
                                          get_target_prime(
                                              targets_data,
                                              tg_batch_offset,
                                              tg_target_stride,
                                              1,
                                              BLANK)];
      break;
    default:
      la = neginf;
    }
    if (s < 2*max_target_length+1)
      log_alpha_data[la_batch_offset + // la_input_stride * 0
                                     + la_target_stride * s] = la;
  }

  for (int64_t block_s = 0; block_s < 2*max_target_length+1; block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;

    // These two only depend on s, so we can cache them.
    int64_t current_char;       // l_s in eq (6)
    bool have_three;            // flag which of the two cases in eq (6) we have
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_char = get_target_prime(
          targets_data,
          tg_batch_offset,
          tg_target_stride,
          s,
          BLANK);
      have_three =
          ((s > 1) &&
            (get_target_prime(
                targets_data,
                tg_batch_offset,
                tg_target_stride,
                s - 2,
                BLANK) != current_char));
    } else {
      current_char = BLANK;
      have_three = false;
    }
    for (int64_t t=1; t < max_input_length; t++) {
      threadgroup_barrier(mem_flags::mem_none); // on cuda 9 we might use partial synchronization of only the threads within the same batch
      if ((t < input_length) && (s < 2 * target_length + 1)) {
        // only for valid t, s. This is equation (6) and (7), la1, la2, la3 are the three summands,
        // lamax is the maximum for the logsumexp trick.
        scalar_t la1 = log_alpha_data[la_batch_offset + la_input_stride * (t-1) + la_target_stride * s];
        scalar_t lamax = la1;
        scalar_t la2, la3;
        if (s > 0) {
          la2 = log_alpha_data[la_batch_offset + la_input_stride * (t-1) + la_target_stride * (s-1)];
          if (la2 > lamax)
            lamax = la2;
        } else {
          la2 = neginf;
        }
        if (have_three) {
          la3 = log_alpha_data[la_batch_offset + la_input_stride * (t-1) + la_target_stride * (s-2)];
          if (la3 > lamax)
            lamax = la3;
        } else {
          la3 = neginf;
        }
        if (lamax == neginf) // when all are neginf. (then the whole thing is neginf, but we can pretend)
          lamax = 0;

        log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = log(exp(la1-lamax)+exp(la2-lamax)+exp(la3-lamax))+lamax
          + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * current_char];
      } else {
        // otherwise we just set to neginf
        if (s < 2*max_target_length+1)
          log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = neginf;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_none); // on cuda 9 we might use partial synchronization of only the threads within the same batch

  // compute the loss (eq (8))
  if (threadIdx.x == 0) {
    scalar_t l1 = log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * (target_length*2)];
    scalar_t l2 = target_length > 0
        ? log_alpha_data
              [la_batch_offset + la_input_stride * (input_length - 1) +
                la_target_stride * (target_length * 2 - 1)]
        : neginf;
    scalar_t m = ((l1 > l2) ? l1 : l2);
    m = ((m == neginf) ? 0 : m);
    scalar_t log_likelihood = log(exp(l1-m)+exp(l2-m))+m;
    neg_log_likelihood_data[b] = -log_likelihood;
  }
}

// The second (backward) half of the forward backward algorithm, (10) and (11). This is parallel to the
// alpha kernel above. (As mentioned above, it might make sense do the calculation in the alpha kernel.)
template<typename scalar_t, typename target_t>
kernel void ctc_loss_backward_log_beta_mps_kernel(
    device scalar_t* log_beta_data         [[ buffer(0) ]],
    device const scalar_t* log_probs_data  [[ buffer(1) ]],
    device const int64_t* input_lengths    [[ buffer(2) ]],
    constant int64_t& max_input_length     [[ buffer(3) ]],
    device const target_t* targets_data    [[ buffer(4) ]],
    device const int64_t* target_lengths   [[ buffer(5) ]],
    constant int64_t& max_target_length    [[buffer(6)]],
    constant int64_t& lp_input_stride      [[ buffer(7) ]],
    constant int64_t& lp_batch_stride      [[ buffer(8) ]],
    constant int64_t& lp_char_stride       [[ buffer(9) ]],
    constant int64_t& lb_batch_stride      [[ buffer(10) ]],
    constant int64_t& lb_input_stride      [[ buffer(11) ]],
    constant int64_t& lb_target_stride     [[ buffer(12) ]],
    device const int64_t* tg_batch_offsets [[ buffer(13) ]],
    constant int64_t& tg_target_stride     [[ buffer(14) ]],
    constant int64_t& batch_size           [[ buffer(15) ]],
    constant int64_t& BLANK                [[ buffer(16) ]],
    // Emular os nomes CUDA
    uint2 threadIdx [[thread_position_in_threadgroup]],         // == threadIdx in CUDA
    uint2 blockIdx  [[threadgroup_position_in_grid]],           // == blockIdx in CUDA
    uint2 blockDim  [[threads_per_threadgroup]],                // == blockDim in CUDA
    uint2 gridDim   [[threadgroups_per_grid]]                   // == gridDim in CUDA
) {
  constexpr scalar_t neginf = -INFINITY;

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;

  if (b >= batch_size)
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t lb_batch_offset = b*lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (input_length == 0)
    return;

  // "first" row, the beta initialization before eq (10) (t=target_length - differes per batch)
  for (int64_t block_s = 2*max_target_length - (2*max_target_length % blockDim.x); block_s >= 0; block_s -= blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t lb;
    if (s == 2*target_length) {
      lb = log_probs_data[lp_batch_offset + (input_length-1) * lp_input_stride + lp_char_stride * BLANK];
    } else if (s == 2 * target_length - 1) { // false for target_length == 0
      int64_t current_target_prime = get_target_prime(
          targets_data,
          tg_batch_offset,
          tg_target_stride,
          s,
          BLANK);
      lb = log_probs_data[lp_batch_offset + (input_length-1) * lp_input_stride + lp_char_stride * current_target_prime];
    } else {
      lb = neginf;
    }
    if (s < 2*max_target_length+1) {
      log_beta_data[lb_batch_offset + (input_length-1) * lb_input_stride + lb_target_stride * s] = lb;
    }
  }

  // go backward in s
  for (int64_t block_s = 2*max_target_length - (2*max_target_length % blockDim.x); block_s >= 0; block_s -= blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    int64_t current_target_prime;
    bool have_three;
    if (s < 2 * target_length + 1 && target_length > 0) {
      current_target_prime = get_target_prime(
          targets_data,
          tg_batch_offset,
          tg_target_stride,
          s,
          BLANK);
      have_three =
          ((s < 2 * target_length - 1) &&
           (get_target_prime(
                targets_data,
                tg_batch_offset,
                tg_target_stride,
                s + 2,
                BLANK) != current_target_prime));
    } else {
      current_target_prime = BLANK;
      have_three = false;
    }
    // now go backward in t. Note that we need to skip the last timestep that we did above.
    for (int64_t t=max_input_length-2; t>=0; t--) {
      threadgroup_barrier(mem_flags::mem_none); // on cuda 9 we might use partial synchronization of only the threads within the same batch item
      if ((t < input_length - 1) && (s < 2 * target_length + 1)) {
        scalar_t lb1 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * s];
        scalar_t lbmax = lb1;
        scalar_t lb2, lb3;

        if (s < 2*target_length) {
          lb2 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * (s+1)];
          if (lb2 > lbmax)
            lbmax = lb2;
        } else {
          lb2 = neginf;
        }
        if (have_three) {
          lb3 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * (s+2)];
          if (lb3 > lbmax)
            lbmax = lb3;
        } else {
          lb3 = neginf;
        }
        if (lbmax == neginf)
          lbmax = 0;

        scalar_t lb = log(exp(lb1-lbmax)+exp(lb2-lbmax)+exp(lb3-lbmax))+lbmax
          + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * current_target_prime];

        log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s] = lb;
      } else if (
          (s < 2 * max_target_length + 1) &&
          (((target_length == 0) && (s > 0)) || (s >= 2 * target_length + 1) ||
           (t >= input_length))) {
        log_beta_data
            [lb_batch_offset + lb_input_stride * t + lb_target_stride * s] =
                neginf;
      }
    }
  }
}


// This implements the subtrahend of equation (16) for all *nonblank* characters.
// It assumes you have probs in gradient_data when called
// and it modifies gradient_data to be, the gradient.
// In order to facilitate this inplace update, We don't actually do this in logspace.
// (The other variant implemented uses log_space and the differences seem to be
//  not so problematic at least with unit normal distributed test activations.)
// Internally this uses atomicAdd because different threads may write to the same
// gradient position.
// This is parallelised over b and s again.
// Note that for us, the Z of eqn (16) is actually constant for all t and it is the
// likelihood - this is why we use the negative log likelihood below.
// We also multiply by the input gradient to keep with standard autograd style.
// I took this trick from [2], for moderate alphabet sizes a log-space
// calculation (with an atomic log add) is similarly in performance, but for large
// alphabets the inplace nature is a considerable advantage.
template<typename scalar_t, typename target_t>
kernel void ctc_loss_backward_collect_nonblank_mps_kernel(
  device scalar_t* gradient_data                 [[ buffer(0) ]],
  device const scalar_t* grad_out_data           [[ buffer(1) ]],
  device const int64_t& grad_out_batch_stride    [[ buffer(2) ]],
  device const scalar_t* log_alpha_data          [[ buffer(3) ]],
  device const scalar_t* log_beta_data           [[ buffer(4) ]],
  device const scalar_t* log_probs_data          [[ buffer(5) ]],
  device const int64_t* input_lengths            [[ buffer(6) ]],
  device const int64_t* target_lengths           [[ buffer(7) ]],
  device const target_t* targets_data            [[ buffer(8) ]],
  device const scalar_t* neg_log_likelihood_data [[ buffer(9) ]],
  constant int64_t& gr_input_stride              [[ buffer(10) ]],
  constant int64_t& gr_batch_stride              [[ buffer(11) ]],
  constant int64_t& gr_char_stride               [[ buffer(12) ]],
  constant int64_t& lp_input_stride              [[ buffer(13) ]],
  constant int64_t& lp_batch_stride              [[ buffer(14) ]],
  constant int64_t& lp_char_stride               [[ buffer(15) ]],
  constant int64_t& la_batch_stride              [[ buffer(16) ]],
  constant int64_t& la_input_stride              [[ buffer(17) ]],
  constant int64_t& la_target_stride             [[ buffer(18) ]],
  constant int64_t& lb_batch_stride              [[ buffer(19) ]],
  constant int64_t& lb_input_stride              [[ buffer(20) ]],
  constant int64_t& lb_target_stride             [[ buffer(21) ]],
  device const int64_t* tg_batch_offsets         [[ buffer(22) ]],
  constant int64_t& tg_target_stride             [[ buffer(23) ]],
  constant int64_t& batch_size                   [[ buffer(24) ]],
  constant int64_t& zero_infinity                [[ buffer(25) ]],
  // Emular os nomes CUDA
  uint2 threadIdx [[thread_position_in_threadgroup]],         // == threadIdx in CUDA
  uint2 blockIdx  [[threadgroup_position_in_grid]],           // == blockIdx in CUDA
  uint2 blockDim  [[threads_per_threadgroup]],                // == blockDim in CUDA
  uint2 gridDim   [[threadgroups_per_grid]]                   // == gridDim in CUDA
) {
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t s = threadIdx.x + ((int64_t) blockIdx.x) * blockDim.x; // note, this directly indexes into targets, not targets prime!

  if (b >= batch_size)
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t gr_batch_offset = b*gr_batch_stride;
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t la_batch_offset = b*la_batch_stride;
  int64_t lb_batch_offset = b*lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (s >= target_length)
    return;

  int64_t target = targets_data[tg_batch_offset + s * tg_target_stride];
  scalar_t nll = neg_log_likelihood_data[b];
  scalar_t gr =  grad_out_data[b * grad_out_batch_stride];

  if (zero_infinity && nll == INFINITY)
    return;

  for (int64_t t = 0; t < input_length; t++) {
    scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * target];
    atomic_fetch_add_explicit((device atomic_float*)&gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * target],
      -exp(log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * (s*2+1)]
                + log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * (s*2+1)]
                + nll - lp) * gr, memory_order_relaxed);
  }
}

// Kernel Metal equivalente a ctc_loss_backward_collect_gpu_kernel
// This is the naive implementation of equation (16). It is parallelised in batch and input timestep.
// It appears to be faster than the above method for small batch sizes.
template<typename scalar_t, typename target_t>
kernel void ctc_loss_backward_collect_mps_kernel(
  device scalar_t* gradient_data                 [[ buffer(0) ]],
  device const scalar_t* grad_out_data           [[ buffer(1) ]],
  constant int64_t& grad_out_batch_stride        [[ buffer(2) ]],
  device const scalar_t* log_alpha_data          [[ buffer(3) ]],
  device const scalar_t* log_beta_data           [[ buffer(4) ]],
  device const scalar_t* log_probs_data          [[ buffer(5) ]],
  device const int64_t* input_lengths            [[ buffer(6) ]],
  constant int64_t& max_input_length             [[ buffer(7) ]],
  device const target_t* targets_data            [[ buffer(8) ]],
  device const int64_t* target_lengths           [[ buffer(9) ]],
  constant int64_t& max_target_length            [[ buffer(10) ]],
  device const scalar_t* neg_log_likelihood_data [[ buffer(11) ]],
  constant int64_t& gr_input_stride              [[ buffer(12) ]],
  constant int64_t& gr_batch_stride              [[ buffer(13) ]],
  constant int64_t& gr_char_stride               [[ buffer(14) ]],
  constant int64_t& lp_input_stride              [[ buffer(15) ]],
  constant int64_t& lp_batch_stride              [[ buffer(16) ]],
  constant int64_t& lp_char_stride               [[ buffer(17) ]],
  constant int64_t& la_batch_stride              [[ buffer(18) ]],
  constant int64_t& la_input_stride              [[ buffer(19) ]],
  constant int64_t& la_target_stride             [[ buffer(20) ]],
  constant int64_t& lb_batch_stride              [[ buffer(21) ]],
  constant int64_t& lb_input_stride              [[ buffer(22) ]],
  constant int64_t& lb_target_stride             [[ buffer(23) ]],
  device const int64_t* tg_batch_offsets         [[ buffer(24) ]],
  constant int64_t& tg_target_stride             [[ buffer(25) ]],
  constant int64_t& batch_size                   [[ buffer(26) ]],
  constant int64_t& num_labels                   [[ buffer(27) ]],
  constant int64_t& BLANK                        [[ buffer(28) ]],
  constant int64_t& zero_infinity                [[ buffer(29) ]],
  // Emular os nomes CUDA
  uint2 threadIdx [[thread_position_in_threadgroup]],         // == threadIdx in CUDA
  uint2 blockIdx  [[threadgroup_position_in_grid]],           // == blockIdx in CUDA
  uint2 blockDim  [[threads_per_threadgroup]],                // == blockDim in CUDA
  uint2 gridDim   [[threadgroups_per_grid]]                   // == gridDim in CUDA
) {

  constexpr scalar_t neginf = -INFINITY;
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t t = threadIdx.x + ((int64_t) blockIdx.x) * blockDim.x;

  if ((t >= max_input_length) || (b >= batch_size))
    return;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t gr_batch_offset = b*gr_batch_stride;
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t la_batch_offset = b*la_batch_stride;
  int64_t lb_batch_offset = b*lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
  for (int s = 0; s < 2*max_target_length+1; s++) {
    if (s < 2 * target_length + 1) { // if target_length == 0, s == 0
      int64_t current_target_prime = get_target_prime(
          targets_data,
          tg_batch_offset,
          tg_target_stride,
          s,
          BLANK);
      scalar_t log_alpha_beta = (log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s]
                                 + log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s]);
      device scalar_t& lcab = gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * current_target_prime];
      if (lcab == neginf) {
        lcab = log_alpha_beta;
      } else {
        scalar_t max = ((lcab > log_alpha_beta) ? lcab : log_alpha_beta);
        lcab = log(exp(lcab-max)+exp(log_alpha_beta-max))+max;
      }
    }
  }

  scalar_t nll = neg_log_likelihood_data[b];
  scalar_t gr =  grad_out_data[b * grad_out_batch_stride];

  for (int64_t c = 0; c < num_labels; c++) {
    device scalar_t& res = gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * c];
    if (t < input_length && (! zero_infinity || nll != INFINITY)) {
      scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * c];
      res = (exp(lp)-exp(res + nll - lp)) * gr;
    }
    else {
      res = 0.;
    }
  }
}

// This is to zero gradients which corresponding to the out-of-sequence position
// Those gradients should not be used in any model update since the input
// elements are padded
template<typename scalar_t>
kernel void ctc_loss_zero_padded_gradients(
  device scalar_t* gradient_data       [[buffer(0)]],   // (T, B, D) layout
  device const int64_t* input_lengths  [[buffer(1)]],   // (B, ) layout
  constant int64_t& gr_timestep_stride [[buffer(2)]],
  constant int64_t& gr_batch_stride    [[buffer(3)]],
  constant int64_t& gr_label_stride    [[buffer(4)]],
  constant int64_t& max_input_length   [[buffer(5)]],   // T
  constant int64_t& batch_size         [[buffer(6)]],   // B
  constant int64_t& num_labels         [[buffer(7)]],   // D
  // Emular os nomes CUDA
  uint2 threadIdx [[thread_position_in_threadgroup]],         // == threadIdx in CUDA
  uint2 blockIdx  [[threadgroup_position_in_grid]],           // == blockIdx in CUDA
  uint2 blockDim  [[threads_per_threadgroup]],                // == blockDim in CUDA
  uint2 gridDim   [[threadgroups_per_grid]]                   // == gridDim in CUDA
) {
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t t = threadIdx.x + ((int64_t) blockIdx.x) * blockDim.x;

  if (b >= batch_size || t >= max_input_length) {
    return;
  }

  scalar_t input_length = input_lengths[b];
  if (t >= input_length) {
    for (int l = 0; l < num_labels; l++)
      gradient_data[
        t * gr_timestep_stride + b * gr_batch_stride + l * gr_label_stride]
      = 0.0f;
  }
}

#define INSTANTIATE_ALPHA(scalar_t, target_t)                       \
template [[host_name("ctc_loss_log_alpha_mps_kernel")]]             \
kernel void ctc_loss_log_alpha_mps_kernel<scalar_t, target_t>(      \
  device scalar_t* log_alpha_data         [[buffer(0)]],            \
  device const scalar_t* log_probs_data  [[buffer(1)]],             \
  device const int64_t* input_lengths     [[buffer(2)]],            \
  constant int64_t& max_input_length      [[buffer(3)]],            \
  device const target_t* targets_data      [[buffer(4)]],           \
  device const int64_t* target_lengths    [[buffer(5)]],            \
  constant int64_t& max_target_length     [[buffer(6)]],            \
  device scalar_t* neg_log_likelihood_data [[buffer(7)]],           \
  constant int64_t& lp_input_stride       [[buffer(8)]],            \
  constant int64_t& lp_batch_stride       [[buffer(9)]],            \
  constant int64_t& lp_char_stride        [[buffer(10)]],           \
  constant int64_t& la_batch_stride       [[buffer(11)]],           \
  constant int64_t& la_input_stride       [[buffer(12)]],           \
  constant int64_t& la_target_stride      [[buffer(13)]],           \
  device const int64_t* tg_batch_offsets  [[buffer(14)]],           \
  constant int64_t& tg_target_stride      [[buffer(15)]],           \
  constant int64_t& batch_size            [[buffer(16)]],           \
  constant int64_t& BLANK                 [[buffer(17)]],           \
  uint2 threadIdx [[thread_position_in_threadgroup]],               \
  uint2 blockIdx  [[threadgroup_position_in_grid]],                 \
  uint2 blockDim  [[threads_per_threadgroup]],                      \
  uint2 gridDim   [[threadgroups_per_grid]]                         \
)

#define INSTANTIATE_BETA(scalar_t, target_t)                    \
template [[host_name("ctc_loss_backward_log_beta_mps_kernel")]] \
kernel void ctc_loss_backward_log_beta_mps_kernel(              \
  device scalar_t* log_beta_data        [[ buffer(0) ]],        \
  device const scalar_t* log_probs_data [[ buffer(1) ]],        \
  device const int64_t* input_lengths [[ buffer(2) ]],          \
  constant int64_t& max_input_length      [[ buffer(3) ]],      \
  device const target_t* targets_data  [[ buffer(4) ]],         \
  device const int64_t* target_lengths [[ buffer(5) ]],         \
  constant int64_t& max_target_length     [[buffer(6)]],        \
  constant int64_t& lp_input_stride       [[ buffer(7) ]],      \
  constant int64_t& lp_batch_stride       [[ buffer(8) ]],      \
  constant int64_t& lp_char_stride        [[ buffer(9) ]],      \
  constant int64_t& lb_batch_stride       [[ buffer(10) ]],     \
  constant int64_t& lb_input_stride       [[ buffer(11) ]],     \
  constant int64_t& lb_target_stride      [[ buffer(12) ]],     \
  device const int64_t* tg_batch_offsets [[ buffer(13) ]],      \
  constant int64_t& tg_target_stride      [[ buffer(14) ]],     \
  constant int64_t& batch_size            [[ buffer(15) ]],     \
  constant int64_t& BLANK                 [[ buffer(16) ]],     \
  uint2 threadIdx [[thread_position_in_threadgroup]],           \
  uint2 blockIdx  [[threadgroup_position_in_grid]],             \
  uint2 blockDim  [[threads_per_threadgroup]],                  \
  uint2 gridDim   [[threadgroups_per_grid]]                     \
)

#define INSTANTIATE_COLLECT_NONBLANK(scalar_t, target_t)                \
template [[host_name("ctc_loss_backward_collect_nonblank_mps_kernel")]] \
kernel void ctc_loss_backward_collect_nonblank_mps_kernel(              \
  device scalar_t* gradient_data                 [[ buffer(0) ]],       \
  device const scalar_t* grad_out_data           [[ buffer(1) ]],       \
  device const int64_t& grad_out_batch_stride    [[ buffer(2) ]],       \
  device const scalar_t* log_alpha_data          [[ buffer(3) ]],       \
  device const scalar_t* log_beta_data           [[ buffer(4) ]],       \
  device const scalar_t* log_probs_data          [[ buffer(5) ]],       \
  device const int64_t* input_lengths            [[ buffer(6) ]],       \
  device const int64_t* target_lengths           [[ buffer(7) ]],       \
  device const target_t* targets_data            [[ buffer(8) ]],       \
  device const scalar_t* neg_log_likelihood_data [[ buffer(9) ]],       \
  constant int64_t& gr_input_stride              [[ buffer(10) ]],      \
  constant int64_t& gr_batch_stride              [[ buffer(11) ]],      \
  constant int64_t& gr_char_stride               [[ buffer(12) ]],      \
  constant int64_t& lp_input_stride              [[ buffer(13) ]],      \
  constant int64_t& lp_batch_stride              [[ buffer(14) ]],      \
  constant int64_t& lp_char_stride               [[ buffer(15) ]],      \
  constant int64_t& la_batch_stride              [[ buffer(16) ]],      \
  constant int64_t& la_input_stride              [[ buffer(17) ]],      \
  constant int64_t& la_target_stride             [[ buffer(18) ]],      \
  constant int64_t& lb_batch_stride              [[ buffer(19) ]],      \
  constant int64_t& lb_input_stride              [[ buffer(20) ]],      \
  constant int64_t& lb_target_stride             [[ buffer(21) ]],      \
  device const int64_t* tg_batch_offsets         [[ buffer(22) ]],      \
  constant int64_t& tg_target_stride             [[ buffer(23) ]],      \
  constant int64_t& batch_size                   [[ buffer(24) ]],      \
  constant int64_t& zero_infinity                [[ buffer(25) ]],      \
  uint2 threadIdx [[thread_position_in_threadgroup]],                   \
  uint2 blockIdx  [[threadgroup_position_in_grid]],                     \
  uint2 blockDim  [[threads_per_threadgroup]],                          \
  uint2 gridDim   [[threadgroups_per_grid]]                             \
)

#define INSTANTIATE_COLLECT(scalar_t, target_t)                    \
template [[host_name("ctc_loss_backward_collect_mps_kernel")]]     \
kernel void ctc_loss_backward_collect_mps_kernel(                  \
  device scalar_t* gradient_data                 [[ buffer(0) ]],  \
  device const scalar_t* grad_out_data           [[ buffer(1) ]],  \
  constant int64_t& grad_out_batch_stride        [[ buffer(2) ]],  \
  device const scalar_t* log_alpha_data          [[ buffer(3) ]],  \
  device const scalar_t* log_beta_data           [[ buffer(4) ]],  \
  device const scalar_t* log_probs_data          [[ buffer(5) ]],  \
  device const int64_t* input_lengths            [[ buffer(6) ]],  \
  constant int64_t& max_input_length             [[ buffer(7) ]],  \
  device const target_t* targets_data            [[ buffer(8) ]],  \
  device const int64_t* target_lengths           [[ buffer(9) ]],  \
  constant int64_t& max_target_length            [[ buffer(10) ]], \
  device const scalar_t* neg_log_likelihood_data [[ buffer(11) ]], \
  constant int64_t& gr_input_stride              [[ buffer(12) ]], \
  constant int64_t& gr_batch_stride              [[ buffer(13) ]], \
  constant int64_t& gr_char_stride               [[ buffer(14) ]], \
  constant int64_t& lp_input_stride              [[ buffer(15) ]], \
  constant int64_t& lp_batch_stride              [[ buffer(16) ]], \
  constant int64_t& lp_char_stride               [[ buffer(17) ]], \
  constant int64_t& la_batch_stride              [[ buffer(18) ]], \
  constant int64_t& la_input_stride              [[ buffer(19) ]], \
  constant int64_t& la_target_stride             [[ buffer(20) ]], \
  constant int64_t& lb_batch_stride              [[ buffer(21) ]], \
  constant int64_t& lb_input_stride              [[ buffer(22) ]], \
  constant int64_t& lb_target_stride             [[ buffer(23) ]], \
  device const int64_t* tg_batch_offsets         [[ buffer(24) ]], \
  constant int64_t& tg_target_stride             [[ buffer(25) ]], \
  constant int64_t& batch_size                   [[ buffer(26) ]], \
  constant int64_t& num_labels                   [[ buffer(27) ]], \
  constant int64_t& BLANK                        [[ buffer(28) ]], \
  constant int64_t& zero_infinity                [[ buffer(29) ]], \
  uint2 threadIdx [[thread_position_in_threadgroup]],              \
  uint2 blockIdx  [[threadgroup_position_in_grid]],                \
  uint2 blockDim  [[threads_per_threadgroup]],                     \
  uint2 gridDim   [[threadgroups_per_grid]]                        \
)

#define INSTANTIATE_PADDED_GRADIENTS(scalar_t, target_t)          \
template [[host_name("ctc_loss_zero_padded_gradients")]] \
kernel void ctc_loss_zero_padded_gradients(              \
  device scalar_t* gradient_data       [[buffer(0)]],    \
  device const int64_t* input_lengths  [[buffer(1)]],    \
  constant int64_t& gr_timestep_stride [[buffer(2)]],    \
  constant int64_t& gr_batch_stride    [[buffer(3)]],    \
  constant int64_t& gr_label_stride    [[buffer(4)]],    \
  constant int64_t& max_input_length   [[buffer(5)]],    \
  constant int64_t& batch_size         [[buffer(6)]],    \
  constant int64_t& num_labels         [[buffer(7)]],    \
  uint2 threadIdx [[thread_position_in_threadgroup]],    \
  uint2 blockIdx  [[threadgroup_position_in_grid]],      \
  uint2 blockDim  [[threads_per_threadgroup]],           \
  uint2 gridDim   [[threadgroups_per_grid]]              \
)

INSTANTIATE_ALPHA(float, int64_t);
INSTANTIATE_BETA(float, int64_t);
INSTANTIATE_COLLECT_NONBLANK(float, int64_t);
INSTANTIATE_COLLECT(float, int64_t);
INSTANTIATE_PADDED_GRADIENTS(float, int64_t);
