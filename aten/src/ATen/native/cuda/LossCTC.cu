// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license
// This is the GPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// We use the equations from above link, but note that [1] has 1-based indexing and we (of course) use 0-based.
// Graves et al call the probabilities y, we use log_probs (also calling them inputs)
// A few optimizations (similar to those here, but also some I didn't take) are described in
// 2. Minmin Sun: http://on-demand.gputechconf.com/gtc/2016/presentation/s6383-minmin-sun-speech-recognition.pdf
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>
#include <c10/macros/Macros.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_ctc_loss_backward_native.h>
#include <ATen/ops/_ctc_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

#include <type_traits>
#include <numeric>

namespace at::native {

namespace {

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1])
// so if l is l_0 l_1 ... l_(tl-1) then this looks up idx in
// l' = BLANK l_0 BLANK l_1 BLANK ... BLANK l_(tl-1) BLANK
// - note that no bound-checking is done
// - it is important to only call it with idx == 0 if the target length is 0
// - __restrict__ impact to be measured, see
//   https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
template <typename target_t>
__device__ static inline int64_t get_target_prime(
    const target_t* __restrict__ target,
    int64_t offset,
    int64_t stride,
    int64_t idx,
    int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

// this kernel is a relatively straightforward implementation of the alpha calculation in the forward backward algorithm (section 4.1).
// A (minor) twist is that we are using log-calculations to enhance numerical stability (log_probs and log_alpha).
// In total it would be more efficient to compute the beta in the same kernel (e.g. cudnn does this). While the beta are not
// needed for the loss itself (just the grad), we can return log_alpha+log_beta (so same space as currently) and the overhead
// is small and the use-case for loss without grad is relatively limited.
// We parallelize by batch and target sequence. Empirically, it is faster to loop over the input (log probs) sequence  and do
// target in parallel, even if it means more frequent __syncthreads.
// In contrast to the cuDNN implementation, we allow large target lengths. For this we need that all previous `s` have been
// computed when we start a new block_s. This is why we have our own for loop here.
template<typename scalar_t, typename target_t>
__global__ void
#if defined (USE_ROCM)
C10_LAUNCH_BOUNDS_2((std::is_same<scalar_t, float>::value ? 1024 : 896), 1)
#endif
ctc_loss_log_alpha_gpu_kernel(scalar_t* __restrict__ log_alpha_data,
                                    const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths, int64_t max_input_length,
                                    const target_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths, int64_t max_target_length,
                                    scalar_t* __restrict__ neg_log_likelihood_data,
                                    int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
                                    const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                    int64_t batch_size, int64_t BLANK) {

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
      log_alpha_data[la_batch_offset + /* la_input_stride * 0 */ + la_target_stride * s] = la;
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
      __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch
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

        log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = std::log(std::exp(la1-lamax)+std::exp(la2-lamax)+std::exp(la3-lamax))+lamax
          + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * current_char];
      } else {
        // otherwise we just set to neginf
        if (s < 2*max_target_length+1)
          log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = neginf;
      }
    }
  }
  __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch

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
    scalar_t log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
    neg_log_likelihood_data[b] = -log_likelihood;
  }
}

// The forward computation. Lot's of admin and a call to the alpha kernel.
// Note: we do not check that the labels are in the valid range. As we use
// them for indexing in the kernels, you'll see memory errors when you
// pass corrupt labels.
// We support both a 2-dimensional tensor as targets (one set of targets in each row) and
// a 1-dimensional tensor where all targets are concatenated (and we use target_lengths
// to figure out where they begin).
// We return log_alpha (currently, might change to (log_alpha+log_beta) to be passed to the
// backward. The dispatch function will only return the loss.
template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> ctc_loss_gpu_template(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  CheckedFrom c = "ctc_loss_gpu";
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  checkAllSameGPU(c, {log_probs_arg, targets_arg});

  checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK(input_lengths.size() == static_cast<size_t>(batch_size), "input_lengths must be of size batch_size");
  TORCH_CHECK(target_lengths.size() == static_cast<size_t>(batch_size), "target_lengths must be of size batch_size");

  int64_t tg_target_stride;

  int64_t max_target_length = 0;
  auto tg_batch_offsets = at::empty({batch_size}, at::device(at::kCPU).dtype(at::kLong));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, pos);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(targets.size(1) >= max_target_length,
             "Expected tensor to have size at least ", max_target_length, " at dimension 1, but got size ", targets.size(1), " for ", targets_arg,
             " (while checking arguments for ", c, ")");
  }
  int64_t max_input_length = log_probs.size(0);
  for (int64_t b = 0; b < batch_size; b++) {
    TORCH_CHECK(input_lengths[b] <= max_input_length,
             "Expected input_lengths to have value at most ", max_input_length, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
  }

  auto target_lengths_t = at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t = at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.cuda();

  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  // Very likely, we could be more clever here, e.g. learning (or generalizing and reusing) from SoftMax.cu...
  constexpr int max_threads = std::is_same<scalar_t, float>::value ? 1024 : 768; // we need 72 or so 32 bit registers for double
  int threads_target = max_threads;
  while (threads_target / 2 >= 2*max_target_length+1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int) batch_size);
  dim3 block(threads_target, threads_batch);
  dim3 grid(1, (batch_size+threads_batch-1)/threads_batch);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ctc_loss_log_alpha_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>(
                      log_alpha.mutable_data_ptr<scalar_t>(),
                      log_probs.const_data_ptr<scalar_t>(), input_lengths_t.const_data_ptr<int64_t>(), log_probs.size(0),
                      targets.const_data_ptr<target_t>(), target_lengths_t.const_data_ptr<int64_t>(), max_target_length,
                      neg_log_likelihood.mutable_data_ptr<scalar_t>(),
                      log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
                      log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
                      tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
                      batch_size, BLANK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(neg_log_likelihood, log_alpha);
}

// The second (backward) half of the forward backward algorithm, (10) and (11). This is parallel to the
// alpha kernel above. (As mentioned above, it might make sense do the calculation in the alpha kernel.)
template<typename scalar_t, typename target_t>
__global__ void
C10_LAUNCH_BOUNDS_2((std::is_same<scalar_t, float>::value ? 1024 : 896), 1)
ctc_loss_backward_log_beta_gpu_kernel(scalar_t* __restrict__ log_beta_data,
                                      const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths, int64_t max_input_length,
                                      const target_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths, int64_t max_target_length,
                                      int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                      int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
                                      const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                      int64_t batch_size, int64_t BLANK) {
  constexpr scalar_t neginf = -INFINITY;

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t lb_batch_offset = b*lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (b >= batch_size)
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
      __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch item
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

        scalar_t lb = std::log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax)+std::exp(lb3-lbmax))+lbmax
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
__global__ void
#if defined (USE_ROCM)
C10_LAUNCH_BOUNDS_2((std::is_same<scalar_t, float>::value ? 1024 : 896), 1)
#endif
ctc_loss_backward_collect_nonblank_gpu_kernel(scalar_t* __restrict__ gradient_data,
                                                     const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
                                                     const scalar_t* __restrict__ log_alpha_data, const scalar_t* __restrict__ log_beta_data,
                                                     const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths,
                                                     const target_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths,
                                                     const scalar_t* __restrict__ neg_log_likelihood_data,
                                                     int64_t gr_input_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
                                                     int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                                     int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
                                                     int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
                                                     const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                              int64_t batch_size, bool zero_infinity) {
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t s = threadIdx.x + blockIdx.x * blockDim.x; // note, this directly indexes into targets, not targets prime!

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
    gpuAtomicAddNoReturn(&gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * target],
              -std::exp(log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * (s*2+1)]
                        + log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * (s*2+1)]
                        + nll - lp) * gr);
  }
}

// This is the naive implementation of equation (16). It is parallelised in batch and input timestep.
// It appears to be faster than the above method for small batch sizes.
template<typename scalar_t, typename target_t>
__global__ void
#if defined (USE_ROCM)
C10_LAUNCH_BOUNDS_2((std::is_same<scalar_t, float>::value ? 1024 : 896), 1)
#endif
ctc_loss_backward_collect_gpu_kernel(scalar_t* __restrict__ gradient_data,
                                                     const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
                                                     const scalar_t* __restrict__ log_alpha_data, const scalar_t* __restrict__ log_beta_data,
                                                     const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths, int64_t max_input_length,
                                                     const target_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths, int64_t max_target_length,
                                                     const scalar_t* __restrict__ neg_log_likelihood_data,
                                                     int64_t gr_input_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
                                                     int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                                     int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
                                                     int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
                                                     const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                     int64_t batch_size, int64_t num_labels, int64_t BLANK, bool zero_infinity) {

  constexpr scalar_t neginf = -INFINITY;
  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
  int64_t t = threadIdx.x + blockIdx.x * blockDim.x;

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
      scalar_t& lcab = gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * current_target_prime];
      if (lcab == neginf) {
        lcab = log_alpha_beta;
      } else {
        scalar_t max = ((lcab > log_alpha_beta) ? lcab : log_alpha_beta);
        lcab = std::log(std::exp(lcab-max)+std::exp(log_alpha_beta-max))+max;
      }
    }
  }

  scalar_t nll = neg_log_likelihood_data[b];
  scalar_t gr =  grad_out_data[b * grad_out_batch_stride];

  for (int64_t c = 0; c < num_labels; c++) {
    scalar_t& res = gradient_data[gr_batch_offset + t * gr_input_stride + gr_char_stride * c];
    if (t < input_length && (! zero_infinity || nll != INFINITY)) {
      scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * c];
      res = (std::exp(lp)-std::exp(res + nll - lp)) * gr;
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
__global__ void
#if defined (USE_ROCM)
C10_LAUNCH_BOUNDS_2((std::is_same<scalar_t, float>::value ? 1024 : 896), 1)
#endif
ctc_loss_zero_padded_gradients(
    scalar_t* __restrict__ gradient_data,   /* (T, B, D) layout */
    const int64_t* __restrict__ input_lengths, /* (B, ) layout */
    int64_t gr_timestep_stride,
    int64_t gr_batch_stride,
    int64_t gr_label_stride,
    int64_t max_input_length, /* T */
    int64_t batch_size, /* B */
    int64_t num_labels  /* D */ ) {
      int64_t b = threadIdx.y + blockIdx.y * blockDim.y;
      int64_t t = threadIdx.x + blockIdx.x * blockDim.x;

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


// The backward. It essentially computes eq 16 by using the above kernels.
// We don't do a lot of checking as we envision this to be called only when backpropagating through a (well-checked) forward.
template<typename scalar_t, ScalarType target_scalar_type>
Tensor ctc_loss_backward_gpu_template(const Tensor& grad_out, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                                      const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  constexpr scalar_t neginf = -INFINITY;
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  int64_t tg_target_stride;

  int64_t max_target_length;
  auto tg_batch_offsets = at::empty({batch_size}, TensorOptions(at::CPU(kLong)));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length = log_alpha.size(2)/2; // targets.size(1) might be larger
  }
  auto target_lengths_t = at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t = at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.cuda();

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  log_beta.fill_(neginf);

  Tensor grad = at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // initialization for log(sum (alpha beta))

  // As above, there may be better configurations to use.
  constexpr int max_threads = std::is_same<scalar_t, float>::value ? 1024 : 896; // we need 72 or so 32 bit registers for double
  int threads_target = max_threads;
  while (threads_target / 2 >= 2*max_target_length+1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int) batch_size);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  {
    dim3 block(threads_target, threads_batch);
    dim3 grid(1, (batch_size+threads_batch-1)/threads_batch);
    ctc_loss_backward_log_beta_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>
      (log_beta.mutable_data_ptr<scalar_t>(),
       log_probs.const_data_ptr<scalar_t>(), input_lengths_t.const_data_ptr<int64_t>(), log_probs.size(0),
       targets.const_data_ptr<target_t>(), target_lengths_t.const_data_ptr<int64_t>(), max_target_length,
       log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
       log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
       tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
       batch_size, BLANK);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Very crude heuristic for what is a small problem., based on linearly regressing problem dimensions on
  // the (capped) difference of timings.
  // Note that for OK problems target length <= input length, so we
  // only consider input length.
  bool is_large = (2*log_probs.size(0)+(24*batch_size)/10+(2*num_labels)/10) > 450;
  if (is_large) { // large alphabet, large batch
    // this computes the probs, minuend in (16)
    at::exp_out(grad, log_probs);
    // now we compute the subtrahend for the blanks. It is a straightforward reduction because we know that
    // blanks are in every other position.
    // maybe we should kernelize this, too.
    auto grad_blank = grad.narrow(2, BLANK, 1);
    grad_blank -= (at::logsumexp(log_alpha.as_strided({batch_size, log_alpha.size(1), max_target_length+1},
                                                      {log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2)*2})
                                 + log_beta.as_strided({batch_size, log_beta.size(1), max_target_length+1},
                                                       {log_beta.stride(0), log_beta.stride(1), log_beta.stride(2)*2}),
                                 2, true)
                   .permute({1, 0, 2})
                   .add_(neg_log_likelihood.view({1, batch_size, 1}))
                   .sub_(log_probs.narrow(2, BLANK, 1))
                   .exp_()
                   );
    // scale by output gradient (blanks and first summand of non-blanks)
    grad *= grad_out.view({1, batch_size, 1});
    if (zero_infinity) {
      grad = at::where(neg_log_likelihood.view({1, batch_size, 1}) == Scalar(INFINITY), at::zeros({}, grad.options()), grad);
    }

    // For the non-blank characters, we use a kernel to compute the subtrahend.
    // Again we might configure block and grid in a better way.
    int threads_target = max_threads;
    while (threads_target / 2 >= max_target_length && threads_target > 1) {
      threads_target /= 2;
    }
    int threads_batch = std::min(max_threads / threads_target, (int) batch_size);
    dim3 block(threads_target, threads_batch);
    dim3 grid(
        std::max<int>(
            (max_target_length + threads_target - 1) / threads_target, 1),
        (batch_size + threads_batch - 1) / threads_batch,
        1);
    ctc_loss_backward_collect_nonblank_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>
      (grad.mutable_data_ptr<scalar_t>(),
       grad_out.const_data_ptr<scalar_t>(), grad_out.stride(0),
       log_alpha.const_data_ptr<scalar_t>(), log_beta.const_data_ptr<scalar_t>(),
       log_probs.const_data_ptr<scalar_t>(), input_lengths_t.const_data_ptr<int64_t>(),
       targets.const_data_ptr<target_t>(), target_lengths_t.const_data_ptr<int64_t>(),
       neg_log_likelihood.const_data_ptr<scalar_t>(),
       grad.stride(0), grad.stride(1), grad.stride(2),
       log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
       log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
       log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
       tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
       batch_size, zero_infinity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else { // small problem, use naive algorithm
    // Still no block/grid configuration guru...
    int threads_input = max_threads;
    while (threads_input / 2 >= log_probs.size(0) && threads_input > 1) {
      threads_input /= 2;
    }
    threads_batch = std::min(max_threads / threads_input, (int) batch_size);
    dim3 block(threads_input, threads_batch);
    dim3 grid((log_probs.size(0) + threads_input-1)/threads_input, (batch_size+threads_batch-1)/threads_batch);
    ctc_loss_backward_collect_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>
      (grad.mutable_data_ptr<scalar_t>(),
       grad_out.const_data_ptr<scalar_t>(), grad_out.stride(0),
       log_alpha.const_data_ptr<scalar_t>(), log_beta.const_data_ptr<scalar_t>(),
       log_probs.const_data_ptr<scalar_t>(), input_lengths_t.const_data_ptr<int64_t>(), log_probs.size(0),
       targets.const_data_ptr<target_t>(), target_lengths_t.const_data_ptr<int64_t>(), max_target_length,
       neg_log_likelihood.const_data_ptr<scalar_t>(),
       grad.stride(0), grad.stride(1), grad.stride(2),
       log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
       log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
       log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
       tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
       batch_size, num_labels, BLANK, zero_infinity);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); // catch launch errors
  }

  // zero those invalid graident elements due to padding
  {
    int threads_input = max_threads;
    while (threads_input / 2 >= log_probs.size(0)) {
      threads_input /= 2;
    }
    threads_batch = std::min(max_threads / threads_input, (int) batch_size);
    dim3 block(threads_input, threads_batch);
    dim3 grid(
      (log_probs.size(0) + threads_input-1)/threads_input,
      (batch_size+threads_batch-1)/threads_batch);
    ctc_loss_zero_padded_gradients<scalar_t><<<grid, block, 0, stream>>>(
      grad.mutable_data_ptr<scalar_t>(),
      input_lengths_t.const_data_ptr<int64_t>(),
      grad.stride(0),
      grad.stride(1),
      grad.stride(2),
      grad.size(0),
      grad.size(1),
      grad.size(2)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return grad;
}

} // namespace

std::tuple<Tensor, Tensor> ctc_loss_gpu(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_cuda", [&] {
      if (targets.scalar_type() == kLong) {
        return ctc_loss_gpu_template<scalar_t, kLong>(log_probs, targets, input_lengths, target_lengths, BLANK);
      } else {
        return ctc_loss_gpu_template<scalar_t, kInt>(log_probs, targets, input_lengths, target_lengths, BLANK);
      }
    });
}

Tensor ctc_loss_backward_gpu(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                             const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("ctc_loss_backward_gpu");
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_backward_cuda", [&] {
      if (targets.scalar_type() == kLong) {
        return ctc_loss_backward_gpu_template<scalar_t, kLong>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
      } else {
        return ctc_loss_backward_gpu_template<scalar_t, kInt>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
      }
    });
}

} // at::native
