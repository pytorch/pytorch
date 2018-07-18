// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license

#include <ATen/ATen.h>
#include "ATen/Dispatch.h"
#include "ATen/TensorUtils.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include <numeric>

namespace at {
namespace native {

namespace {

// __restrict__ impact to be measured, https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
__device__ inline int64_t get_target_prime(const int64_t* __restrict__ target, int64_t offset, int64_t stride, int64_t idx, int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

template<typename scalar_t>
__global__ void ctc_loss_gpu_kernel(scalar_t* __restrict__ log_alpha_data,
                                    const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths, int64_t max_input_length,
                                    const int64_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths, int64_t max_target_length, 
                                    scalar_t* __restrict__ neg_log_likelihood_data,
                                    int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                    int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
                                    const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                    int64_t batch_size, int64_t BLANK) {

  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t la_batch_offset = b*la_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (b >= batch_size)
    return;

  // first row (t=0)
  for (int64_t block_s = 0; block_s < 2*max_target_length+1; block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t la;
    switch (s) {
    case 0:
      la = log_probs_data[lp_batch_offset + lp_char_stride * BLANK];
      break;
    case 1:
      if (target_length > 0) {
        la = log_probs_data[lp_batch_offset + lp_char_stride * get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK)];
      }
      else {
        la = neginf;
      }
      break;
    default:
      la = neginf;
    }
    if (s < 2*max_target_length+1)
      log_alpha_data[la_batch_offset + /* la_input_stride * 0 */ + la_target_stride * s] = la;
  }

  for (int64_t t=1; t < max_input_length; t++) {
    __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch
    for (int64_t block_s = 0; block_s < 2*max_target_length+1; block_s += blockDim.x) { // maybe it is better to do consecutive targets...
      int64_t s = threadIdx.x + block_s;
      if ((t < input_length) && (target_length > 0) && (s < 2*target_length+1)) {
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
        if ((s > 1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s-2, BLANK) != // should be cached? (only depends on s)
                        get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK))) {
          la3 = log_alpha_data[la_batch_offset + la_input_stride * (t-1) + la_target_stride * (s-2)];
          if (la3 > lamax)
            lamax = la3;
        } else {
          la3 = neginf;
        }
        if (lamax == neginf)
          lamax = 0;

        log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = std::log(std::exp(la1-lamax)+std::exp(la2-lamax)+std::exp(la3-lamax))+lamax
          + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK)];
      } else {
        if (s < 2*max_target_length+1)
          log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = neginf;
      }
    }
  }
  __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch

  if (threadIdx.x == 0) {
    scalar_t l1 = log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * (target_length*2)];
    scalar_t l2 = log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * (target_length*2-1)];
    scalar_t m = ((l1 > l2) ? l1 : l2);
    m = ((m == neginf) ? 0 : m);
    scalar_t log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
    neg_log_likelihood_data[b] = -log_likelihood;
  }
}

template<typename scalar_t>
std::tuple<Tensor, Tensor> ctc_loss_gpu_template(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)

  CheckedFrom c = "ctc_loss_gpu";
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  auto input_lengths_arg = TensorArg(input_lengths, "input_lengths", 3);
  auto target_lengths_arg = TensorArg(target_lengths, "target_lengths", 4);
  checkAllSameGPU(c, {log_probs_arg, targets_arg, input_lengths_arg, target_lengths_arg});
  checkScalarType(c, targets_arg, kLong);
  checkScalarType(c, input_lengths_arg, kLong);
  checkScalarType(c, target_lengths_arg, kLong);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  AT_CHECK(BLANK < num_labels, "blank must be in label range");
  checkSize(c, input_lengths_arg, {batch_size});
  checkSize(c, target_lengths_arg, {batch_size});

  size_t lp_input_stride = log_probs.stride(0);
  size_t lp_char_stride = log_probs.stride(2);
  size_t tg_target_stride;

  int64_t max_target_length;
  Tensor tg_batch_offsets;
  if (targets.dim() == 1) { // concatenated targets
    tg_batch_offsets = at::zeros_like(target_lengths);
    auto tmp = tg_batch_offsets.narrow(0, 1, batch_size-1);
    at::cumsum_out(tmp, target_lengths, 0);
    max_target_length = at::max(target_lengths).toCLong();
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, tmp[batch_size-1].toCLong()+target_lengths[batch_size-1].toCLong());
  }
  else { // batch x max_target_length
    // dim is 2
    tg_batch_offsets = at::arange(0, targets.stride(0)*batch_size, targets.stride(0), target_lengths.options());
    tg_target_stride = targets.stride(1);
    max_target_length = targets.size(1);
    checkSize(c, targets_arg, 0, batch_size);
    AT_CHECK(targets.size(1) >= max_target_length,
             "Expected tensor to have size at least ", max_target_length, " at dimension 1, but got size ", targets.size(1), " for ", targets_arg,
             " (while checking arguments for ", c, ")");
  }
  int max_input_length = at::max(input_lengths).toCLong();
  AT_CHECK(log_probs.size(0) >= max_input_length,
           "Expected tensor to have size at least ", max_input_length, " at dimension 1, but got size ", targets.size(0), " for ", targets_arg,
           " (while checking arguments for ", c, ")");

  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  // tile range with optimal number of threads
  // I think something could be learned (or genralized and reused) from SoftMax.cu...
  constexpr int max_threads = 1024; 
  int threads_target = max_threads;
  while (threads_target / 2 >= 2*max_target_length+1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int) batch_size);
  dim3 block(threads_target, threads_batch);
  dim3 grid((2*max_target_length+1 + threads_target-1)/threads_target, (batch_size+threads_batch-1)/threads_batch);

  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  ctc_loss_gpu_kernel<scalar_t><<<grid, block, 0, stream>>>(log_alpha.data<scalar_t>(),
                      log_probs.data<scalar_t>(), input_lengths.data<int64_t>(), log_probs.size(0),
                      targets.data<int64_t>(), target_lengths.data<int64_t>(), max_target_length,
                      neg_log_likelihood.data<scalar_t>(),
                      log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
                      log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
                      tg_batch_offsets.data<int64_t>(), tg_target_stride,
                      batch_size, BLANK);


  return std::make_tuple(neg_log_likelihood, log_alpha);
}

template<typename scalar_t>
__global__ void ctc_loss_backward_betas_gpu_kernel(scalar_t* __restrict__ log_beta_data,
                                             const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths, int64_t max_input_length,
                                             const int64_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths, int64_t max_target_length, 
                                             int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                             int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
                                             const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                             int64_t batch_size, int64_t BLANK) {
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();

  int64_t b = threadIdx.y + blockIdx.y * blockDim.y;

  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  int64_t lp_batch_offset = b*lp_batch_stride;
  int64_t lb_batch_offset = b*lb_batch_stride;
  int64_t tg_batch_offset = tg_batch_offsets[b];

  if (b >= batch_size)
    return;

  // first row (t=0)
  for (int64_t block_s = 0; block_s < 2*max_target_length+1; block_s += blockDim.x) {
    int64_t s = threadIdx.x + block_s;
    scalar_t lb;
    if (s == 2*target_length) {
      lb = log_probs_data[lp_batch_offset + (input_length-1) * lp_input_stride + lp_char_stride * BLANK];
    } else if ((target_length > 0) && (s == 2*target_length-1)) {
      int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
      lb = log_probs_data[lp_batch_offset + (input_length-1) * lp_input_stride + lp_char_stride * current_target_prime];
    } else {
      lb = neginf;
    }
    if (s < 2*max_target_length+1) {
      log_beta_data[lb_batch_offset + (input_length-1) * lb_input_stride + lb_target_stride * s] = lb;
    }
  }

  for (int64_t t=max_input_length-2; t>=0; t--) {
    __syncthreads(); // on cuda 9 we might use partial synchronization of only the threads within the same batch
    for (int64_t block_s = 0; block_s < 2*max_target_length+1; block_s += blockDim.x) { // maybe it is better to do consecutive targets or alternatively try to change s and t (but has more syncthreads...
      int64_t s = threadIdx.x + block_s;
      if ((t < input_length-1) && (target_length > 0) && (s < 2*target_length+1)) {
        scalar_t lb1 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * s];
        scalar_t lbmax = lb1;
        scalar_t lb2, lb3;
        int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK); // maybe cache? (only depends on s)

        if (s < 2*target_length) {
          lb2 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * (s+1)];
          if (lb2 > lbmax)
            lbmax = lb2;
        } else {
          lb2 = neginf;
        }
        if ((s < 2*target_length-1) &&
            (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s+2, BLANK) != current_target_prime)) {
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
      } else if ((s < 2*max_target_length+1) || (t >= input_length)) {
          log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s] = neginf;
      }
    }
  }
}

template<typename scalar_t>
__global__ void ctc_loss_backward_collect_gpu_kernel(scalar_t* __restrict__ gradient_data,
                                                     const scalar_t* __restrict__ grad_out_data, int64_t grad_out_batch_stride,
                                                     const scalar_t* __restrict__ log_alpha_data, const scalar_t* __restrict__ log_beta_data,
                                                     const scalar_t*log_probs_data, const int64_t* __restrict__ input_lengths, int64_t max_input_length,
                                                     const int64_t* __restrict__ targets_data, const int64_t* __restrict__ target_lengths, int64_t max_target_length,
                                                     const scalar_t* __restrict__ neg_log_likelihood_data,
                                                     int64_t gr_input_stride, int64_t gr_batch_stride, int64_t gr_char_stride,
                                                     int64_t lp_input_stride, int64_t lp_batch_stride, int64_t lp_char_stride,
                                                     int64_t la_batch_stride, int64_t la_input_stride, int64_t la_target_stride,
                                                     int64_t lb_batch_stride, int64_t lb_input_stride, int64_t lb_target_stride,
                                                     const int64_t* __restrict__ tg_batch_offsets, int64_t tg_target_stride,
                                                     int64_t batch_size, int64_t num_labels, int64_t BLANK) {

  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
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
    if ((target_length > 0) && (s < 2*target_length+1)) {
      int64_t current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK); // maybe cache? (only depends on s)
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
    if (t < input_length) {
      scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * c];
      res = std::exp(lp)-std::exp(res + nll - lp) * gr;
    }
    else {
      res = 0.;
    }
  }
}

template<typename scalar_t>
Tensor ctc_loss_backward_gpu_template(const Tensor& grad_out, const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths,
                                      const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK) {
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  size_t lp_input_stride = log_probs.stride(0);
  size_t lp_char_stride = log_probs.stride(2);
  size_t tg_target_stride;

  int64_t max_target_length;
  Tensor tg_batch_offsets;
  if (targets.dim() == 1) { // concatenated targets
    tg_batch_offsets = at::zeros_like(target_lengths);
    auto tmp = tg_batch_offsets.narrow(0, 1, batch_size-1);
    at::cumsum_out(tmp, target_lengths, 0);
    max_target_length = at::max(target_lengths).toCLong();
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim shoudl be 2
    tg_batch_offsets = at::arange(0, targets.stride(0)*batch_size, targets.stride(0), target_lengths.options());
    tg_target_stride = targets.stride(1);
    max_target_length = targets.size(1);
  }

  Tensor log_beta = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
  Tensor grad = at::full_like(log_probs, neginf); // initialization for log(sum (alpha beta)) 

  // tile range with optimal number of threads
  // I think something could be learned (or genralized and reused) from SoftMax.cu...
  constexpr int max_threads = 1024; 
  int threads_target = max_threads;
  while (threads_target / 2 >= 2*max_target_length+1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int) batch_size);

  cudaStream_t stream = globalContext().getCurrentCUDAStream();

  {
  dim3 block(threads_target, threads_batch);
  dim3 grid((2*max_target_length+1 + threads_target-1)/threads_target, (batch_size+threads_batch-1)/threads_batch);

  ctc_loss_backward_betas_gpu_kernel<scalar_t><<<grid, block, 0, stream>>>(log_beta.data<scalar_t>(),
                      log_probs.data<scalar_t>(), input_lengths.data<int64_t>(), log_probs.size(0),
                      targets.data<int64_t>(), target_lengths.data<int64_t>(), max_target_length,
                      log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
                      log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
                      tg_batch_offsets.data<int64_t>(), tg_target_stride,
                      batch_size, BLANK);
  }

  {
  // better grid, block
  int threads_input = max_threads;
  while (threads_input / 2 >= log_probs.size(0)) {
    threads_input /= 2;
  }
  threads_batch = std::min(max_threads / threads_input, (int) batch_size);
  dim3 block(threads_input, threads_batch);
  dim3 grid((log_probs.size(0) + threads_input-1)/threads_input, (batch_size+threads_batch-1)/threads_batch);

  ctc_loss_backward_collect_gpu_kernel<scalar_t><<<grid, block, 0, stream>>>(
                     grad.data<scalar_t>(),
                     grad_out.data<scalar_t>(), grad_out.stride(0),
                     log_alpha.data<scalar_t>(), log_beta.data<scalar_t>(),
                     log_probs.data<scalar_t>(), input_lengths.data<int64_t>(), log_probs.size(0),
                     targets.data<int64_t>(), target_lengths.data<int64_t>(), max_target_length,
                     neg_log_likelihood.data<scalar_t>(),
                     grad.stride(0), grad.stride(1), grad.stride(2),
                     log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
                     log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2),
                     log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
                     tg_batch_offsets.data<int64_t>(), tg_target_stride,
                     batch_size, num_labels, BLANK);
  }
  return grad;
}

} // namespace

std::tuple<Tensor, Tensor> ctc_loss_gpu(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK) {
  return AT_DISPATCH_FLOATING_TYPES(log_probs.type(), "ctc_loss", [&] {
    return ctc_loss_gpu_template<scalar_t>(log_probs, targets, input_lengths, target_lengths, BLANK);
  });
}

Tensor ctc_loss_backward_gpu(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths,
                             const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK) {
  return AT_DISPATCH_FLOATING_TYPES(log_probs.type(), "ctc_loss_backward", [&] {
      return ctc_loss_backward_gpu_template<scalar_t>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK);
  });
}

} } // at::native
