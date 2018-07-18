// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license

#include <ATen/ATen.h>
#include "ATen/Dispatch.h"
#include "ATen/TensorUtils.h"

#include <numeric>

namespace at {
namespace native {

namespace {

inline int64_t get_target_prime(int64_t* target, int64_t offset, int64_t stride, int64_t idx, int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

template<typename scalar_t>
std::tuple<Tensor, Tensor> ctc_loss_cpu_template(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();

  CheckedFrom c = "ctc_loss_cpu";
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  auto input_lengths_arg = TensorArg(input_lengths, "input_lengths", 3);
  auto target_lengths_arg = TensorArg(target_lengths, "target_lengths", 4);
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


  auto log_probs_data = log_probs.data<scalar_t>();
  auto log_alpha_data = log_alpha.data<scalar_t>();
  auto targets_data = targets.data<int64_t>();
  auto neg_log_likelihood_data = neg_log_likelihood.data<scalar_t>(); // we assume stride one for the only dimension for freshly allocated tensor
  size_t lp_input_stride = log_probs.stride(0);
  size_t lp_char_stride = log_probs.stride(2);
  size_t la_input_stride = log_alpha.stride(1);
  size_t la_target_stride = log_alpha.stride(2);

  log_alpha.narrow(1, 0, 1).fill_(neginf); // or do this inside the batch loop?
  #pragma omp parallel for
  for (int64_t b = 0; b < batch_size; b++) {
    int64_t input_length = input_lengths[b].toCLong();
    int64_t target_length = target_lengths[b].toCLong();
    int64_t lp_batch_offset = b*log_probs.stride(1);
    int64_t la_batch_offset = b*log_alpha.stride(0);
    int64_t tg_batch_offset = tg_batch_offsets[b].toCLong();
    log_alpha_data[la_batch_offset /* + 0 * la_input_stride + 0 * la_target_stride */ ] =
      log_probs_data[lp_batch_offset + /* 0 * lp_input_stride */ + lp_char_stride * BLANK];
    if (target_length > 0)
      log_alpha_data[la_batch_offset + la_target_stride * 1 /* + 0 * la_input_stride */ ] =
        log_probs_data[lp_batch_offset + /* 0 * lp_input_stride */ + lp_char_stride * get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK)];

    for (int64_t t=1; t<input_length; t++) {
      // this loop over s could be parallel/vectorized
      for (int64_t s=0; s<2*target_length+1; s++) {
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
        if ((s > 1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s-2, BLANK) !=
                        get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK))) {
          la3 = log_alpha_data[la_batch_offset + la_input_stride * (t-1) + la_target_stride * (s-2)];
          if (la3 > lamax)
            lamax = la3;
        } else {
          la3 = neginf;
        }
        if (lamax == neginf) // cannot do neginf-neginf
          lamax = 0;

        log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s] = std::log(std::exp(la1-lamax)+std::exp(la2-lamax)+std::exp(la3-lamax))+lamax
                   + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK)];
      }
    }
    scalar_t l1 = log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * (target_length*2)];
    scalar_t l2 = log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * (target_length*2-1)];
    scalar_t m = std::max(l1, l2);
    m = ((m == neginf) ? 0 : m);
    scalar_t log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
    neg_log_likelihood_data[b] = -log_likelihood;
  }

  return std::make_tuple(neg_log_likelihood, log_alpha);
}


template<typename scalar_t>
Tensor ctc_loss_backward_cpu_template(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths,
                                      const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK) {
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  Tensor log_collected_alpha_beta = at::full_like(log_probs, neginf);

  size_t tg_target_stride;
  Tensor tg_batch_offsets;

  if (targets.dim() == 1) { // concatenated targets
    tg_batch_offsets = at::zeros_like(target_lengths);
    auto tmp = tg_batch_offsets.narrow(0, 1, batch_size-1);
    at::cumsum_out(tmp, target_lengths, 0);
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim is 2
    tg_batch_offsets = at::arange(0, targets.stride(0)*batch_size, targets.stride(0), target_lengths.options());
    tg_target_stride = targets.stride(1);
  }

  Tensor log_beta = at::empty_like(log_alpha);  // could be optimized to use only 2 rows
  auto log_probs_data = log_probs.data<scalar_t>();
  auto log_alpha_data = log_alpha.data<scalar_t>();
  auto log_beta_data = log_beta.data<scalar_t>();
  auto targets_data = targets.data<int64_t>();
  auto log_collected_alpha_beta_data = log_collected_alpha_beta.data<scalar_t>();
  size_t lp_input_stride = log_probs.stride(0);
  size_t lp_char_stride = log_probs.stride(2);
  size_t la_input_stride = log_alpha.stride(1);
  size_t la_target_stride = log_alpha.stride(2);
  size_t lb_input_stride = log_beta.stride(1);
  size_t lb_target_stride = log_beta.stride(2);
  size_t ab_input_stride = log_collected_alpha_beta.stride(0);
  size_t ab_char_stride = log_collected_alpha_beta.stride(2);

  #pragma omp parallel for
  for (int64_t b = 0; b < batch_size; b++) {
    int64_t input_length = input_lengths[b].toCLong();
    int64_t target_length = target_lengths[b].toCLong();
    int64_t lp_batch_offset = b*log_probs.stride(1);
    int64_t la_batch_offset = b*log_alpha.stride(0);
    int64_t lb_batch_offset = b*log_beta.stride(0);
    int64_t ab_batch_offset = b*log_collected_alpha_beta.stride(1);
    int64_t tg_batch_offset = tg_batch_offsets[b].toCLong();

    log_beta.narrow(0, b, 1).narrow(1, input_length-1, 1).fill_(neginf);
    if (input_length > 0) {
      log_beta_data[lb_batch_offset + (input_length-1) * lb_input_stride + 2*target_length * lb_target_stride] =
        log_probs_data[lp_batch_offset + (input_length-1) * lp_input_stride + lp_char_stride * BLANK];

      log_collected_alpha_beta_data[ab_batch_offset + (input_length-1) * ab_input_stride + ab_char_stride * BLANK] =
         log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * 2*target_length]
        + log_beta_data[lb_batch_offset + lb_input_stride * (input_length-1) + lb_target_stride * 2*target_length];

      if (target_length > 0) {
        auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 2*target_length-1, BLANK);
        log_beta_data[la_batch_offset + (input_length-1) * lb_input_stride + (2*target_length-1) * lb_target_stride] =
          log_probs_data[lp_batch_offset + (input_length-1) * lp_input_stride + lp_char_stride * current_target_prime];

        // the first two are a blank and a non-blank, so we know they are different and we don't need to do log+
        log_collected_alpha_beta_data[ab_batch_offset + (input_length-1) * ab_input_stride + ab_char_stride * current_target_prime] =
          log_alpha_data[la_batch_offset + la_input_stride * (input_length-1) + la_target_stride * (2*target_length-1)]
          + log_beta_data[lb_batch_offset + lb_input_stride * (input_length-1) + lb_target_stride * (2*target_length-1)];
      }
    }

    for (int64_t t=input_length-2; t>=0; t--) {
      // this loop over s could be parallel/vectorized and doesn't really need to be descending...
      for (int64_t s=2*target_length; s>=0; s--) {
        scalar_t lb1 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * s];
        scalar_t lbmax = lb1;
        scalar_t lb2, lb3;
        auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
        if (s < 2*target_length) {
          lb2 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * (s+1)];
          if (lb2 > lbmax)
            lbmax = lb2;
        } else {
          lb2 = neginf;
        }
        if ((s < 2*target_length-1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s+2, BLANK) !=
                                        current_target_prime)) {
          lb3 = log_beta_data[lb_batch_offset + lb_input_stride * (t+1) + lb_target_stride * (s+2)];
          if (lb3 > lbmax)
            lbmax = lb3;
        } else {
          lb3 = neginf;
        }
        if (lbmax == neginf)
          lbmax = 0;

        log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s] = std::log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax)+std::exp(lb3-lbmax))+lbmax
                   + log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * current_target_prime];

        // one might check whether one can vectorize this better when done after the t-loop...
        // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
        scalar_t log_alpha_beta =  log_alpha_data[la_batch_offset + la_input_stride * t + la_target_stride * s]
                                  + log_beta_data[lb_batch_offset + lb_input_stride * t + lb_target_stride * s];

        scalar_t& lcab = log_collected_alpha_beta_data[ab_batch_offset + t * ab_input_stride + ab_char_stride * current_target_prime];

        if (lcab == neginf) {
          lcab = log_alpha_beta;
        } else {
          scalar_t max = std::max(lcab, log_alpha_beta);
          lcab = std::log(std::exp(lcab-max)+std::exp(log_alpha_beta-max))+max;
        }
      }
    }

    // this could be a great target for further vectorization
    scalar_t nll = *neg_log_likelihood[b].data<scalar_t>();
    scalar_t gr =  *grad[b].data<scalar_t>();
    for (int64_t t = 0; t < input_length; t++) { // or go for the full thing?
      for (int64_t c = 0; c < num_labels; c++) {
        scalar_t& res = log_collected_alpha_beta_data[ab_batch_offset + t * ab_input_stride + ab_char_stride * c];
        scalar_t lp = log_probs_data[lp_batch_offset + t * lp_input_stride + lp_char_stride * c];
        res = std::exp(lp)-std::exp(res + nll - lp) * gr;
      }
      // should we zero the remaining grad?
    }

  }
  return log_collected_alpha_beta;
}

} // namespace

std::tuple<Tensor, Tensor> ctc_loss_cpu(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK) {
  return AT_DISPATCH_FLOATING_TYPES(log_probs.type(), "ctc_loss", [&] {
    return ctc_loss_cpu_template<scalar_t>(log_probs, targets, input_lengths, target_lengths, BLANK);
  });
}

Tensor ctc_loss_backward_cpu(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths,
                             const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK) {
  return AT_DISPATCH_FLOATING_TYPES(log_probs.type(), "ctc_loss_backward", [&] {
      return ctc_loss_backward_cpu_template<scalar_t>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK);
  });
}

Tensor ctc_loss(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK, int64_t reduction) {
  auto& ctx = at::globalContext();

  bool use_cudnn =
    detail::getCUDAHooks().compiledWithCuDNN() &&
    (detail::getCUDAHooks().versionCuDNN() >= 7000) &&
    ctx.userEnabledCuDNN() &&
    (BLANK == 0) && (targets.dim()==1) &&
    (log_probs.type().scalarType() == at::kFloat) &&
    (log_probs.type().backend() == Backend::CUDA);

  if (use_cudnn) {
    use_cudnn = (target_lengths.max().toCLong() <= 256)
      && ((input_lengths == log_probs.size(0)).all().toCByte());
  }

  Tensor res;
  if (use_cudnn) {
    res = std::get<0>(at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, BLANK, ctx.deterministicCuDNN()));
  } else {
    res = std::get<0>(at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, BLANK));
  }
  if (reduction == Reduction::ElementwiseMean) {
    return (res / target_lengths.toType(res.type())).mean();
  } else if (reduction == Reduction::Sum) {
    return res.sum();
  }
  return res;
}

} } // at::native
