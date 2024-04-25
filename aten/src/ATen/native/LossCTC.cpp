// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license
// This is the CPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// We use the equations from above link, but note that [1] has 1-based indexing and we (of course) use 0-based.
// Graves et al call the probabilities y, we use log_probs (also calling them inputs)
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Fill.h>
#include <c10/util/irange.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_ctc_loss.h>
#include <ATen/ops/_ctc_loss_backward.h>
#include <ATen/ops/_ctc_loss_backward_native.h>
#include <ATen/ops/_ctc_loss_native.h>
#include <ATen/ops/_cudnn_ctc_loss.h>
#include <ATen/ops/_use_cudnn_ctc_loss.h>
#include <ATen/ops/ctc_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

#include <type_traits>
#include <utility>

namespace at::native {

namespace {

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1]) note that no bound-checking is done
template<typename target_t>
static inline int64_t get_target_prime(target_t* target, int64_t offset, int64_t stride, int64_t idx, int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor, size_t, std::vector<int64_t>> ctc_loss_allocate_outputs(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)

  CheckedFrom c = "ctc_loss_allocate_outputs";
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK((int64_t) input_lengths.size() == batch_size, "input_lengths must be of size batch_size");
  TORCH_CHECK((int64_t) target_lengths.size() == batch_size, "target_lengths must be of size batch_size");

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t tg_target_stride;
  int64_t max_target_length = 0;
  std::vector<int64_t> tg_batch_offsets(batch_size);
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (const auto i : c10::irange(batch_size)) {
      TORCH_CHECK(target_lengths[i] >= 0,
                  "Expected target_lengths to have value at least ", 0, ", but got value ", target_lengths[i],
                  " (while checking arguments for ", c, ")");
      tg_batch_offsets[i] = pos;
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
    for (const auto i : c10::irange(batch_size)) {
      TORCH_CHECK(target_lengths[i] >= 0,
                  "Expected target_lengths to have value at least ", 0, ", but got value ", target_lengths[i],
                  " (while checking arguments for ", c, ")");
      tg_batch_offsets[i] = i * tg_batch_stride;
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
  for (const auto b : c10::irange(batch_size)) {
    TORCH_CHECK(input_lengths[b] >= 0,
             "Expected input_lengths to have value at least ", 0, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
    TORCH_CHECK(input_lengths[b] <= max_input_length,
             "Expected input_lengths to have value at most ", max_input_length, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
  }

  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  return std::make_tuple(neg_log_likelihood, log_alpha, tg_target_stride, tg_batch_offsets);
}

// This kernel is a relatively straightforward implementation of the alpha calculation in the forward backward algorithm (section 4.1).
// A (minor) twist is that we are using log-calculations to enhance numerical stability (log_probs and log_alpha).
// The function returns the loss and the alphas, the alphas are kept for the backward step. The wrapper (ctc_loss below) hides
// the alphas from the user by only returning the loss.
template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> ctc_loss_cpu_template(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;

  Tensor neg_log_likelihood, log_alpha;
  size_t tg_target_stride;
  std::vector<int64_t> tg_batch_offsets;

  if (targets.scalar_type() == kLong) {
    std::tie(neg_log_likelihood, log_alpha,tg_target_stride, tg_batch_offsets) =
        ctc_loss_allocate_outputs<scalar_t, kLong>(
            log_probs, targets, input_lengths, target_lengths, BLANK);
  } else {
    std::tie(neg_log_likelihood, log_alpha, tg_target_stride, tg_batch_offsets) =
        ctc_loss_allocate_outputs<scalar_t, kInt>(
            log_probs, targets, input_lengths, target_lengths, BLANK);
  }

  int64_t batch_size = log_probs.size(1);
  auto lpp  = log_probs.permute({1,0,2});
  auto log_probs_a_global = lpp.accessor<const scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<scalar_t, 3>();
  auto targets_data = targets.const_data_ptr<target_t>();
  auto neg_log_likelihood_a = neg_log_likelihood.accessor<scalar_t, 1>();

  // alpha calculation for the first row, the three equations for alpha_1 above eq (6)
  // first the default
  log_alpha.narrow(1, 0, 1).fill_(neginf);
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      int64_t input_length = input_lengths[b];
      int64_t target_length = target_lengths[b];
      auto log_probs_a = log_probs_a_global[b];
      auto log_alpha_a = log_alpha_a_global[b];
      int64_t tg_batch_offset = tg_batch_offsets[b];

      if (input_length == 0) {
        scalar_t log_likelihood = target_length == 0 ? 0 : neginf;
        neg_log_likelihood_a[b] = -log_likelihood;
        continue;
      }

      // the first two items of alpha_t above eq (6)
      log_alpha_a[0][0] = log_probs_a[0][BLANK];
      if (target_length > 0)
        log_alpha_a[0][1] = log_probs_a[0][get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK)];

      // now the loop over the inputs
      for (const auto t : c10::irange(1, input_length)) {
        for (const auto s : c10::irange(2*target_length+1)) {
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          // this loop over s could be parallel/vectorized, too, but the required items are one index apart
          // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
          // for the cuda implementation, that gave a speed boost.
          // This is eq (6) and (7), la1,2,3 are the three summands. We keep track of the maximum for the logsumexp calculation.

          scalar_t la1 = log_alpha_a[t-1][s];
          scalar_t lamax = la1;
          scalar_t la2, la3;
          if (s > 0) {
            la2 = log_alpha_a[t-1][s-1];
            if (la2 > lamax)
              lamax = la2;
          } else {
            la2 = neginf;
          }
          if ((s > 1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s-2, BLANK) !=
                          current_target_prime)) {
            la3 = log_alpha_a[t-1][s-2];
            if (la3 > lamax)
              lamax = la3;
          } else {
            la3 = neginf;
          }
          if (lamax == neginf) // cannot do neginf-neginf
            lamax = 0;
          // this is the assignment of eq (6)
          log_alpha_a[t][s] = std::log(std::exp(la1-lamax)+std::exp(la2-lamax)+std::exp(la3-lamax))+lamax + log_probs_a[t][current_target_prime];
        }
      }
      // the likelihood is the sum of the last two alphas, eq (8), the loss is the negative log likelihood
      if (target_length == 0) {
        // if the target is empty then there is no preceding BLANK state and hence there is no path to merge
        neg_log_likelihood_a[b] = -log_alpha_a[input_length-1][0];
      } else {
        scalar_t l1 = log_alpha_a[input_length-1][target_length*2];
        scalar_t l2 = log_alpha_a[input_length-1][target_length*2-1];
        scalar_t m = std::max(l1, l2);
        m = ((m == neginf) ? 0 : m);
        scalar_t log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
        neg_log_likelihood_a[b] = -log_likelihood;
      }
    }
  });

  return std::make_tuple(neg_log_likelihood, log_alpha);
}

// This is the backward. It consists of two phases:
// a) computing the beta analogous to the alphas in the forward (backward half of the forward-backward algorithm) (eq (10) and (11))
// b) collecting the per-activation characters for all s and wrapping the gradient (eq (16), the collection is the sum)
template<typename scalar_t, ScalarType target_scalar_type>
Tensor ctc_loss_backward_cpu_template(const Tensor& grad_out, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                                      const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  int64_t max_input_length = log_probs.size(0);
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  Tensor grad = at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // at this point, this is log of empty sum

  // The admin bits. We don't do much checking and assume that the forward did.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t tg_target_stride;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t max_target_length;
  std::vector<int64_t> tg_batch_offsets(batch_size);

  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (const auto i : c10::irange(batch_size)) {
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (const auto i : c10::irange(batch_size)) {
      tg_batch_offsets[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length = targets.size(1);
  }

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // could be optimized to use only 2 rows
  auto lpp  = log_probs.permute({1,0,2});
  auto log_probs_a_global = lpp.accessor<const scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<const scalar_t, 3>();
  auto log_beta_a_global = log_beta.accessor<scalar_t, 3>();
  auto gp = grad.permute({1,0,2});
  auto grad_a_global = gp.accessor<scalar_t, 3>();
  auto targets_data = targets.const_data_ptr<target_t>();
  auto grad_out_a = grad_out.accessor<const scalar_t, 1>();

  auto create_fill_iterator = [](const Tensor& tensor, IntArrayRef squash_dims) {
    return TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
        .check_all_same_dtype(false)
        .add_output(tensor)
        .resize_outputs(false)
        .declare_static_shape(tensor.sizes(), squash_dims)
        .build();
  };
  const auto fill_iter = create_fill_iterator(grad, /*squash_dims=*/1);
  const auto fill_1d_iter = create_fill_iterator(grad, /*squash_dims=*/{0, 1});
  const auto fill_log_beta_1d_iter = create_fill_iterator(log_beta, /*squash_dims=*/{0, 1});

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    TensorIterator fill_iter_local(fill_iter);
    TensorIterator fill_1d_iter_local(fill_1d_iter);
    TensorIterator fill_log_beta_1d_iter_local(fill_log_beta_1d_iter);

    for (const auto b : c10::irange(start, end)) {
      scalar_t nll = neg_log_likelihood.accessor<scalar_t, 1>()[b];
      auto grad_a = grad_a_global[b];
      if (zero_infinity && nll == std::numeric_limits<scalar_t>::infinity()) {
        // grad_batch.zero_();
        fill_iter_local.unsafe_replace_operand(0, grad_a.data());
        fill_stub(kCPU, fill_iter_local, 0);
        continue;
      }

      auto log_probs_a = log_probs_a_global[b];
      auto log_alpha_a = log_alpha_a_global[b];
      auto log_beta_a = log_beta_a_global[b];
      int64_t input_length = input_lengths[b];
      int64_t target_length = target_lengths[b];
      int64_t tg_batch_offset = tg_batch_offsets[b];

      // the initialization of beta before eq (10)
      // here we do the fill for each batch item separately, as the input lengths will differ, so the t in which
      // we start varies
      if (input_length > 0) {
        // log_beta.select(0, b).select(1, input_length-1).fill_(neginf);
        fill_log_beta_1d_iter_local.unsafe_replace_operand(
            0, log_beta_a[input_length - 1].data());
        fill_stub(kCPU, fill_log_beta_1d_iter_local, neginf);

        log_beta_a[input_length-1][2*target_length] = log_probs_a[input_length-1][BLANK];
        grad_a[input_length-1][BLANK] = log_alpha_a[input_length-1][2*target_length] + log_beta_a[input_length-1][2*target_length];

        if (target_length > 0) {
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 2*target_length-1, BLANK);
          log_beta_a[input_length-1][2*target_length-1] = log_probs_a[input_length-1][current_target_prime];

          // the first two are a blank and a non-blank, so we know they are different and we don't need to do log+
          grad_a[input_length-1][current_target_prime] = log_alpha_a[input_length-1][2*target_length-1] + log_beta_a[input_length-1][2*target_length-1];
        }
      }

      // now loop applying eq (10) / (11)
      for (int64_t t=input_length-2; t>=0; t--) {
        // this loop over s could be parallel/vectorized and doesn't really need to be descending...
        // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
        // for the cuda implementation, that gave a speed boost.
        for (int64_t s=2*target_length; s>=0; s--) {
          scalar_t lb1 = log_beta_a[t+1][s];
          scalar_t lbmax = lb1;
          scalar_t lb2, lb3;
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          if (s < 2*target_length) {
            lb2 = log_beta_a[t+1][s+1];
            if (lb2 > lbmax)
              lbmax = lb2;
          } else {
            lb2 = neginf;
          }
          if ((s < 2*target_length-1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s+2, BLANK) !=
                                          current_target_prime)) {
            lb3 = log_beta_a[t+1][s+2];
            if (lb3 > lbmax)
              lbmax = lb3;
          } else {
            lb3 = neginf;
          }
          if (lbmax == neginf)
            lbmax = 0;

          log_beta_a[t][s] = std::log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax)+std::exp(lb3-lbmax))+lbmax + log_probs_a[t][current_target_prime];
          // one might check whether one can vectorize this better when done after the t-loop...
          // now that we have beta, we fill in the sum of alpha*beta in eq (16)
          // in contrast to the cuda implementation, we only parallelize over the batch, so we don't have a concurrency
          // issue (several s can map to the same target character)
          // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
          scalar_t log_alpha_beta =  log_alpha_a[t][s] + log_beta_a[t][s];
          scalar_t &lcab = grad_a[t][current_target_prime];
          if (lcab == neginf) {
            lcab = log_alpha_beta;
          } else {
            scalar_t max = std::max(lcab, log_alpha_beta);
            lcab = std::log(std::exp(lcab-max)+std::exp(log_alpha_beta-max))+max;
          }
        }
      }

      // now grad has the sum of eq (16)
      // now we wrap up the calculation by adding in the remaining items of eq (16)
      // this could be a great target for further vectorization.
      // grad is the output gradient, nll is the loss. Note that the likelihood -nll is the Z of eq (16)
      scalar_t gr = grad_out_a[b];
      for (const auto t : c10::irange(input_length)) { // or go for the full thing?
        for (const auto c : c10::irange(num_labels)) {
          scalar_t& res = grad_a[t][c];
          scalar_t lp = log_probs_a[t][c];
          res = (std::exp(lp)-std::exp(res + nll - lp)) * gr;
        }
      }

      // zero the remainder
      for (auto l : c10::irange(input_length, max_input_length)) {
        // grad_batch.select(0, l).zero_();
        fill_1d_iter_local.unsafe_replace_operand(0, grad_a[l].data());
        fill_stub(kCPU, fill_1d_iter_local, 0);
      }
    }
  });
  return grad;
}

} // namespace

std::tuple<Tensor, Tensor> ctc_loss_meta(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backwards
  return AT_DISPATCH_FLOATING_TYPES(
      log_probs.scalar_type(), "ctc_loss_meta", [&] {
        Tensor neg_log_likelihood, log_alpha;
        if (targets.scalar_type() == kLong) {
          std::tie(neg_log_likelihood, log_alpha, std::ignore, std::ignore) =  ctc_loss_allocate_outputs<scalar_t, kLong>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        } else {
          std::tie(neg_log_likelihood, log_alpha, std::ignore, std::ignore) = ctc_loss_allocate_outputs<scalar_t, kInt>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        }
        return std::make_tuple(neg_log_likelihood, log_alpha);
      });
}

std::tuple<Tensor, Tensor> ctc_loss_cpu(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backwards
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_cpu", [&] {
      if (targets.scalar_type() == kLong) {
        return ctc_loss_cpu_template<scalar_t, kLong>(log_probs, targets, input_lengths, target_lengths, BLANK);
      } else {
        return ctc_loss_cpu_template<scalar_t, kInt>(log_probs, targets, input_lengths, target_lengths, BLANK);
      }
  });
}


std::tuple<Tensor, Tensor> ctc_loss_tensor(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK, bool zero_infinity) {
  TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.const_data_ptr<int64_t>(), tlc.numel());

  return at::_ctc_loss(log_probs, targets, il, tl, BLANK, zero_infinity);
}

Tensor ctc_loss_backward_cpu(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                             const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_backward_cpu", [&] {
      if (targets.scalar_type() == kLong) {
        return ctc_loss_backward_cpu_template<scalar_t,kLong>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
      } else {
        return ctc_loss_backward_cpu_template<scalar_t,kInt>(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
      }
  });
}

Tensor ctc_loss_backward_tensor(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    const Tensor& input_lengths,
    const Tensor& target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {
  TORCH_CHECK(
      isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false),
      "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return at::_ctc_loss_backward(grad, log_probs, targets, il, tl, neg_log_likelihood, log_alpha, BLANK, zero_infinity);
}

namespace {

Tensor get_clamped_target_length(
    IntArrayRef target_lengths,
    const TensorOptions& options) {
  return at::tensor(target_lengths, options).clamp_min(1);
}

Tensor get_clamped_target_length(
    const Tensor & target_lengths,
    const TensorOptions& options) {
  return target_lengths.clamp_min(1);
}

// this wrapper function dispatches to the native and cudnn implementations and hides the alpha/grad from the user (by just returning the loss)
// the gradient is implemented for _cudnn_ctc_loss (just in derivatives.yaml) and _ctc_loss and this function has automatic gradients
// it also handles the reduction if desired
template <typename LengthsType>
Tensor ctc_loss_impl(const Tensor& log_probs_, const Tensor& targets, LengthsType input_lengths, LengthsType target_lengths, int64_t BLANK, int64_t reduction, bool zero_infinity) {
  auto is_batched = log_probs_.dim() == 3;
  Tensor log_probs = is_batched ? log_probs_ : log_probs_.unsqueeze(1);
  bool use_cudnn =
      (log_probs.device().type() == at::kCUDA) &&
      at::_use_cudnn_ctc_loss(
          log_probs, targets, input_lengths, target_lengths, BLANK);

  Tensor res;
  if (use_cudnn) {
    // non-deterministic ctc loss on cudnn disabled due to inconsistent results
    // see: https://github.com/pytorch/pytorch/issues/21680
    res = std::get<0>(at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, BLANK, /*deterministic=*/true, zero_infinity));
  } else {
    // if the targets are on CPU (which you need for CuDNN, let's move them to
    // GPU as a service for the user)
    res = std::get<0>(at::_ctc_loss(
        log_probs,
        targets.to(log_probs.device(), kLong),
        input_lengths,
        target_lengths,
        BLANK,
        zero_infinity));
    if (zero_infinity) {
      res = at::where(res == Scalar(std::numeric_limits<double>::infinity()), at::zeros({}, res.options()), res);
    }
  }
  if (reduction == at::Reduction::Mean) {
    auto target_lengths_t = get_clamped_target_length(target_lengths, res.options());
    return (res / target_lengths_t).mean();
  } else if (reduction == at::Reduction::Sum) {
    return res.sum();
  }
  return is_batched ? std::move(res) : res.squeeze(0);
}

} // namespace

Tensor ctc_loss(const Tensor& log_probs_, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, int64_t reduction, bool zero_infinity) {
  return ctc_loss_impl(log_probs_, targets, input_lengths, target_lengths, BLANK, reduction, zero_infinity);
}

// Convenience function accepting Tensors
Tensor ctc_loss(const Tensor& log_probs, const Tensor& targets, const Tensor& input_lengths, const Tensor& target_lengths, int64_t BLANK, int64_t reduction, bool zero_infinity) {
  if (at::areAnyTensorSubclassLike(
          {log_probs, targets, input_lengths, target_lengths})) {
    // Composite Compliant path for TensorSubclasses
    return ctc_loss_impl(log_probs, targets, input_lengths, target_lengths, BLANK, reduction, zero_infinity);
  }

  // Fast path (which accesses data_ptr) and less operator dispatches for
  // regular tensors
  TORCH_CHECK(isIntegralType(input_lengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(target_lengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.const_data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.const_data_ptr<int64_t>(), tlc.numel());
  return at::native::ctc_loss(log_probs, targets, il, tl, BLANK, reduction, zero_infinity);
}

} // at::native
