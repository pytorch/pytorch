#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/Exception.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/binary_cross_entropy_backward_native.h>
#include <ATen/ops/binary_cross_entropy_native.h>
#include <ATen/ops/binary_cross_entropy_with_logits_native.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/cosine_embedding_loss_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/hinge_embedding_loss_native.h>
#include <ATen/ops/huber_loss_backward.h>
#include <ATen/ops/huber_loss_backward_native.h>
#include <ATen/ops/huber_loss_native.h>
#include <ATen/ops/kl_div_native.h>
#include <ATen/ops/l1_loss_native.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log_sigmoid.h>
#include <ATen/ops/margin_ranking_loss_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/min.h>
#include <ATen/ops/mse_loss_backward.h>
#include <ATen/ops/mse_loss_backward_native.h>
#include <ATen/ops/mse_loss_meta.h>
#include <ATen/ops/mse_loss_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/pairwise_distance.h>
#include <ATen/ops/poisson_nll_loss_native.h>
#include <ATen/ops/smooth_l1_loss_backward.h>
#include <ATen/ops/smooth_l1_loss_backward_native.h>
#include <ATen/ops/smooth_l1_loss_meta.h>
#include <ATen/ops/smooth_l1_loss_native.h>
#include <ATen/ops/soft_margin_loss.h>
#include <ATen/ops/soft_margin_loss_backward.h>
#include <ATen/ops/soft_margin_loss_backward_native.h>
#include <ATen/ops/soft_margin_loss_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/triplet_margin_loss_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/zeros_like.h>
#endif

constexpr float EPSILON = 1e-12;

namespace {
  static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
    if (reduction == at::Reduction::Mean) {
      return unreduced.mean();
    } else if (reduction == at::Reduction::Sum) {
      return unreduced.sum();
    }
    return unreduced;
  }
}

namespace at::meta {

TORCH_META_FUNC(smooth_l1_loss)
(const Tensor& input, const Tensor& target, const int64_t reduction, double beta) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  // TODO: Reduce this extra TensorIterator construction for Reduction::Mean & Sum.
  // We do another TensorIterator construction in the IMPL for the two cases.
  build_borrowing_binary_op(maybe_get_output(), input, target);
  if (reduction == Reduction::None) {
    return;
  }

  TORCH_INTERNAL_ASSERT(reduction == Reduction::Mean || reduction == Reduction::Sum);
  maybe_get_output().resize_({});
}

TORCH_META_FUNC(mse_loss)
(const Tensor& input, const Tensor& target, const int64_t reduction) {
  build_borrowing_binary_op(maybe_get_output(), input, target);
  if (reduction == Reduction::None) {
    return;
  }

  TORCH_INTERNAL_ASSERT(reduction == Reduction::Mean || reduction == Reduction::Sum);
  maybe_get_output().resize_({});
}

} // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(smooth_l1_stub);
DEFINE_DISPATCH(smooth_l1_backward_stub);
DEFINE_DISPATCH(huber_stub);
DEFINE_DISPATCH(huber_backward_stub);
DEFINE_DISPATCH(mse_stub);
DEFINE_DISPATCH(mse_backward_stub);

TORCH_IMPL_FUNC(smooth_l1_loss_out)
(const Tensor& input, const Tensor& target, int64_t reduction, double beta, const Tensor& result) {
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
    smooth_l1_stub(iter.device_type(), iter, beta);
    if (reduction == Reduction::Mean) {
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    smooth_l1_stub(device_type(), *this, beta);
  }
}

TORCH_IMPL_FUNC(mse_loss_out)
(const Tensor& input, const Tensor& target, int64_t reduction, const Tensor& result) {
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
    mse_stub(iter.device_type(), iter);
    if (reduction == Reduction::Mean) {
      at::mean_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    } else {
      at::sum_out(const_cast<Tensor&>(result), iter.output(), IntArrayRef{});
    }
  } else {
    mse_stub(device_type(), *this);
  }
}

Tensor cosine_embedding_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  auto targ_dim = target.dim();
  TORCH_CHECK(
      targ_dim == 1 || targ_dim == 0,
      "0D or 1D target tensor expected, multi-target not supported");
  if (targ_dim == 1) {
    TORCH_CHECK(
        input1.dim() == 2 && input2.dim() == 2,
        "1D target tensor expects 2D input tensors, but found inputs with sizes ",
        input1.sizes(),
        " and ",
        input2.sizes(),
        ".");
  } else {
    TORCH_CHECK(
        input1.dim() == 1 && input2.dim() == 1,
        "0D target tensor expects 1D input tensors, but found inputs with sizes ",
        input1.sizes(),
        " and ",
        input2.sizes(),
        ".");
  }

  auto prod_sum = (input1 * input2).sum(targ_dim);
  auto mag_square1 = (input1 * input1).sum(targ_dim) + EPSILON;
  auto mag_square2 = (input2 * input2).sum(targ_dim) + EPSILON;
  auto denom = (mag_square1 * mag_square2).sqrt_();
  auto cos = prod_sum / denom;

  auto zeros = at::zeros_like(cos, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto pos = 1 - cos;
  auto neg = (cos - margin).clamp_min_(0);
  auto output_pos = at::where(target == 1, pos, zeros);
  auto output_neg = at::where(target == -1, neg, zeros);
  auto output = output_pos + output_neg;
  return apply_loss_reduction(output, reduction);
}

Tensor hinge_embedding_loss(const Tensor& self, const Tensor& target, double margin, int64_t reduction) {
  auto zeros = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto margin_diff = (margin - self);
  // For Composite Compliance,
  // In Forward AD, if `margin_diff` is a CCT but its tangent isn't,
  // using inplace clamp_min doesn't work because we end up writing
  // the CCT in-place to the tangent
  auto margin_clamp = (margin_diff._fw_grad(/*level*/ 0).defined() &&
                       isTensorSubclassLike(margin_diff))
      ? margin_diff.clamp_min(0)
      : margin_diff.clamp_min_(0);
  auto output_margin = at::where(target != 1, margin_clamp, zeros);
  auto output_self = at::where(target != -1, self, zeros);
  auto output = output_margin + output_self;
  return apply_loss_reduction(output, reduction);
}

Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin,
                           double p, double eps, bool swap, int64_t reduction) {
  auto a_dim = anchor.dim();
  auto p_dim = positive.dim();
  auto n_dim = negative.dim();
  TORCH_CHECK(
      a_dim == p_dim && p_dim == n_dim,
      "The anchor, positive, and negative tensors are expected to have "
      "the same number of dimensions, but got: anchor ", a_dim, "D, "
      "positive ", p_dim, "D, and negative ", n_dim, "D inputs")

  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
  // The distance swap is described in the paper "Learning shallow
  // convolutional feature descriptors with triplet losses" by V. Balntas, E.
  // Riba et al.  If True, and if the positive example is closer to the
  // negative example than the anchor is, swaps the positive example and the
  // anchor in the loss computation.
  if (swap) {
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    dist_neg = at::min(dist_neg, dist_swap);
  }
  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  return apply_loss_reduction(output, reduction);
}

Tensor margin_ranking_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  auto unclamped_output = (-target * (input1 - input2) + margin);
  // For Composite Compliance,
  // In Forward AD, if `margin_diff` is a CCT but its tangent isn't,
  // using inplace clamp_min doesn't work because we end up writing
  // the CCT in-place to the tangent
  auto output = (unclamped_output._fw_grad(/*level*/ 0).defined() &&
                 isTensorSubclassLike(unclamped_output))
      ? unclamped_output.clamp_min(0)
      : unclamped_output.clamp_min_(0);
  return apply_loss_reduction(output, reduction);
}

Tensor kl_div(const Tensor& input, const Tensor& target, int64_t reduction, bool log_target) {
  TORCH_CHECK(!input.is_complex() && !target.is_complex(),
              "kl_div: Complex inputs not supported.");
  TORCH_CHECK(!at::isIntegralType(input.scalar_type(), /*include_bool*/true) &&
              !at::isIntegralType(target.scalar_type(), /*include_bool*/true),
              "kl_div: Integral inputs not supported.");
  Tensor output;
  if (log_target) {
    output = at::exp(target) * (target - input);
  } else {
    output = at::xlogy(target, target) - target * input;
  }
  return apply_loss_reduction(output, reduction);
}

Tensor binary_cross_entropy_cpu(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction) {
    Tensor loss = at::empty_like(input);
    return at::native::binary_cross_entropy_out_cpu(
        input, target, weight_opt, reduction, loss);
}

Tensor& binary_cross_entropy_out_cpu(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, Tensor& loss) {
    Tensor loss_squeezed = at::squeeze(loss);

    auto iter = TensorIteratorConfig()
      .add_output(loss_squeezed)
      .add_owned_const_input(at::squeeze(input))
      .add_owned_const_input(at::squeeze(target))
      .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        loss.scalar_type(),
        "binary_cross_entropy",
        [&] {
          at::native::cpu_kernel(
              iter, [](scalar_t input_val, scalar_t target_val) {
                TORCH_CHECK(
                    (input_val >= 0) && (input_val <= 1),
                    "all elements of input should be between 0 and 1");
                TORCH_CHECK(
                    (target_val >= 0) && (target_val <= 1),
                    "all elements of target should be between 0 and 1");

                // Binary cross entropy tensor is defined by the equation:
                // L = -w (y ln(x) + (1-y) ln(1-x))
                return (target_val - scalar_t(1)) *
                    std::max(scalar_t(std::log1p(-input_val)), scalar_t(-100)) -
                    target_val *
                    std::max(scalar_t(std::log(input_val)), scalar_t(-100));
              });
        });

    if (weight_opt.has_value() && weight_opt->defined()) {
        loss.mul_(*weight_opt);
    }
    if (reduction != at::Reduction::None) {
        Tensor loss_reduced = apply_loss_reduction(loss, reduction);
        loss.resize_as_(loss_reduced).copy_(loss_reduced);
    }
    return loss;
}

Tensor binary_cross_entropy_backward_cpu(const Tensor& grad, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction) {
    Tensor grad_input = at::empty_like(input);
    return at::native::binary_cross_entropy_backward_out_cpu(
        grad, input, target, weight_opt, reduction, grad_input);
}

Tensor& binary_cross_entropy_backward_out_cpu(const Tensor& grad, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, int64_t reduction, Tensor& grad_input) {
    Tensor grad_input_squeezed = at::squeeze(grad_input);

    auto iter = TensorIteratorConfig()
      .add_output(grad_input_squeezed)
      .add_owned_const_input(at::squeeze(grad))
      .add_owned_const_input(at::squeeze(input))
      .add_owned_const_input(at::squeeze(target))
      .build();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        grad_input.scalar_type(),
        "binary_cross_entropy_backward",
        [&] {
          at::native::cpu_kernel(
              iter,
              [](scalar_t grad_val, scalar_t input_val, scalar_t target_val) {
                // The gradient is the partial derivative of BCELoss
                // with respect to x
                // d(L)/d(x) = -w (y - x) / (x - x^2)
                return grad_val * (input_val - target_val) /
                    (scalar_t(std::max(
                        (scalar_t(1) - input_val) * input_val,
                        scalar_t(EPSILON))));
              });
        });

    if (weight_opt.has_value() && weight_opt->defined()) {
        grad_input.mul_(*weight_opt);
    }
    if (reduction == at::Reduction::Mean) {
        grad_input.div_(input.numel());
    }
    return grad_input;
}

Tensor binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& pos_weight_opt, int64_t reduction) {
  auto log_sigmoid_input = at::log_sigmoid(input);
  if (pos_weight_opt.has_value() && pos_weight_opt->defined()) {
      // pos_weight need to be broadcasted, thus mul(target) is not inplace.
      auto log_weight = (*pos_weight_opt- 1).mul(target).add_(1);
      log_sigmoid_input.mul_(log_weight);
  }

  Tensor loss = (1 - target).mul_(input).sub_(log_sigmoid_input);

  if (weight_opt.has_value() && weight_opt->defined()) {
      loss.mul_(*weight_opt);
  }

  return apply_loss_reduction(loss, reduction);
}

Tensor poisson_nll_loss(const Tensor& input, const Tensor& target, const bool log_input, const bool full, const double eps, const int64_t reduction)
{
    Tensor loss;
    if (log_input) {
        loss = at::exp(input) - target * input;
    } else {
        loss = input - target * at::log(input + eps);
    }

    if (full) {
        auto stirling_term = target * at::log(target) - target + 0.5 * at::log(2 * c10::pi<double> * target);
        loss += stirling_term.masked_fill(target <= 1, 0);
    }

    return apply_loss_reduction(loss, reduction);
}

Tensor& soft_margin_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto z = at::exp(-target * input);
  // inplace version of: grad_input = -norm * target * z / (1. + z) * grad_output;
  at::mul_out(grad_input, target, z).mul_(-norm);
  z.add_(1);
  grad_input.div_(z).mul_(grad_output);
  return grad_input;
}

Tensor soft_margin_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  auto grad_input = at::empty({0}, input.options());
  at::soft_margin_loss_backward_out(grad_input, grad_output, input, target, reduction);
  return grad_input;
}

Tensor& soft_margin_loss_out(const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output) {
  // compute inplace variant of: output = at::log1p(at::exp(-input * target));
  at::neg_out(output, input).mul_(target).exp_().log1p_();
  if (reduction != Reduction::None) {
    auto tmp = apply_loss_reduction(output, reduction);
    output.resize_({});
    output.copy_(tmp);
  }
  return output;
}

Tensor soft_margin_loss(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  auto output = at::empty({0}, input.options());
  at::soft_margin_loss_out(output, input, target, reduction);
  return output;
}

Tensor& smooth_l1_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double beta, Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(target)
    .add_const_input(grad_output)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();
  smooth_l1_backward_stub(iter.device_type(), iter, norm, beta);
  return grad_input;
}

Tensor smooth_l1_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double beta) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::smooth_l1_loss_backward_out(grad_input, grad_output, input, target, reduction, beta);
}

Tensor huber_loss(const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  Tensor loss = at::empty_like(input);
  auto iter = TensorIterator::borrowing_binary_op(loss, input, target);
  huber_stub(iter.device_type(), iter, delta);
  return apply_loss_reduction(loss, reduction);
}

Tensor& huber_loss_out(const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& result) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  auto iter = TensorIterator::borrowing_binary_op(result, input, target);
  huber_stub(iter.device_type(), iter, delta);
  if (reduction != Reduction::None) {
    auto reduced = apply_loss_reduction(result, reduction);
    result.resize_({});
    result.copy_(reduced);
  }
  return result;
}

Tensor huber_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  auto grad_input = at::zeros_like(input, MemoryFormat::Contiguous);
  return at::huber_loss_backward_out(grad_input, grad_output, input, target, reduction, delta);
}

Tensor& huber_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& grad_input) {
  auto norm = (reduction == Reduction::Mean) ? (1. / input.numel()) : 1.;
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(target)
    .add_const_input(grad_output)
    .build();
  huber_backward_stub(iter.device_type(), iter, norm, delta);
  return grad_input;
}

Tensor mse_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::mse_loss_backward_out(grad_input, grad_output, input, target, reduction);
}

Tensor& mse_loss_backward_out(const Tensor& grad_output,
    const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(target)
    .add_const_input(grad_output)
    .build();
  mse_backward_stub(iter.device_type(), iter, norm);
  return grad_input;
}

Tensor l1_loss(const Tensor& input, const Tensor& target, int64_t reduction) {
  return apply_loss_reduction((input - target).abs(), reduction);
}
}  // namespace at::native
