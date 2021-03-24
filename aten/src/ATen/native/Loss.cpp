#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

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

namespace at { namespace native {

DEFINE_DISPATCH(smooth_l1_stub);
DEFINE_DISPATCH(smooth_l1_backward_stub);
DEFINE_DISPATCH(huber_stub);
DEFINE_DISPATCH(huber_backward_stub);
DEFINE_DISPATCH(mse_stub);
DEFINE_DISPATCH(mse_backward_stub);

Tensor cosine_embedding_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");

  auto prod_sum = (input1 * input2).sum(1);
  auto mag_square1 = (input1 * input1).sum(1) + EPSILON;
  auto mag_square2 = (input2 * input2).sum(1) + EPSILON;
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
  auto margin_clamp = (margin - self).clamp_min_(0);
  auto output_margin = at::where(target != 1, margin_clamp, zeros);
  auto output_self = at::where(target != -1, self, zeros);
  auto output = output_margin + output_self;
  return apply_loss_reduction(output, reduction);
}

Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin,
                           double p, double eps, bool swap, int64_t reduction) {
  auto dist_pos = at::pairwise_distance(anchor, positive, p, eps);
  auto dist_neg = at::pairwise_distance(anchor, negative, p, eps);
  if (swap) {
    auto dist_swap = at::pairwise_distance(positive, negative, p, eps);
    dist_neg = at::min(dist_neg, dist_swap);
  }
  auto output = at::clamp_min(margin + dist_pos - dist_neg, 0);
  return apply_loss_reduction(output, reduction);
}

Tensor margin_ranking_loss(const Tensor& input1, const Tensor& input2, const Tensor& target, double margin, int64_t reduction) {
  auto output =  (-target * (input1 - input2) + margin).clamp_min_(0);
  return apply_loss_reduction(output, reduction);
}

Tensor _kl_div_log_target(const Tensor& input, const Tensor& target, int64_t reduction) {
  auto output = at::exp(target) * (target - input);
  return apply_loss_reduction(output, reduction);
}

Tensor _kl_div_non_log_target(const Tensor& input, const Tensor& target, int64_t reduction) {
  auto output_pos = target * (at::log(target) - input);
  auto zeros = at::zeros_like(output_pos, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto output = at::where(target > 0, output_pos, zeros);
  return apply_loss_reduction(output, reduction);
}

Tensor kl_div(const Tensor& input, const Tensor& target, int64_t reduction, bool log_target) {
  return log_target ? _kl_div_log_target(input, target, reduction)
                    : _kl_div_non_log_target(input, target, reduction);
}

Tensor kl_div_backward_cpu(const Tensor& grad, const Tensor& input, const Tensor& target, int64_t reduction, bool log_target) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_expand = grad.expand_as(input);
  if (!log_target) {
    auto iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_input(target)
      .add_input(grad_expand)
      .build();
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kl_div_backward_cpu", [&]() {
      cpu_serial_kernel(iter, [](scalar_t target_val, scalar_t grad_val) -> scalar_t{
        return target_val > 0 ? -target_val * grad_val : 0;
      });
    });
  }
  else {
    grad_input = -at::exp(target) * grad_expand;
  }

  if (reduction == at::Reduction::Mean) {
    return grad_input / input.numel();
  }
  return grad_input;
}

Tensor binary_cross_entropy_cpu(const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
    Tensor loss = at::empty_like(input);
    return at::native::binary_cross_entropy_out_cpu(loss, input, target, weight, reduction);
}

Tensor& binary_cross_entropy_out_cpu(Tensor& loss, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
    Tensor loss_squeezed = at::squeeze(loss);

    auto iter = TensorIteratorConfig()
      .add_output(loss_squeezed)
      .add_input(at::squeeze(input))
      .add_input(at::squeeze(target))
      .build();

    AT_DISPATCH_FLOATING_TYPES(loss.scalar_type(), "binary_cross_entropy", [&] {
        at::native::cpu_kernel(
            iter,
            [] (scalar_t input_val, scalar_t target_val) {
                TORCH_CHECK(
                    (input_val >= 0) && (input_val <= 1),
                    "all elements of input should be between 0 and 1"
                );

                // Binary cross entropy tensor is defined by the equation:
                // L = -w (y ln(x) + (1-y) ln(1-x))
                return (target_val - scalar_t(1))
                    * std::max(scalar_t(std::log(scalar_t(1) - input_val)), scalar_t(-100))
                    - target_val * std::max(scalar_t(std::log(input_val)), scalar_t(-100));
            }
        );
    });
    if (weight.defined()) {
        loss.mul_(weight);
    }
    if (reduction != at::Reduction::None) {
        Tensor loss_reduced = apply_loss_reduction(loss, reduction);
        loss.resize_as_(loss_reduced).copy_(loss_reduced);
    }
    return loss;
}

Tensor binary_cross_entropy_backward_cpu(const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
    Tensor grad_input = at::empty_like(input);
    return at::native::binary_cross_entropy_backward_out_cpu(grad_input, grad, input, target, weight, reduction);
}

Tensor& binary_cross_entropy_backward_out_cpu(Tensor& grad_input, const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction) {
    Tensor grad_input_squeezed = at::squeeze(grad_input);

    auto iter = TensorIteratorConfig()
      .add_output(grad_input_squeezed)
      .add_input(at::squeeze(grad))
      .add_input(at::squeeze(input))
      .add_input(at::squeeze(target))
      .build();

    AT_DISPATCH_FLOATING_TYPES(grad_input.scalar_type(), "binary_cross_entropy_backward", [&] {
        at::native::cpu_kernel(
            iter,
            [] (scalar_t grad_val, scalar_t input_val, scalar_t target_val) {
                // The gradient is the partial derivative of BCELoss
                // with respect to x
                // d(L)/d(x) = -w (y - x) / (x - x^2)
                return grad_val * (input_val - target_val)
                    / (scalar_t(std::max(
                        (scalar_t(1) - input_val) * input_val,
                        scalar_t(EPSILON)
                    )));
            }
        );
    });
    if (weight.defined()) {
        grad_input.mul_(weight);
    }
    if (reduction == at::Reduction::Mean) {
        grad_input.div_(input.numel());
    }
    return grad_input;
}

Tensor binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
    Tensor loss;
    auto max_val = (-input).clamp_min_(0);
    if (pos_weight.defined()) {
        // pos_weight need to be broadcasted, thus mul(target) is not inplace.
        auto log_weight = (pos_weight - 1).mul(target).add_(1);
        loss = (1 - target).mul_(input).add_(log_weight.mul_(((-max_val).exp_().add_((-input - max_val).exp_())).log_().add_(max_val)));
    } else {
        loss = (1 - target).mul_(input).add_(max_val).add_((-max_val).exp_().add_((-input -max_val).exp_()).log_());
    }

    if (weight.defined()) {
        loss.mul_(weight);
    }

    return apply_loss_reduction(loss, reduction);
}

Tensor binary_cross_entropy_with_logits_backward(const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
    Tensor grad_input;
    if (pos_weight.defined()) {
        // pos_weight need to be broadcasted, thus mul(target) is not inplace.
        auto t = pos_weight.mul(target);
        grad_input = t.add(1).sub_(target).mul_(input.sigmoid()).sub_(t).mul_(grad);
    } else {
        grad_input = (input.sigmoid() - target).mul_(grad);
    }

    if (weight.defined()) {
        grad_input.mul_(weight);
    }

    if (reduction == at::Reduction::Mean) {
        return grad_input / input.numel();
    }

    return grad_input;
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

Tensor& soft_margin_loss_backward_out(Tensor& grad_input, const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
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

Tensor& soft_margin_loss_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  // compute inplace variant of: output = at::log(1. + at::exp(-input * target));
  at::neg_out(output, input).mul_(target).exp_().add_(1.).log_();
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

Tensor smooth_l1_loss(const Tensor& input, const Tensor& target, const int64_t reduction, double beta) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  if (beta == 0) {
      return at::native::l1_loss(input, target, reduction);
  }
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, input, target);
  smooth_l1_stub(iter.device_type(), iter, beta);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& smooth_l1_loss_out(Tensor& result, const Tensor& input, const Tensor& target, int64_t reduction, double beta) {
  TORCH_CHECK(beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  if (beta == 0) {
      return at::native::l1_loss_out(result, input, target, reduction);
  }
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, input, target);
    smooth_l1_stub(iter.device_type(), iter, beta);
    if (reduction == Reduction::Mean) {
      at::mean_out(result, iter.output(), 0);
    } else {
      at::sum_out(result, iter.output(), 0);
    }
  } else {
    auto iter = TensorIterator::binary_op(result, input, target);
    smooth_l1_stub(iter.device_type(), iter, beta);
  }
  return result;
}

Tensor& smooth_l1_loss_backward_out(Tensor& grad_input, const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double beta) {
  if (beta <= 0)
      return at::native::l1_loss_backward_out(grad_input, grad_output, input, target, reduction);
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_input(input)
    .add_input(target)
    .add_input(grad_output)
    .build();
  smooth_l1_backward_stub(iter.device_type(), iter, norm, beta);
  return grad_input;
}

Tensor smooth_l1_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, double beta) {
  if (beta <= 0)
      return at::native::l1_loss_backward(grad_output, input, target, reduction);
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::smooth_l1_loss_backward_out(grad_input, grad_output, input, target, reduction, beta);
}

Tensor huber_loss(const Tensor& input, const Tensor& target, int64_t reduction, double delta) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  Tensor loss = at::empty_like(input);
  auto iter = TensorIterator::binary_op(loss, input, target);
  huber_stub(iter.device_type(), iter, delta);
  return apply_loss_reduction(loss, reduction);
}

Tensor& huber_loss_out(const Tensor& input, const Tensor& target, int64_t reduction, double delta, Tensor& result) {
  TORCH_CHECK(delta > 0, "huber_loss does not support non-positive values for delta.")
  auto iter = TensorIterator::binary_op(result, input, target);
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
    .add_input(input)
    .add_input(target)
    .add_input(grad_output)
    .build();
  huber_backward_stub(iter.device_type(), iter, norm, delta);
  return grad_input;
}

Tensor mse_loss(const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, input, target);
  mse_stub(iter.device_type(), iter);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& mse_loss_out(Tensor&result, const Tensor& input, const Tensor& target, int64_t reduction) {
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, input, target);
    mse_stub(iter.device_type(), iter);
    if (reduction == Reduction::Mean) {
      at::mean_out(result, iter.output(), 0);
    } else {
      at::sum_out(result, iter.output(), 0);
    }
  } else {
    auto iter = TensorIterator::binary_op(result, input, target);
    mse_stub(iter.device_type(), iter);;
  }
  return result;
}

Tensor mse_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::mse_loss_backward_out(grad_input, grad_output, input, target, reduction);
}

Tensor& mse_loss_backward_out(Tensor& grad_input, const Tensor& grad_output,
    const Tensor& input, const Tensor& target, int64_t reduction) {
  auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
  auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_input(input)
    .add_input(target)
    .add_input(grad_output)
    .build();
  mse_backward_stub(iter.device_type(), iter, norm);
  return grad_input;
}

Tensor l1_loss(const Tensor& input, const Tensor& target, int64_t reduction) {
  const auto float_type = c10::toValueType(input.scalar_type());
  Tensor result = at::empty({0}, input.options().dtype(float_type));
  return at::l1_loss_out(result, input, target, reduction);
}

Tensor& l1_loss_out(Tensor& result, const Tensor& input, const Tensor& target, int64_t reduction) {
  if (reduction != Reduction::None) {
    auto diff = at::sub(input, target);
    auto loss = diff.is_complex() ? diff.abs() : diff.abs_();
    if (reduction == Reduction::Mean) {
      return at::mean_out(result, loss, IntArrayRef{});
    } else {
      return at::sum_out(result, loss, IntArrayRef{});
    }
  } else {
    auto diff = input.is_complex() ? at::sub(input, target) : at::sub_out(result, input, target);
    return at::abs_out(result, diff);
  }
}

Tensor l1_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
  Tensor grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::l1_loss_backward_out(grad_input, grad_output, input, target, reduction);
}

Tensor& l1_loss_backward_out(Tensor& grad_input, const Tensor& grad_output,
    const Tensor& input, const Tensor& target, int64_t reduction) {
  auto norm = reduction == Reduction::Mean ? grad_output / input.numel() : grad_output;
  return at::sub_out(grad_input, input, target).sgn_().mul_(norm);
}

}}  // namespace at::native
