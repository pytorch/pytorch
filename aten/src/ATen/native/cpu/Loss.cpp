
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/binary_cross_entropy_with_logits.h>
#endif

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

namespace at {
namespace native {
namespace {
Tensor& binary_cross_entropy_with_logits_out_cpu(
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& pos_weight_opt,
    int64_t reduction,
    Tensor& loss) {
// See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> pos_weight_maybe_owned = at::borrow_from_optional_tensor(pos_weight_opt);
  const Tensor& pos_weight = *pos_weight_maybe_owned;

  Tensor loss_squeezed = at::squeeze(loss);
  if (pos_weight_opt.has_value()) {
    auto iter = TensorIteratorConfig()
      .add_output(loss_squeezed)
      .add_owned_input(at::squeeze(input))
      .add_owned_input(at::squeeze(target))
      .add_owned_input(at::squeeze(pos_weight))
      .build();
    AT_DISPATCH_FLOATING_TYPES(loss.scalar_type(), "binary_cross_entropy_with_logits", [&] {
      at::native::cpu_kernel(
          iter,
          [] (scalar_t input_val, scalar_t target_val, scalar_t pos_weight_val) {
            auto max_val = std::max(-input_val, static_cast<scalar_t>(0));
            auto log_weight = (pos_weight_val - 1) * target_val + 1;
            auto weight_multiplier = std::log(std::exp(-max_val) + std::exp(-input_val - max_val)) + max_val;
            return (1 - target_val) * input_val + (log_weight * weight_multiplier);
          }
      );
    });
  } else {
    auto iter = TensorIteratorConfig()
      .add_output(loss_squeezed)
      .add_owned_input(at::squeeze(input))
      .add_owned_input(at::squeeze(target))
      .build();
    AT_DISPATCH_FLOATING_TYPES(loss.scalar_type(), "binary_cross_entropy_with_logits", [&] {
      at::native::cpu_kernel(
          iter,
          [] (scalar_t input_val, scalar_t target_val) {
            auto max_val = std::max(-input_val, static_cast<scalar_t>(0));
            return (1 - target_val) * input_val + max_val + std::log(std::exp(-max_val) + std::exp(-input_val - max_val));
          }
      );
    });
  }
  if (weight.defined()) {
    loss.mul_(weight);
  }

  if (reduction != at::Reduction::None) {
    Tensor loss_reduced = apply_loss_reduction(loss, reduction);
    loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }
  return loss;
}
} // namespace

Tensor binary_cross_entropy_with_logits_cpu(const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& pos_weight_opt, int64_t reduction) {
  Tensor loss = at::empty_like(input);
  return binary_cross_entropy_with_logits_out_cpu(
      input, target, weight_opt, pos_weight_opt, reduction, loss);
}


} // namespace native
} // namespace at
