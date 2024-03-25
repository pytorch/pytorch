#include <ATen/functorch/DynamicLayer.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/irange.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/xnnpack/Engine.h>

namespace at::functorch {

// NOTE: [functorch's PyTorch Operator Hacks]
//
// This file contains hacks for composite PyTorch operators that are problematic.
// For example, the composite op might have in-place operations,
// or call data_ptr. We have some idea of how to fix these things in the long term
// e.g., upstream the changes to PyTorch.
//
// TODO: all of these should be fixed in a more blessed way. In particular,
// it is bad if any of these go out-of-sync with the implementations in
// pytorch/pytorch.

// TODO: upstream into core

namespace {
Tensor index_select_backward_hack(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add(dim, index, grad);
}

static optional<std::tuple<Tensor,int64_t>> unwrap(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (wrapped) {
    if (wrapped->level().has_value()) {
      return std::make_tuple(wrapped->value(), *wrapped->level());
    }
    return unwrap(wrapped->value());
  }
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    return std::make_tuple(batched->value(), batched->level());
  }
  return nullopt;
}

static bool can_perform_inplace(const Tensor& a, const Tensor& b) {
  // TODO: generalize this to more transforms
  auto a_ = unwrap(a);
  auto b_ = unwrap(b);
  if (!a_.has_value() && b_.has_value()) {
    return false;
  }
  if (!a_.has_value() && !b_.has_value()) {
    return true;
  }
  if (a_.has_value() && !b_.has_value()) {
    return true;
  }
  TORCH_INTERNAL_ASSERT(a_.has_value() && b_.has_value());

  // If b has any wrapper that a does not, then we cannot do a.inplace_(b)
  if (std::get<1>(*a_) < std::get<1>(*b_)) {
    return false;
  }
  if (std::get<1>(*a_) > std::get<1>(*b_)) {
    return can_perform_inplace(std::get<0>(*a_), b);
  }
  return can_perform_inplace(std::get<0>(*a_), std::get<0>(*b_));
}

// TODO: linear is pretty important for performance, but I'm not sure how to work
// around the in-place.
Tensor linear_hack(const Tensor& input, const Tensor& weight, const c10::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(std::in_place);

  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (at::native::xnnpack::use_linear(input, weight, *bias)) {
    return at::native::xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  if (input.dim() == 3 && bias->defined() && input.is_contiguous()) {
    // Also hit the fused path for contiguous 3D input.
    const auto input_sizes = input.sizes();
    const auto result = at::addmm(*bias, input.view({input_sizes[0] * input_sizes[1], input_sizes[2]}), weight.t());
    return result.view({input_sizes[0], input_sizes[1], result.size(1)});
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    const auto& stack = getDynamicLayerStack();
    bool any_vmap_layers = std::any_of(
        stack.begin(), stack.end(),
        [](const DynamicLayer& dl){ return dl.key() == TransformType::Vmap; });
    if (any_vmap_layers) {
      return output.add(*bias);
    }
    return output.add_(*bias);
  }
  return output;
}

static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

Tensor binary_cross_entropy_with_logits_hack(
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& pos_weight_opt,
    int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& pos_weight = c10::value_or_else(pos_weight_opt, [] {return Tensor();});

  Tensor loss;
  auto max_val = (-input).clamp_min(0);
  if (pos_weight.defined()) {
    // pos_weight need to be broadcasted, thus mul(target) is not inplace.
    auto log_weight = (pos_weight - 1).mul(target).add_(1);
    loss = (1 - target).mul(input).add(log_weight.mul(((-max_val).exp_().add((-input - max_val).exp_())).log_().add_(max_val)));
  } else {
    loss = (1 - target).mul(input).add_(max_val).add_((-max_val).exp_().add((-input -max_val).exp_()).log_());
  }

  if (weight.defined()) {
    loss = loss * weight;
  }

  return apply_loss_reduction(loss, reduction);
}

Tensor trace_backward_decomp(const Tensor& grad, IntArrayRef sizes) {
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");
  }
  auto grad_input = at::zeros(sizes[0] * sizes[1], grad.options());
  auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));
  // Workaround using index_put instead of yet unsupported index_fill_
  grad_input = grad_input.index_put({indices}, grad);
  return grad_input.view(sizes);
}
}

// dropout hack
// TODO: make the following changes in pytorch/pytorch
namespace dropout_hack {

namespace {

template<bool inplace>
using Ctype = typename std::conditional<inplace, Tensor&, Tensor>::type;

static Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  std::vector<int64_t> sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (C10_UNUSED const auto i : c10::irange(2, input.dim())) {
    sizes.push_back(1);
  }
  // NB: THIS WAS CHANGED FROM THE ORIGINAL
  return at::empty(sizes, input.options());
}

static bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return (input.is_cuda() || input.is_xpu() || input.is_lazy()) && p > 0 && p < 1 && input.numel() > 0;
}

// NB: sure, we could have used different overloads here, but I would feel insecure
// knowing that this dispatch depends only on the constness of the references
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul_(noise);
}

template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul(noise);
}

template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only

  // NB: THIS WAS CHANGED FROM THE ORIGINAL
  Tensor noise;
  if (feature_dropout) {
    auto empty = make_feature_noise(input);
    noise = at::bernoulli(empty, 1 - p);
  } else {
    // NB: it is important that this is at::empty and not at::empty_like
    auto empty = at::empty({}, input.options()).expand(input.sizes());
    noise = at::bernoulli(empty, 1 - p);
  }

  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}

#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )

static Tensor dropout(const Tensor& input, double p, bool train) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (train && is_fused_kernel_acceptable(input, p)) {
      return std::get<0>(at::native_dropout(input, p, train));
    }
    return _dropout<false>(input, p, train);
  }();
  namedinference::propagate_names(result, input);
  return result;
}

Tensor& dropout_(Tensor& input, double p, bool train) {
  return _dropout<true>(input, p, train);
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {
  return _feature_dropout<false>(input, p, train);
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
  return _feature_dropout<true>(input, p, train);
}

Tensor alpha_dropout(const Tensor& input, double p, bool train) {
  return _alpha_dropout<false>(input, p, train);
}

Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
  return _alpha_dropout<true>(input, p, train);
}

Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<false>(input, p, train);
}

Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<true>(input, p, train);
}

}
} // dropout_hack

TORCH_LIBRARY_IMPL(aten, FuncTorchDynamicLayerFrontMode, m) {
  m.impl("index_select_backward", index_select_backward_hack);
  m.impl("linear", linear_hack);
  m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_hack);
  m.impl("trace_backward", trace_backward_decomp);

  m.impl("dropout", dropout_hack::dropout);
  m.impl("feature_dropout", dropout_hack::feature_dropout);
  m.impl("alpha_dropout", dropout_hack::alpha_dropout);
  m.impl("feature_alpha_dropout", dropout_hack::feature_alpha_dropout);

  m.impl("dropout_", dropout_hack::dropout_);
  m.impl("feature_dropout_", dropout_hack::feature_dropout_);
  m.impl("alpha_dropout_", dropout_hack::alpha_dropout_);
  m.impl("feature_alpha_dropout_", dropout_hack::feature_alpha_dropout_);
}

} // namespace at::functorch
