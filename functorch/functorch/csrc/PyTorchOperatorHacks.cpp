#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/Constants.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/BatchedTensorImpl.h>

namespace at { namespace functorch {

// TODO: all of these should be fixed in a more blessed way. In particular,
// it is bad if any of these go out-of-sync with the implementations in
// pytorch/pytorch.
//
// This file contains hacks for composite PyTorch operators that are problematic.
// For example, the composite op might have in-place operations,
// or call data_ptr. We have some idea of how to fix these things in the long term
// (e.g. functionalization for the in-place operations).

// TODO: can replace with better conditional functionalization
static Tensor value_selecting_reduction_backward_hack(
    const Tensor& grad,
    int64_t dim,
    const Tensor& indices,
    IntArrayRef sizes,
    bool keepdim) {
  if (!keepdim && sizes.size() > 0) {
    auto grad_ = grad.unsqueeze(dim);
    auto indices_ = indices.unsqueeze(dim);
    return at::zeros(sizes, grad_.options()).scatter(dim, indices_, grad_);
  }
  return at::zeros(sizes, grad.options()).scatter(dim, indices, grad);
}

// TODO: upstream into core
Tensor index_select_backward_hack(const Tensor& grad, IntArrayRef self_sizes, int64_t dim, const Tensor& index) {
  return at::zeros(self_sizes, grad.options()).index_add(dim, index, grad);
}

// TODO: https://github.com/pytorch/pytorch/issues/69991
Tensor frobenius_norm_dim_hack(const Tensor& self, IntArrayRef dim, bool keepdim) {
  if (dim.size() == 1 || dim.size() == 0) {
    return at::norm(self, 2, dim, keepdim);
  } else {
    auto dim_ = dim.vec();
    maybe_wrap_dims(dim_, self.dim());
    TORCH_CHECK(dim_[0] != dim_[1], "Expected dims to be different, got ", dim, " instead");
    if (self.is_complex()){
      return at::sqrt(at::sum(at::real(self.conj() * self), dim_, keepdim));
    } else {
      return at::sqrt(at::sum((self * self), dim_, keepdim));
    }
  }
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
    : c10::MaybeOwned<Tensor>::owned(c10::in_place);

  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_linear(input, weight, *bias)) {
    return xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    // TODO(rzou): I'm a little uncomfortable with this
    if (can_perform_inplace(output, *bias)) {
      return output.add_(*bias);
    }
    return output.add(*bias);
  }
  return output;
}

TORCH_LIBRARY_IMPL(aten, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  m.impl("value_selecting_reduction_backward", value_selecting_reduction_backward_hack);
  m.impl("index_select_backward", index_select_backward_hack);
  m.impl("frobenius_norm.dim", frobenius_norm_dim_hack);
  m.impl("linear", linear_hack);
}

}}
