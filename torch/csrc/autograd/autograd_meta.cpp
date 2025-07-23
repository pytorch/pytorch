#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_has_same_storage_numel.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta.h>
#include <ATen/ops/zeros.h>
#endif

namespace torch::autograd {

using at::Tensor;

// [Forward Grad View/inplace]
// It is important to us to allow view and inplace to work with dual Tensors.
// These operations should either compute the right gradient or raise a
// user-friendly error.

// The basic case where all Tensors are dual Tensors is as follows:
//     # Have:
//     #   foo is a dual Tensor that is not a view
//     #   bar is a dual Tensor of appropriate size (depending on cases) that is
//     not a view
//
//     # Case 1: no view
//     foo.copy_(bar)
//
//     # Case 2: with view, propagate from view to base
//     view = foo[0]
//     view.copy_(bar)
//
//     # Case 3: with view, propagate from base to view
//     view = foo[0]
//     foo.copy_(bar)
//
//     # In both cases, the forward grad of foo must be properly updated.
//     # In the second and third cases, the forward grad of view must match
//     # the one of foo for the subset they have in common.
//
// All these cases can be handled by the following layout constraint on the
// forward grad:
//   - A Tensor and its forward grad (for all levels) must have the same
//   metadata (size, stride
//     conj/neg bit and storage offset). Storage offset must be in this metadata
//     because of as_strided. conj/neg bit must be part of this metadata because
//     of ops like `real`.
//   - View operations must create a forward grad that is a view of the base's
//   forward grad.
//   - Inplace operations must modify the input's forward grad inplace.
//
// This layout constraint is ensured in the `set_fw_grad` function below

// More complex cases arise when non-dual Tensor interact with dual Tensors.
// The two most important cases are:
//
//     # Have:
//     #   foo is a regular Tensor that is not a view
//     #   bar is a dual Tensor of appropriate size (depending on cases) that is
//     not a view
//
//     # Case 4: Changes on the view must propagate to its base
//     view = foo[0]
//     # view is still a regular Tensor here
//     view.copy_(bar)
//     # Now both view and foo are dual Tensor with appropriate forward grad
//
//     # Case 5: Changes on the base must propagate on all its views
//     view = foo[0]
//     # view is still a regular Tensor here
//     base.copy_(bar)
//     # Now both view and foo are dual Tensor with appropriate forward grad
//
//     # NB there is a case 6 involving changes on a view propagating to other
//     views # but it is fully described by the two others and is skipped in
//     this discussion.
//
// Case 4 is handled by set_fw_grad by properly setting the forward grad of the
// base if needed. Case 5 is handled in fw_grad by reading the forward grad from
// the base if needed.

namespace utils {

// Enforcing that the metadata between the primal and tangent are same has two
// goals:
// - When properties of the primal are checked in composite op's to determine
//   control flow, the code path decided upon is also reasonable for the tangent
// - Make sure that when the same as_strided is applied to both primal and
//   and tangent, it behaves similarly.
//
// We do that by checking:
//   1) the storages have same properties: size and conj/neg-ness
//   2) the same indices refer to the same elements in storage
//      (we are more strict than necessary here to satisfy the goal 1)
bool has_same_meta(const Variable& base, const Variable& other) {
  if (!base.defined() || !other.defined()) {
    return false;
  }
  // 1) The storages have the same properties
  if (!at::_has_same_storage_numel(base, other)) {
    return false;
  }
  if (base.is_conj() != other.is_conj() || base.is_neg() != other.is_neg()) {
    return false;
  }

  // Technically dim and size belong as part of (2), so we shouldn't really care
  // if a zero-numel tensor violates these. But since these properties
  // (unlike offset and strides) often determine control flow in composite ops
  // it is useful to enforce that they match for primal and tangent here so
  // nothing funny happens later (See goal 1).
  if (base.dim() != other.dim()) {
    return false;
  }
  for (const auto i : c10::irange(base.dim())) {
    if (base.sym_sizes()[i] != other.sym_sizes()[i]) {
      return false;
    }
  }

  // The check below will always be vacuously true for 0-element tensors
  if (base.sym_numel() == 0 && other.sym_numel() == 0) {
    return true;
  }

  // 2) The same indices refer to the same elements in storage
  if (base.sym_storage_offset() != other.sym_storage_offset()) {
    return false;
  }

  for (const auto i : c10::irange(base.dim())) {
    if (base.sym_strides()[i] != other.sym_strides()[i] &&
        base.sym_sizes()[i] != 1 && base.sym_sizes()[i] != 0) {
      return false;
    }
  }
  return true;
}

} // namespace utils

// This function is will ensure that the fw_grad_ is properly a view of the base
// for inplace ops on Tensors that do not have forward grad originally.
void AutogradMeta::set_fw_grad(
    const at::TensorBase& new_grad_base,
    const at::TensorBase& self_base,
    uint64_t level,
    bool is_inplace_op) {
  TORCH_CHECK(
      !new_grad_base._fw_grad(level).defined(),
      "Setting a forward grad that "
      "itself has a forward gradient at the same level",
      level,
      " is not supported.");
  TORCH_INTERNAL_ASSERT(
      (new_grad_base.is_floating_point() || new_grad_base.is_complex()) &&
          (self_base.is_floating_point() || self_base.is_complex()),
      "Expected both tensor and its forward grad to be floating point or complex");
  // Lazy initialization
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fw_grad_) {
      fw_grad_ = std::make_shared<ForwardGrad>();
    }
  }
  if (fw_grad_->contains(level)) {
    // Setting the forward grad again is only allowed if it is a no-op.
    // We do allow this case to simplify writing codegen for inplace ops.
    TORCH_INTERNAL_ASSERT(
        new_grad_base.defined(),
        "Cannot set a forward grad that is an undefined Tensor. Use "
        "_fw_primal(level) to get a new Tensor with this forward grad unset.");

    TORCH_INTERNAL_ASSERT(
        is_inplace_op,
        "Only inplace operations can re-set the forward grad of a Tensor that "
        "already has one.");

    TORCH_INTERNAL_ASSERT(
        fw_grad_->value(level).is_same(new_grad_base),
        "Cannot set a value of a forward grad if it "
        "already exists. Inplace operations should modify it inplace.");
  } else {
    // TODO(alband) remove this spurious version counter bump
    Tensor new_grad(new_grad_base);
    at::OptionalTensorRef self_ref(self_base);
    const Tensor& self = *self_ref;

    TORCH_CHECK(
        self.is_same_size(new_grad),
        "Trying to set a forward gradient that has a different size than that "
        "of the original Tensor, this is not supported. Tensor is of size ",
        self.sizes(),
        " while the given "
        "forward gradient is of size ",
        new_grad.sizes(),
        ".");

    if (is_inplace_op && is_view_) {
      auto this_view_meta = static_cast<DifferentiableViewMeta*>(this);

      // For inplace ops on a Tensor that does not already have a forward grad
      // and is a view, we propagate the tangent to the base and ensure that the
      // new_grad is a view of that base's tangent. This ensure that case 4 from
      // [Forward Grad View/inplace] above works fine What happens in this long
      // if statement is:
      //   - Check if the base already has a grad
      //   - If not, set a new fw_grad for it full of zeros
      //   - Take a view of the base's forward grad
      //   - Copy the given new_grad into this view
      //   - Use this view as the new new_grad
      if (this_view_meta->has_fw_view()) {
        auto& view_info = this_view_meta->get_forward_view();
        auto& base = view_info.base_;

        if (!base._fw_grad(level).defined()) {
          // Enforce same meta here to make sure that the view op below is
          // always valid
          Tensor new_base_fw_grad;
          if (utils::has_same_meta(new_grad, base) &&
              utils::has_same_meta(new_grad, self)) {
            // TODO extend this special case to when the underlying storage of
            // new_grad can be reused.
            new_base_fw_grad = new_grad;
          } else {
            new_base_fw_grad =
                at::_new_zeros_with_same_feature_meta(new_grad, base);
            new_base_fw_grad._set_conj(base.is_conj());
            new_base_fw_grad._set_neg(base.is_neg());

            // Update new_grad to be a view of the base
            Tensor new_fw_grad_value;
            if (view_info.has_view_fn()) {
              new_fw_grad_value = view_info.view_fn()(new_base_fw_grad);
            } else {
              new_fw_grad_value = new_base_fw_grad.as_strided(
                  self.sizes(), self.strides(), self.storage_offset());
            }

            new_fw_grad_value.copy_(new_grad);
            new_grad = new_fw_grad_value;
          }

          base._set_fw_grad(new_base_fw_grad, level, /* is_inplace_op */ false);
        }
      }
    }

    // Enforce the basic layout constraint
    if (!utils::has_same_meta(new_grad, self)) {
      if (is_view_) {
        auto this_view_meta = static_cast<DifferentiableViewMeta*>(this);
        TORCH_INTERNAL_ASSERT(
            !this_view_meta->has_fw_view(),
            "Expected the output of forward differentiable view operations to have the tangent have the same layout as primal")
      }
      auto res = at::_new_zeros_with_same_feature_meta(new_grad, self);
      res._set_conj(self.is_conj());
      res._set_neg(self.is_neg());
      res.copy_(new_grad);
      new_grad = res;
    }

    fw_grad_->set_value(new_grad, level);
  }
}

const Variable& AutogradMeta::fw_grad(
    uint64_t level,
    const at::TensorBase& self) const {
  // TLS that disables forward AD.
  if (!c10::AutogradState::get_tls_state().get_fw_grad_mode()) {
    return ForwardGrad::undef_grad();
  }

  // Ensure that concurrent fw_grad() "reads" are thread safe
  std::lock_guard<std::mutex> lock(mutex_);

  const auto& direct_fw_grad =
      fw_grad_ ? fw_grad_->value(level) : ForwardGrad::undef_grad();

  if (!direct_fw_grad.defined() && is_view_) {
    // For view that don't have a forward grad, check if their base has one that
    // has been defined by an inplace operation.
    // This ensure that case 5 from [Forward Grad View/inplace] above works fine
    auto const_view_meta =
        static_cast<const torch::autograd::DifferentiableViewMeta*>(this);
    // This is ok to do as we ONLY modify fw_grad_ and this field is properly
    // locked in all methods
    if (const_view_meta->has_fw_view()) {
      const auto& view_info = const_view_meta->get_forward_view();
      const auto& base = view_info.base_;

      const auto& base_val = base._fw_grad(level);
      if (base_val.defined()) {
        // Lazy initialization of fw_grad_
        const_view_meta->fw_grad_ = std::make_shared<ForwardGrad>();

        Variable new_val;
        if (view_info.has_view_fn()) {
          new_val = view_info.view_fn()(base_val);
        } else {
          new_val = base_val.as_strided(
              self.sizes(), self.strides(), self.storage_offset());
        }

        const_view_meta->fw_grad_->set_value(new_val, level);
        return const_view_meta->fw_grad_->value(level);
      }
    }
  }
  return direct_fw_grad;
}

} // namespace torch::autograd
