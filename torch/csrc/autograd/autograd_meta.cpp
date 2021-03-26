#include <torch/csrc/autograd/variable.h>

namespace torch {
namespace autograd {

using at::Tensor;

// [Forward Grad View/inplace]
// It is important to us to allow view and inplace to work with dual Tensors. These operations
// should either compute the right gradient or raise a user-friendly error.

// The basic case where all Tensors are dual Tensors is as follows:
//     # Have:
//     #   foo is a dual Tensor that is not a view
//     #   bar is a dual Tensor of appropriate size (depending on cases) that is not a view
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
// All these cases can be handled by the following layout constraint on the forward grad:
//   - A Tensor and its forward grad (for all levels) must have the same metadata (size, stride
//     and storage offset). Storage offset must be in this metadata because of as_strided.
//   - View operations must create a forward grad that is a view of the base's forward grad.
//   - Inplace operations must modify the input's forward grad inplace.
//
// This layout constraint is ensured in the `set_fw_grad` function below


// More complex cases arrise when non-dual Tensor interact with dual Tensors.
// The two most important cases are:
//
//     # Have:
//     #   foo is a regular Tensor that is not a view
//     #   bar is a dual Tensor of appropriate size (depending on cases) that is not a view
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
//     # NB there is a case 6 involving changes on a view propagating to other views
//     # but it is fully described by the two others and is skipped in this discussion.
//
// Case 4 is handled by set_fw_grad by properly setting the forward grad of the base if needed.
// Case 5 is handled in fw_grad by reading the forward grad from the base if needed.


namespace {
  // Check if two Tensor have the same storage offset, sizes and strides
  bool has_same_meta(const Variable& base, const Variable& other) {
    if (!base.defined() || !other.defined()) {
      return false;
    }
    if (base.storage_offset() != other.storage_offset()) {
      return false;
    }
    if (base.dim() != other.dim()) {
      return false;
    }
    for (int64_t i=0; i<base.dim(); ++i) {
      if (base.sizes()[i] != other.sizes()[i]) {
        return false;
      }
      if (base.strides()[i] != other.strides()[i]) {
        return false;
      }
    }
    return true;
  }

  Tensor new_with_same_meta(const Variable& base) {
    // We need to create a storage of the same size to be able to have the same
    // viewing behavior in all cases
    // Explicit type here to appease Windows build
    int64_t nelement_in_storage = base.storage().nbytes() / base.itemsize();
    auto new_tensor = at::zeros({nelement_in_storage}, base.options());
    auto res = new_tensor.as_strided(base.sizes(), base.strides(), base.storage_offset());
    return res;
  }
} // anonymous namespace

// This function is will ensure that the fw_grad_ is properly a view of the base for inplace ops on
// Tensors that do not have forward grad originally.
void AutogradMeta::set_fw_grad(const Variable& new_grad_, const Variable& self, uint64_t level, bool is_inplace_op) {
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
    TORCH_INTERNAL_ASSERT(new_grad_.defined(), "Cannot set a forward grad that is an undefined Tensor. Use "
                          "_fw_primal(level) to get a new Tensor with this forward grad unset.");

    TORCH_INTERNAL_ASSERT(is_inplace_op, "Only inplace operations can re-set the forward grad of a Tensor that "
                          "already has one.");

    TORCH_INTERNAL_ASSERT(fw_grad_->value(level).is_same(new_grad_), "Cannot set a value of a forward grad if it "
                          "already exists. Inplace operations should modify it inplace.");
  } else {
    // TODO(alband) remove this spurious version counter bump
    auto new_grad = new_grad_;

    TORCH_CHECK(self.is_same_size(new_grad_), "Trying to set a forward gradient that has a different size than that "
                "of the original Tensor, this is not supported. Tensor is of size ", self.sizes(), " while the given "
                "forward gradient is of size ", new_grad_.sizes(), ".");

    if (is_inplace_op && is_view_) {
      auto this_view_meta = static_cast<DifferentiableViewMeta*>(this);

      // For inplace ops on a Tensor that does not already have a forward grad and is a view, we propagate
      // the tangent to the base and ensure that the new_grad is a view of that base's tangent.
      // This ensure that case 4 from [Forward Grad View/inplace] above works fine
      // What happens in this long if statement is:
      //   - Check if the base already has a grad
      //   - If not, set a new fw_grad for it full of zeros
      //   - Take a view of the base's forward grad
      //   - Copy the given new_grad into this view
      //   - Use this view as the new new_grad
      if (this_view_meta->has_fw_view()) {
        auto view_info = this_view_meta->get_forward_view();
        auto& base = view_info.base_;

        if (!base._fw_grad(level).defined()) {
          // Enforce same meta here to make sure that the view op below is always valid
          Tensor new_base_fw_grad;
          if (has_same_meta(new_grad, base)) {
            // TODO extend this special case to when the underlying storage of new_grad
            // can be re-used.
            new_base_fw_grad = new_grad;
          } else {
            new_base_fw_grad = new_with_same_meta(base);

            // Update new_grad to be a view of the base
            Tensor new_fw_grad_value;
            if (view_info.has_view_fn()) {
              new_fw_grad_value = view_info.view_fn()(new_base_fw_grad);
            } else {
              new_fw_grad_value = new_base_fw_grad.as_strided(self.sizes(), self.strides(), self.storage_offset());
            }

            new_fw_grad_value.copy_(new_grad);
            new_grad = new_fw_grad_value;
          }

          base._set_fw_grad(new_base_fw_grad, level, /* is_inplace_op */ false);
        }
      }
    }

    // Enforce the basic layout constraint
    if (!has_same_meta(new_grad, self)) {
      Tensor new_grad_with_meta = new_with_same_meta(self);
      new_grad_with_meta.copy_(new_grad);
      new_grad = new_grad_with_meta;
    }

    fw_grad_->set_value(new_grad, level);
  }
}

const Variable& AutogradMeta::fw_grad(uint64_t level, const Variable& self) const {
  // Ensure that concurent fw_grad() "reads" are thread safe
  std::lock_guard<std::mutex> lock(mutex_);

  const auto& direct_fw_grad = fw_grad_ ? fw_grad_->value(level) : ForwardGrad::undef_grad();

  if (!direct_fw_grad.defined() && is_view_) {
    // For view that don't have a forward grad, check if their base has one that
    // has been defined by an inplace operation.
    // This ensure that case 5 from [Forward Grad View/inplace] above works fine
    auto const_view_meta = static_cast<const torch::autograd::DifferentiableViewMeta*>(this);
    // This is ok to do as we ONLY modify fw_grad_ and this field is properly locked in all methods
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto this_view_meta = const_cast<torch::autograd::DifferentiableViewMeta*>(const_view_meta);
    if (this_view_meta->has_fw_view()) {
      const auto& view_info = this_view_meta->get_forward_view();
      const auto& base = view_info.base_;

      const auto& base_val = base._fw_grad(level);
      if (base_val.defined()) {
        // Lazy initialization of fw_grad_
        this_view_meta->fw_grad_ = std::make_shared<ForwardGrad>();

        Variable new_val;
        if (view_info.has_view_fn()) {
          new_val = view_info.view_fn()(base_val);
        } else {
          new_val = base_val.as_strided(self.sizes(), self.strides(), self.storage_offset());
        }

        this_view_meta->fw_grad_->set_value(new_val, level);
        return this_view_meta->fw_grad_->value(level);
      }
    }
  }
  return direct_fw_grad;
}

}} // torch::autograd
