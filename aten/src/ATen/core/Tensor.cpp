#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/tensor.h>
// TODO: dependency shindig
#include <torch/csrc/autograd/generated/Functions.h>

#include <iostream>

namespace at {

void Tensor::enforce_invariants() {
  if (impl_.get() == nullptr) {
    throw std::runtime_error("TensorImpl with nullptr is not supported");
  }
  // Following line throws if the method is not a POD data type or is not
  // supported by ATen
  scalar_type();
  if (defined()) {
    TORCH_INTERNAL_ASSERT(
        impl_->dtype_initialized(),
        "Partially-initialized tensor not supported by Tensor");
    TORCH_INTERNAL_ASSERT(
        !impl_->is_sparse(),
        "Sparse Tensors are supported by Tensor, but invariant checking isn't implemented.  Please file a bug.");
    TORCH_INTERNAL_ASSERT(
        impl_->storage_initialized(),
        "Partially-initialized tensor not supported by Tensor");
  }
}

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

std::string Tensor::toString() const {
  return type().toString();
}

Tensor Tensor::variable_data() const noexcept {
  auto self_impl_copy = unsafeGetTensorImpl()->shallow_copy_and_detach(
    /*version_counter=*/0,
    /*allow_tensor_metadata_change=*/false);
  self_impl_copy->set_autograd_meta(nullptr);
  return at::Tensor(self_impl_copy);
}

Tensor Tensor::tensor_data() const noexcept {
  auto self_impl_copy = unsafeGetTensorImpl()->shallow_copy_and_detach(
    /*version_counter=*/unsafeGetTensorImpl()->version_counter(),
    /*allow_tensor_metadata_change=*/unsafeGetTensorImpl()->allow_tensor_metadata_change());
  return at::Tensor(self_impl_copy);
}

// View Variables
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool Tensor::is_view() const noexcept {
  if (torch::autograd::impl::get_autograd_meta(*this)) {
    return torch::autograd::impl::get_autograd_meta(*this)->is_view_;
  } else {
    return false;
  }
}

const Tensor& Variable::base() const {
  if (is_view()) {
    // is_view() implies get_autograd_meta()
    auto diff_view_meta = static_cast<torch::autograd::DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(*this));
    return diff_view_meta->base_;
  } else {
    throw std::runtime_error("Can't get base of non-view Variable");
  }
}

namespace {
  std::string singleton_string;
}

const std::string& Tensor::name() const noexcept {
  if (torch::autograd::impl::get_autograd_meta(*this)) {
    return torch::autograd::impl::get_autograd_meta(*this)->name_;
  } else {
    return singleton_string;
  }
}

namespace {
  std::shared_ptr<Node> singleton_shared_ptr;
}

const std::shared_ptr<Node>& Tensor::grad_fn() const {
  if (is_view()) {
    // NB: is_view() ==> get_autograd_meta()
    auto diff_view_meta = static_cast<DifferentiableViewMeta*>(torch::autograd::impl::get_autograd_meta(*this));
    std::lock_guard<std::mutex> lock(diff_view_meta->mutex_);
    if (!diff_view_meta->grad_fn_ && !diff_view_meta->base_.requires_grad()) {
      return diff_view_meta->grad_fn_;
    }
    auto current_version = this->_version();
    if (diff_view_meta->attr_version != current_version) {
      AT_ASSERT(diff_view_meta->output_nr_ == 0);
      auto fn = std::make_shared<generated::AsStridedBackward>();
      fn->self_geometry = at::TensorGeometry(diff_view_meta->base_);
      fn->size = sizes().vec();
      fn->stride = strides().vec();
      fn->storage_offset = storage_offset();
      fn->set_next_edges(collect_next_edges(diff_view_meta->base_));
      fn->add_input_metadata(
        diff_view_meta->base_.type()
      , sizes() // Note: sizes(), not base_.sizes(), is intentional
      , diff_view_meta->base_.device());
      diff_view_meta->grad_fn_ = std::move(fn);
      diff_view_meta->attr_version = current_version;
    }
    return diff_view_meta->grad_fn_;
  } else {
    if (torch::autograd::impl::get_autograd_meta(*this)) {
      return torch::autograd::impl::get_autograd_meta(*this)->grad_fn_;
    } else {
      return singleton_shared_ptr;
    }
  }
}

void Tensor::remove_hook(unsigned pos) const {
  auto &list = torch::autograd::impl::materialize_autograd_meta(*this)->cpp_hooks_list;
  TORCH_CHECK(list && pos < list->size() , "Invalid index, no hook at position ", pos);
  // Hook will be ignored
  (*list)[pos] = nullptr;
}

unsigned Tensor::_register_hook(std::function<Tensor(const Tensor&)> hook) const {
  TORCH_CHECK(requires_grad(), "cannot register a hook on a variable that "
                           "doesn't require gradient");
  // NB: materialize_autograd_meta unnecessary due to requires grad check
  auto &list = torch::autograd::impl::get_autograd_meta(*this)->cpp_hooks_list;
  if(!list) {
    torch::autograd::impl::create_cpp_hook(*this);
  }
  unsigned idx = list->size();
  list->push_back(hook);
  return idx;
}

} // namespace at
