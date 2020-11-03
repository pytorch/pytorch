#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/VariableHooksInterface.h>

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
    std::cerr << "[" << toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

std::string Tensor::toString() const {
  std::string base_str;
  if (scalar_type() == ScalarType::Undefined) {
    base_str = "UndefinedType";
  } else {
    base_str = std::string(at::toString(options().backend())) + at::toString(scalar_type()) + "Type";
  }
  return base_str;
}

Tensor Tensor::variable_data() const {
  return impl::GetVariableHooks()->variable_data(*this);
}

Tensor Tensor::tensor_data() const {
  return impl::GetVariableHooks()->tensor_data(*this);
}

// View Variables
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool Tensor::is_view() const {
  return impl_->view_meta() ? impl_->view_meta()->is_view() : false;
}

const Tensor& Tensor::_base() const {
  TORCH_CHECK(impl_->view_meta());
  return impl_->view_meta()->_base();
}

const std::string& Tensor::name() const {
  return impl::GetVariableHooks()->name(*this);
}

const std::shared_ptr<torch::autograd::Node>& Tensor::grad_fn() const {
  return impl::GetVariableHooks()->grad_fn(*this);
}

void Tensor::remove_hook(unsigned pos) const {
  impl::GetVariableHooks()->remove_hook(*this, pos);
}

unsigned Tensor::_register_hook(std::function<Tensor(const Tensor&)> hook) const {
  return impl::GetVariableHooks()->_register_hook(*this, std::move(hook));
}

ViewMeta::ViewMeta(at::TensorImpl* self_impl, at::Tensor base, CreationMeta creation_meta)
    : creation_meta(creation_meta) {
  base_ = std::move(base);
  TORCH_CHECK(base_.defined(), "base is undefined");
  if (base_.is_view()) {
    base_ = base_._base();
  }
  is_view_ = true;
  self_impl->set_version_counter(base_.unsafeGetTensorImpl()->version_counter());
  attr_version = self_impl->version_counter().current_version();
}

ViewMeta::~ViewMeta() {
  base_.reset();
}

} // namespace at
