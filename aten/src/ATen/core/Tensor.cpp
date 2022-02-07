#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>

#include <iostream>

namespace at {

const TensorBase& get_tensor_base(const Tensor &t) {
  return t;
}

TensorBase TensorBase::__dispatch_contiguous(c10::MemoryFormat memory_format) const {
  OptionalTensorRef self(*this);
  return at::_ops::contiguous::call(*self, memory_format);
}

const TensorBase& TensorBase::fill_(const c10::Scalar &fill_value) const {
  Tensor self(*this);
  at::_ops::fill__Scalar::call(self, fill_value);
  return *this;
}

const TensorBase& TensorBase::zero_() const {
  Tensor self(*this);
  at::_ops::zero_::call(self);
  return *this;
}

void TensorBase::enforce_invariants() {
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
        !impl_->has_storage() || impl_->is_meta() || impl_->storage_initialized(),
        "Partially-initialized tensor not supported by Tensor");
  }
}

void TensorBase::print() const {
  if (defined()) {
    std::cerr << "[" << toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

std::string TensorBase::toString() const {
  std::string base_str;
  if (scalar_type() == ScalarType::Undefined) {
    base_str = "UndefinedType";
  } else {
    base_str = std::string(at::toString(options().computeDispatchKey())) + at::toString(scalar_type()) + "Type";
  }
  return base_str;
}

TensorBase TensorBase::variable_data() const {
  return impl::GetVariableHooks()->variable_data(*this);
}

TensorBase TensorBase::tensor_data() const {
  return impl::GetVariableHooks()->tensor_data(*this);
}

bool TensorBase::is_leaf() const {
  return impl::GetVariableHooks()->is_leaf(*this);
}

int64_t TensorBase::output_nr() const {
  return impl::GetVariableHooks()->output_nr(*this);
}

void TensorBase::set_data(const TensorBase & new_data) const {
  impl::GetVariableHooks()->set_data(*this, new_data);
}

TensorBase TensorBase::data() const {
  return impl::GetVariableHooks()->data(*this);
}

int64_t TensorBase::_version() const {
  return impl::GetVariableHooks()->_version(*this);
}

void TensorBase::retain_grad() const {
  impl::GetVariableHooks()->retain_grad(*this);
}

bool TensorBase::retains_grad() const {
  return impl::GetVariableHooks()->retains_grad(*this);
}

void Tensor::backward(const Tensor & gradient, c10::optional<bool> retain_graph, bool create_graph, c10::optional<TensorList> inputs) const {
  // NB: Adding this wrapper to _backward here because we'd like our
  // 'backwards' api to accept the 'inputs' argument optionally. Since code gen
  // currently does not support optional of TensorList our approach is to replace
  // backward in native_functions.yaml with _backward and call it here instead.
  if (inputs.has_value()) {
    TORCH_CHECK(inputs.value().size() > 0, "'inputs' argument to backward cannot be empty")
    this->_backward(inputs.value(), gradient, retain_graph, create_graph);
  } else {
    this->_backward({}, gradient, retain_graph, create_graph);
  }
}

void Tensor::_backward(ITensorList inputs,
        const c10::optional<Tensor>& gradient,
        c10::optional<bool> keep_graph,
        bool create_graph) const {
  return impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
}

const TensorBase& TensorBase::requires_grad_(bool _requires_grad) const {
  impl::GetVariableHooks()->requires_grad_(*this, _requires_grad);
  return *this;
}

// View Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool TensorBase::is_view() const {
  return impl::GetVariableHooks()->is_view(*this);
}

const TensorBase& TensorBase::_base() const {
  return impl::GetVariableHooks()->base(*this);
}

const std::string& TensorBase::name() const {
  return impl::GetVariableHooks()->name(*this);
}

const std::shared_ptr<torch::autograd::Node>& TensorBase::grad_fn() const {
  return impl::GetVariableHooks()->grad_fn(*this);
}

void TensorBase::remove_hook(unsigned pos) const {
  impl::GetVariableHooks()->remove_hook(*this, pos);
}

unsigned TensorBase::_register_hook(std::function<TensorBase(const TensorBase&)> hook) const {
  return impl::GetVariableHooks()->_register_hook(*this, std::move(hook));
}

} // namespace at
