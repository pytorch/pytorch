#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/FunctionalTensorImplBase.h>

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
    base_str = std::string(at::toString(options().computeDispatchKey())) + at::toString(scalar_type()) + "Type";
  }
  return base_str;
}

Tensor Tensor::variable_data() const {
  return impl::GetVariableHooks()->variable_data(*this);
}

Tensor Tensor::tensor_data() const {
  return impl::GetVariableHooks()->tensor_data(*this);
}

bool Tensor::is_leaf() const {
  return impl::GetVariableHooks()->is_leaf(*this);
}

int64_t Tensor::output_nr() const {
  return impl::GetVariableHooks()->output_nr(*this);
}

void Tensor::set_data(const Tensor & new_data) const {
  impl::GetVariableHooks()->set_data(*this, new_data);
}

Tensor Tensor::data() const {
  return impl::GetVariableHooks()->data(*this);
}

int64_t Tensor::_version() const {
  return impl::GetVariableHooks()->_version(*this);
}

void Tensor::retain_grad() const {
  impl::GetVariableHooks()->retain_grad(*this);
}

bool Tensor::retains_grad() const {
  return impl::GetVariableHooks()->retains_grad(*this);
}

void Tensor::_backward(TensorList inputs,
        const c10::optional<Tensor>& gradient,
        c10::optional<bool> keep_graph,
        bool create_graph) const {
  return impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
}

const Tensor& Tensor::requires_grad_(bool _requires_grad) const {
  impl::GetVariableHooks()->requires_grad_(*this, _requires_grad);
  return *this;
}

// View Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const char* ViewMetaTypeToString(ViewMeta::Type t) {
  switch (t) {
      case ViewMeta::Type::noOp:
        return "noOp";
      case ViewMeta::Type::invalid:
        return "invalid";
      case ViewMeta::Type::_conj:
        return "_conj";
      case ViewMeta::Type::_indices:
        return "_indices";
      case ViewMeta::Type::_neg_view:
        return "_neg_view";
      case ViewMeta::Type::_reshape_alias:
        return "_reshape_alias";
      case ViewMeta::Type::_values:
        return "_values";
      case ViewMeta::Type::alias:
        return "alias";
      case ViewMeta::Type::as_strided:
        return "as_strided";
      case ViewMeta::Type::as_strided_:
        return "as_strided_";
      case ViewMeta::Type::detach:
        return "detach";
      case ViewMeta::Type::detach_:
        return "detach_";
      case ViewMeta::Type::diagonal:
        return "diagonal";
      case ViewMeta::Type::expand:
        return "expand";
      case ViewMeta::Type::indices:
        return "indices";
      case ViewMeta::Type::permute:
        return "permute";
      case ViewMeta::Type::select_int:
        return "select_int";
      case ViewMeta::Type::slice_Tensor:
        return "slice_Tensor";
      case ViewMeta::Type::split_Tensor:
        return "split_Tensor";
      case ViewMeta::Type::split_with_sizes:
        return "split_with_sizes";
      case ViewMeta::Type::squeeze:
        return "squeeze";
      case ViewMeta::Type::squeeze_dim:
        return "squeeze_dim";
      case ViewMeta::Type::squeeze_:
        return "squeeze_";
      case ViewMeta::Type::squeeze__dim:
        return "squeeze__dim";
      case ViewMeta::Type::t:
        return "t";
      case ViewMeta::Type::t_:
        return "t_";
      case ViewMeta::Type::transpose_int:
        return "transpose_int";
      case ViewMeta::Type::transpose_:
        return "transpose_";
      case ViewMeta::Type::unbind_int:
        return "unbind_int";
      case ViewMeta::Type::unfold:
        return "unfold";
      case ViewMeta::Type::unsqueeze:
        return "unsqueeze";
      case ViewMeta::Type::unsqueeze_:
        return "unsqueeze_";
      case ViewMeta::Type::values:
        return "values";
      case ViewMeta::Type::view:
        return "view";
      case ViewMeta::Type::view_dtype:
        return "view_dtype";
      case ViewMeta::Type::view_as_complex:
        return "view_as_complex";
      case ViewMeta::Type::view_as_real:
        return "view_as_real";
    default:
      return "UNKNOWN_VIEW_TYPE";
  }
}

std::ostream& operator<<(std::ostream& str, ViewMeta::Type rhs) {
  return str << ViewMetaTypeToString(rhs);
}


const at::Tensor& Alias::base() const {
  return base_;
}

// metas is taken by value on purpose - we want to copy the vector.
void Alias::add_update(const at::Tensor& updated_val, std::vector<ViewMeta> metas) {
  updates_.push_back({updated_val, metas});
  generation_++;
}

void Alias::apply_update(const Update& update) {
  // TODO: Should handle more kinds of view ops. Only do reshape now.
  at::Tensor t = update.new_val;
  for(int i = update.view_metas.size()-1; i >= 0; --i) {
    switch (update.view_metas[i].view_type) {
      case ViewMeta::Type::view:
          t = t.view_copy(update.view_metas[i].source_size);
          break;
      case ViewMeta::Type::noOp:
          break;
      default:
          TORCH_CHECK(false, "Other types are not supported yet.");
    }
  }
  base_.replace_(t);
}

void Alias::SyncUpdateOperations() {
  for (auto& update_data: updates_) {
    apply_update(update_data);
  }
  updates_.clear();
}

bool Tensor::is_view() const {
  return impl::GetVariableHooks()->is_view(*this);
}

const Tensor& Tensor::_base() const {
  return impl::GetVariableHooks()->base(*this);
}

bool Tensor::has_view_meta() const {
    // We want this function to be fast, since it's used to detect if two tensors alias.
    // The first check is a fastpath for normal TensorImpl backends that haven't opted into
    // the functionalization pass.
    auto functional_impl = dynamic_cast<const at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
    return functional_impl != nullptr && functional_impl->is_view();
}
void Tensor::sync_() const {
  auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
  // Skip if this is not a functional tensor.
  // This happens when we call repr(tensor), since we need to make sure the tensor is up to date before printing.
  if (functional_impl != nullptr) {
    functional_impl->sync_();
  }
}
bool Tensor::is_up_to_date() const {
  auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
    return functional_impl != nullptr && functional_impl->is_up_to_date();
}
void Tensor::maybe_add_update() const {
    auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
    functional_impl->maybe_add_update(*this);
}
void Tensor::set_view_meta(const at::Tensor& other, ViewMeta meta) const {
    auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
    functional_impl->set_view_meta(other, meta);
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

bool Tensor::is_alias_of(const at::Tensor& other) const {
  // If self and other are the same
  if (unsafeGetTensorImpl() == other.unsafeGetTensorImpl()) return true;
  // For tensors without storage, check alias_ information
  if (has_view_meta() && other.has_view_meta()) {
     return at::unsafeGetFunctionalImplBase(*this)->alias()->base().unsafeGetTensorImpl()
         == at::unsafeGetFunctionalImplBase(other)->alias()->base().unsafeGetTensorImpl();
  }

  // Note: this might break in some cases with view ops that don't use the functionalization pass yet.
  return impl_->storage().is_alias_of(other.storage());
}

unsigned Tensor::_register_hook(std::function<Tensor(const Tensor&)> hook) const {
  return impl::GetVariableHooks()->_register_hook(*this, std::move(hook));
}

} // namespace at
