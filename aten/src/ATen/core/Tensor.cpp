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
    case ViewMeta::Type::real:
      return "real";
    case ViewMeta::Type::imag:
      return "imag";
    case ViewMeta::Type::as_strided:
      return "as_strided";
    case ViewMeta::Type::tensor_split_sections:
      return "tensor_split_sections";
    case ViewMeta::Type::tensor_split_indices:
      return "tensor_split_indices";
    case ViewMeta::Type::tensor_split_tensor_indices_or_sections:
      return "tensor_split_tensor_indices_or_sections";
    case ViewMeta::Type::contiguous:
      return "contiguous";
    case ViewMeta::Type::expand:
      return "expand";
    case ViewMeta::Type::flatten_using_ints:
      return "flatten_using_ints";
    case ViewMeta::Type::flatten_named_out_dim:
      return "flatten_named_out_dim";
    case ViewMeta::Type::flatten_using_names:
      return "flatten_using_names";
    case ViewMeta::Type::flatten_DimnameList:
      return "flatten_DimnameList";
    case ViewMeta::Type::permute:
      return "permute";
    case ViewMeta::Type::numpy_T:
      return "numpy_T";
    case ViewMeta::Type::reshape:
      return "reshape";
    case ViewMeta::Type::_reshape_alias:
      return "_reshape_alias";
    case ViewMeta::Type::select_Dimname:
      return "select_Dimname";
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
    case ViewMeta::Type::squeeze_dimname:
      return "squeeze_dimname";
    case ViewMeta::Type::t:
      return "t";
    case ViewMeta::Type::view_as:
      return "view_as";
    case ViewMeta::Type::_indices:
      return "_indices";
    case ViewMeta::Type::values:
      return "values";
    case ViewMeta::Type::unbind_int:
      return "unbind_int";
    case ViewMeta::Type::unbind_Dimname:
      return "unbind_Dimname";
    case ViewMeta::Type::swapaxes:
      return "swapaxes";
    case ViewMeta::Type::swapdims:
      return "swapdims";
    case ViewMeta::Type::unfold:
      return "unfold";
    case ViewMeta::Type::alias:
      return "alias";
    case ViewMeta::Type::view_as_real:
      return "view_as_real";
    case ViewMeta::Type::view_as_complex:
      return "view_as_complex";
    case ViewMeta::Type::_conj:
      return "_conj";
    case ViewMeta::Type::_neg_view:
      return "_neg_view";
    case ViewMeta::Type::chunk:
      return "chunk";
    case ViewMeta::Type::diagonal:
      return "diagonal";
    case ViewMeta::Type::diagonal_Dimname:
      return "diagonal_Dimname";
    case ViewMeta::Type::expand_as:
      return "expand_as";
    case ViewMeta::Type::narrow:
      return "narrow";
    case ViewMeta::Type::narrow_Tensor:
      return "narrow_Tensor";
    case ViewMeta::Type::movedim_intlist:
      return "movedim_intlist";
    case ViewMeta::Type::movedim_int:
      return "movedim_int";
    case ViewMeta::Type::reshape_as:
      return "reshape_as";
    case ViewMeta::Type::detach:
      return "detach";
    case ViewMeta::Type::transpose_int:
      return "transpose_int";
    case ViewMeta::Type::transpose_Dimname:
      return "transpose_Dimname";
    case ViewMeta::Type::unsqueeze:
      return "unsqueeze";
    case ViewMeta::Type::_values:
      return "_values";
    case ViewMeta::Type::indices:
      return "indices";
    case ViewMeta::Type::view:
      return "view";
    case ViewMeta::Type::view_dtype:
      return "view_dtype";
      default:
        return "UNKNOWN_VIEW_TYPE";
  }
}

std::ostream& operator<<(std::ostream& str, ViewMeta::Type rhs) {
  return str << ViewMetaTypeToString(rhs);
}

Alias::Alias(const at::Tensor& base) : base_(base.clone()) {
  auto impl = dynamic_cast<at::FunctionalTensorImplBase*>(base_.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(impl != nullptr);
  // Note [Marking Alias Tensors]
  // In the functionalization pass, calling a view operation on a tensor forks it off an Alias object
  // which is shared between the original tensor and the view.
  // The Alias object contains a base tensor, which is used to sync operations across the view tensors.
  // Throughout the Alias' life cylc,e we need to perform various views and reshapes on the base alias tensor.
  // We don't want that process to recursively fork off another Alias object though, so we explicitly mark
  // the base tensor as an alias.
  // Another option would have been to implement separate {view}_copy() kernels for every corresponding view op,
  // which have fallthrough kernels registered in the functionalization pass.
  impl->mark_as_alias();
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
  // TODO: Handle the other important view ops. This requires computing their inverses.
  at::Tensor t = update.new_val;
  for(int i = update.view_metas.size()-1; i >= 0; --i) {
    switch (update.view_metas[i].view_type) {
      case ViewMeta::Type::view:
          t = t.view(update.view_metas[i].source_size);
          break;
      case ViewMeta::Type::noOp:
          break;
      default:
          TORCH_CHECK(false, "Other types are not supported yet.");
    }
  }
  base_ = t.clone();
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
  // Only tensors that are opted into the functionalization pass can have a view meta.
  auto functional_impl = dynamic_cast<const at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
  return functional_impl != nullptr && functional_impl->is_view();
}

void Tensor::sync_() const {
  // Only tensors that are opted into the functionalization pass can be synced
  auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  functional_impl->sync_();
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
