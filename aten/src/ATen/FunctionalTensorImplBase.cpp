#include <ATen/FunctionalTensorImplBase.h>

#include <ATen/core/List.h>
#include <c10/util/Exception.h>

namespace at {

FunctionalTensorImplBase::FunctionalTensorImplBase(c10::DispatchKeySet keyset, caffe2::TypeMeta dtype, c10::Device device)
  : TensorImpl(
      keyset.add(DispatchKey::Functionalize),
      dtype,
      device
    ) {}

FunctionalTensorImplBase::FunctionalTensorImplBase(caffe2::TypeMeta dtype, c10::Device device)
  : TensorImpl(
      c10::DispatchKeySet(DispatchKey::Functionalize),
      dtype,
      device
    ) {}

void FunctionalTensorImplBase::mark_as_alias() {
  is_alias_tensor_ = true;
}

void FunctionalTensorImplBase::sync_() {
  if (is_up_to_date()) {
    return;
  }
  // Apply all updates on alias_
  alias_->SyncUpdateOperations();
  // Reapply views to Get the viewed tensor from updated base in alias_
  auto t = alias_->base();
  for (auto& view_meta: view_metas_) {
    switch (view_meta.view_type) {
      case ViewMeta::Type::view:
          t = t.view(view_meta.size);
          break;
      case ViewMeta::Type::noOp:
          break;
      default:
          TORCH_CHECK(false, "Tried to run the functionalization pass on an unsupported view: ", view_meta.view_type);
    }
  }
  // We want the new tensor to have separate memory from the alias.
  // Note that the clone required for functorch, but probably unnecessary for backends
  // like vulkan and xla, because their view operators already create fresh tensors.
  t = t.clone();
  // This call to replace_() doesn't go through the dispatcher.
  // It's a virtual method implemented directly on TensorImpl subclasses.
  replace_(t.unsafeGetTensorImpl());
  generation_ = alias_->generation();
}

bool FunctionalTensorImplBase::is_up_to_date() const {
  if (alias_) {
    return generation_ == alias_->generation();
  }
  return true;
}

void FunctionalTensorImplBase::maybe_add_update(const Tensor& updated_val) {
  // If the mutated tensor wasn't a view, we don't need to do anything
  if (is_view()) {
    alias_->add_update(updated_val.clone(), view_metas_);
    // TODO: add this back see if anything breaks?
    //generation_ = alias_->generation(); // self is fully up to date at this point
  }
}

void FunctionalTensorImplBase::set_view_meta(const Tensor& other, at::ViewMeta meta) {
    auto other_impl = dynamic_cast<at::FunctionalTensorImplBase*>(other.unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other_impl != nullptr);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_ == nullptr);
    if (other_impl->alias_ != nullptr) {
        // if the other tensor is already a view, copy its ViewMeta vector and push the current one.
        std::vector<at::ViewMeta> metas = other_impl->view_metas_;  // copy
        metas.push_back(meta);
        //alias_ = other_impl->alias_; // refcount bump
        alias_ = other_impl->alias_; // refcount bump
        view_metas_ = metas;
    } else {
        std::shared_ptr<Alias> alias = std::make_shared<Alias>(const_cast<Tensor&>(other));
        at::ViewMeta noop_view_info(at::ViewMeta::Type::noOp, other.sizes().vec(), other.sizes().vec());
        // The original tensor wasn't a view - turn it into a (no-op) view.
        other_impl->alias_ = alias;
        other_impl->view_metas_ = std::vector<at::ViewMeta>{noop_view_info};
        // Turn the new tensor into a view too
        alias_ = alias;
        view_metas_ = std::vector<at::ViewMeta>{meta};
    }
}

bool FunctionalTensorImplBase::is_view() const {
    return view_metas_.size() != 0;
}

const char* FunctionalTensorImplBase::tensorimpl_type_name() const {
    return "FunctionalTensorImplBase";
}

namespace functionalization {
namespace impl {

void maybe_add_update(Tensor& self) {
  auto functional_base_impl = dynamic_cast<at::FunctionalTensorImplBase*>(self.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_base_impl != nullptr);
  functional_base_impl->maybe_add_update(self);
}

void set_view_meta(const at::Tensor& out, const at::Tensor& t, ViewMeta meta) {
  auto out_impl = dynamic_cast<at::FunctionalTensorImplBase*>(out.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(out_impl != nullptr);
  out_impl->set_view_meta(t, meta);
}

void set_view_meta(const c10::List<at::Tensor>& outs, const at::Tensor& t, ViewMeta meta) {
  for (auto& out: outs.vec()) {
    auto out_impl = dynamic_cast<at::FunctionalTensorImplBase*>(out.unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(out_impl != nullptr);
    out_impl->set_view_meta(t, meta);
  }
}

bool is_alias_tensor(const Tensor& t) {
  auto t_impl = dynamic_cast<at::FunctionalTensorImplBase*>(t.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t_impl != nullptr);
  return t_impl->is_alias_tensor();
}


bool is_alias_tensor(const c10::List<Tensor>& t_list) {
  for (auto& t: t_list.vec()) {
    auto t_impl = dynamic_cast<at::FunctionalTensorImplBase*>(t.unsafeGetTensorImpl());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t_impl != nullptr);
    // There should be an invariant that either all or none of the tensors are alias tensors.
    // But we don't want to loop over the entire list.
    return t_impl->is_alias_tensor();
  }
  return false;
}

void maybe_sync(const Tensor& t) {
  if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
    // Unfortunately, we can't easily guarantee that wrapped numbers (scalar-tensors)
    // get wrapped up in a FunctionalTensorWrapper object, since they skip the dispatcher.
    // That shouldn't matter, since I don't think we're allowed to assign to wrapped numbers anyway.
    return;
  }
  auto functional_impl = dynamic_cast<at::FunctionalTensorImplBase*>(t.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  if (functional_impl->is_view() && !functional_impl->is_up_to_date()) {
    functional_impl->sync_();
  }
}
void maybe_sync(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    maybe_sync(*t);
  }
}
void maybe_sync(const c10::List<Tensor>& t_list) {
  for (auto& t: t_list.vec()) {
    maybe_sync(t);
  }
}
void maybe_sync(const at::TensorList t_list) {
  for (auto& t: t_list) {
    maybe_sync(t);
  }
}
void maybe_sync(const c10::List<c10::optional<Tensor>>& t_list) {
  for (auto& t: t_list.vec()) {
    maybe_sync(t);
  }
}

ViewMeta get_meta_view(const Tensor& self, IntArrayRef size) {
  return ViewMeta(ViewMeta::Type::view, /*size=*/size.vec(), /*source_size=*/self.sizes().vec());
}

} // namespace impl
} // namespace functionalization

} // namespace at
