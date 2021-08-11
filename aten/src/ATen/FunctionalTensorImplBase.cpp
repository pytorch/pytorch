#include <ATen/FunctionalTensorImplBase.h>

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
  replace_(t);
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
    generation_ = alias_->generation(); // self is fully up to date at this point
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
    return alias_ != nullptr;
}

const char* FunctionalTensorImplBase::tensorimpl_type_name() const {
    return "FunctionalTensorImplBase";
}
} // namespace at
