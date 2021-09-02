
#include <ATen/FunctionalTensorWrapper.h>

#include <ATen/FunctionalInverses.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>

#include <c10/util/irange.h>

namespace at {

FunctionalTensorWrapper::FunctionalTensorWrapper(Tensor value)
  : c10::TensorImpl(
      c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(value.device(), value.numel(), value.dtype())),
      c10::DispatchKeySet(DispatchKey::Functionalize),
      value.dtype()
    ),
    value_(value),
    // Note: "level" is a concept that we don't know how to compute in core.
    // For now I'm retroactively setting this in functorch,
    // but once Open Multiple Dispatch lands we should be able to calculate this in core.
    level_(-1)
{
  // shallow_copy_from overwrites the storage and dispatch keyset...
  auto functional_storage = storage_;
  shallow_copy_from(value.getIntrusivePtr());
  storage_ = functional_storage;
  // TODO: should FunctionalTensorWrapper set_storage_access_should_throw()?
  // Their storage is similar to meta tensors, which does not throw.
  // However, users might expect to legally use storage() inside of a functionalized program, and this could be a foot gun.
  storage_access_should_throw_ = false;
  TORCH_INTERNAL_ASSERT(value_.defined());
  // TODO: test using the python key to trace a function that uses the functionalization pass.
  // I think I need this to make sure that replace_() calls hit the python key.
  key_set_ = (value_.key_set() & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::FuncTorchDynamicLayerBackMode))
     | c10::DispatchKeySet(c10::DispatchKey::Functionalize);
}

void FunctionalTensorWrapper::maybe_add_update() {
  auto storage_impl = static_cast<functionalization::FunctionalStorageImpl*>(unsafe_storage().unsafeGetStorageImpl());
  auto updated = storage_impl->maybe_add_update(value(), view_metas_);
  if (updated) {
    // At this point, we know that the current tensor is up to date with its alias.
    // Why? maybe_add_update() is only called after we've called an inplace operation on the tensor.
    // The pre-condition in the codegen is that we always sync tensor inputs before calling a mutation op.
    generation_ = storage_impl->generation();
  }
}
bool FunctionalTensorWrapper::is_aliased() const {
  return static_cast<functionalization::FunctionalStorageImpl*>(unsafe_storage().unsafeGetStorageImpl())->is_aliased();
}

bool FunctionalTensorWrapper::is_up_to_date() const {
  auto alias_generation = static_cast<functionalization::FunctionalStorageImpl*>(unsafe_storage().unsafeGetStorageImpl())->generation();
  return generation_ == alias_generation;
}

// Note [Functionalization: Alias Removal]
// When someone calls a view() op during the functionalization pass, we fork the current tensor into a view,
// and link this current tensor and the new one together to preserve the aliasing relationship.
//
// How do we do that?
//
// Every FunctionalTensorWrapper contains a dummy FunctionalTensorStorage, which subclasses from c10::StorageImpl.
// It doesn't contain any data (similar to MetaTensor storage), but it contains an Alias object that knows about the base tensor.
// Both the new and old tensor point to the same FunctionalTensorStorage.
//
// As mutations are applied to any of the views, we also queue each mutation up on the Alias object, so we can replay them.
// When the user requests a tensor that's had a view taken, we check if it's up to date.
// If it's not up to date, we first replay all of the queued up mutations onto the alias, and then re-apply the current view
// on top of the newly updated alias.
//
// Why do we queue up and lazily run mutations on the alias, instead of updating the alias eagerly?
// This behavior was taken from pytorch/xla, which the alias-removal logic was inspired from.
// One benefit of the laziness is that we save work in the cases where a user has multiple views and mutates one of them,
// but never uses the other views later in the program (in which case we'll never update the alias).
// It also has downsides though: repeatedly applying mutations to the same view withing syncing will slowly leak memory.
//
// Corresponding diagram because a picture is 1000 words:
//
// b = a.view(...)
//
//        a                                                    b
//        |                                                    |     If the user asks for b and it’s out of date,
//       \/                                                    \/    We regenerate b by replaying it’s views from the alias.
// . - - - - - - - - - - - - - .                    . - - - - - - - - - - - - - .
// |  FunctionalTensorWrapper  |                    |  FunctionalTensorWrapper  |
// . - - - - - - - - - - - - - .                    . - - - - - - - - - - - - - .
// |     value   |   storage   |                    |    storage    |   Value   |
// . - - - - - - - - - - - - - .                    . - - - - - - - - - - - - - .
//          |                   \                  /                      |
//          |                     \              /                        |
//          |                       . - - - - - - - - - - - - .           |
//          |                       |  FunctionalStorageImpl  |           |
//          |                       . - - - - - - - - - - - - .           |
//          |                       |         Alias           |           |
//          |                       . - - - - - - - - - - - - .           |
//          |                       /     mutations to a or b             |
//          |                     /       are queued onto Alias           |
//          |                   /                                         |
//         \/                 /                                           \/
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
// |        TensorImpl         |                             |             TensorImpl        |
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
// |   value   |   storage     |                             |    storage    |     Value     |
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
//          |                                                             |
//          |                                                             |
//          |                                                             |
//          |   In this picture the two tensor views their own storages,  |
//          |   have their own storages, but backends like functorch      |
//         \/   are allowed to re-alias underneath the pass               \/
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
// |    underyling_storage     |                             |      underyling_storage       |
// . - - - - - - - - - - - - - .                             . - - - - - - - - - - - - - - - .
//
// See Note [Functionalization: Alias Removal Part 2] for more details on the mutation replay logic.
void FunctionalTensorWrapper::set_view_meta(const Tensor& other, at::functionalization::ViewMeta meta) {
    TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(other));
    auto other_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(other);
    TORCH_INTERNAL_ASSERT(this != other_impl);
    TORCH_INTERNAL_ASSERT(!is_aliased())
    auto other_storage_impl = static_cast<functionalization::FunctionalStorageImpl*>(other.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl());
    if (!other_storage_impl->is_aliased()) {
        // The original tensor wasn't a view - turn it into a (no-op) view.
        other_storage_impl->set_alias(other_impl->value());
    }

    // Copy the original tensor's ViewMeta vector and push the current one.
    if (other_impl->view_metas_.size() > 0) {
        view_metas_ = other_impl->view_metas_;  // copy
    }
    view_metas_.push_back(meta);
    storage_ = other_impl->storage_; // Ensure that storages are aliased properly.
}

// See Note [Functionalization Pass - Inplace View Ops]
void FunctionalTensorWrapper::mutate_view_meta(at::functionalization::ViewMeta meta) {
    auto self_impl_storage = static_cast<functionalization::FunctionalStorageImpl*>(unsafe_storage().unsafeGetStorageImpl());
    if (!self_impl_storage->is_aliased()) {
        // The current tensor isn't a view - turn it into one.
        self_impl_storage->set_alias(value());
    }

    view_metas_.push_back(meta);
    sync_(/*force_sync=*/true);
}

// Note [Functionalization: Mutation Removal]
// Mutation removal is used to take a program like this:
//
// a.add_(b)
//
// and replace it with a slightly different program that has the same semantics:
//
// tmp = a.add(b)
// a.replace_(tmp)
//
// Where the replace_() call is implemented directly in the functionalization pass, so it is transparent to the backend.
// This is useful for backends that aren't able to handle certain types of mutations, like functorch.
//
// Why do we need to wrap every tensor in a FunctionalTensorWrapper? Consider this program:
//
// Before:
// tensor.add_(batched_tensor)
//
// After:
// tmp = tensor.add(batched_tensor)
// tensor.replace_(tmp)
//
// In the above, tmp is a batched tensor (because adding a normal tensor to a batched tensor does broadcasting and creates a batched tensor).
// But we can't just replace the underlying memory backing `tensor` with `tmp` - a batched tensor takes up more space!
// Instead, every input, intermediate and output of the program is wrapped in a FunctionalTensorImpl, which wraps the underlying tensor.
void FunctionalTensorWrapper::replace_(const Tensor& other) {
    // TODO: going to need to change this if we want nested functionalize() transforms.
    TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(other));
    value_ = other;
}


// Note [Functionalization Pass - Inplace View Ops]
// So, these ops are special - they're mutation AND view ops. They get special codegen.
// An example is transpose_, e.g. `a.transpose_()`
// Calling transpose_() should ensure that a gets an alias, and append the new ViewMeta to a's current list of ViewMetas.
// We also need to force a sync (even if a is already up to date), because a's underlying tensor hasn't actually
// been updated to reflect the new view yet.
void FunctionalTensorWrapper::sync_(bool force_sync) {
  auto storage_impl = static_cast<functionalization::FunctionalStorageImpl*>(unsafe_storage().unsafeGetStorageImpl());

  // Our view ops shouldn't re-enter the functionalization pass
  at::AutoDispatchBelowFunctionalize guard;
  if (!force_sync && (!storage_impl->is_aliased() || is_up_to_date())) {
    return;
  }
  // Apply all updates on alias_
  auto t = storage_impl->sync_update_operations();
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  // Reapply views to Get the viewed tensor from updated base in alias_
  for (auto& view_meta: view_metas_) {
    t = view_meta.forward_fn(t, view_meta.out_index);
  }
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  replace_(t);
  generation_ = storage_impl->generation();
}

const char* FunctionalTensorWrapper::tensorimpl_type_name() const {
    return "FunctionalTensorWrapper";
}

namespace functionalization {
namespace impl {

Tensor wrapFunctionalTensor(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isFunctionalTensor(tensor));
  return at::detail::make_tensor<FunctionalTensorWrapper>(tensor);
}
TensorList wrapFunctionalTensor(const c10::List<Tensor>& t_list) {
    std::vector<Tensor> outputs(t_list.size());
    for (const auto i : c10::irange(t_list.size())) {
        outputs[i] = wrapFunctionalTensor(t_list[i]);
    }
    return outputs;
}
std::vector<Tensor> wrapFunctionalTensor(const std::vector<Tensor>& t_list) {
    std::vector<Tensor> outputs(t_list.size());
    for (const auto i : c10::irange(t_list.size())) {
        outputs[i] = wrapFunctionalTensor(t_list[i]);
    }
    return outputs;
}
TensorList wrapFunctionalTensor(const TensorList& t_list) {
    std::vector<Tensor> outputs(t_list.size());
    for (const auto i : c10::irange(t_list.size())) {
        outputs[i] = wrapFunctionalTensor(t_list[i]);
    }
    return outputs;
}

Tensor unwrapFunctionalTensor(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(tensor));
  auto impl = unsafeGetFunctionalWrapper(tensor);
  return impl->value();
}
c10::optional<Tensor> unwrapFunctionalTensor(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    return c10::make_optional<Tensor>(unwrapFunctionalTensor(*t));
  }
  return c10::nullopt;
}
c10::List<Tensor> unwrapFunctionalTensor(const c10::List<Tensor> t_list) {
  c10::List<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs[i] = unwrapFunctionalTensor(t_list[i]);
  }
  return outputs;
}
c10::List<c10::optional<Tensor>> unwrapFunctionalTensor(const c10::List<c10::optional<Tensor>> t_list) {
  c10::List<c10::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs[i] = unwrapFunctionalTensor(t_list[i]);
  }
  return outputs;
}
TensorList unwrapFunctionalTensor(const TensorList& t_list) {
    std::vector<Tensor> outputs(t_list.size());
    for (const auto i : c10::irange(t_list.size())) {
        outputs[i] = unwrapFunctionalTensor(t_list[i]);
    }
    return outputs;
}

void sync(const Tensor& t) {
  if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
    // Unfortunately, we can't easily guarantee that wrapped numbers (scalar-tensors)
    // get wrapped up in a FunctionalTensorWrapper object, since they skip the dispatcher.
    // That shouldn't matter, since I don't think we're allowed to assign to wrapped numbers anyway.
    return;
  }
  // Not every tensor that hits a functionalization kernel is necessarily a functional tensor.
  // For example, xla_tensor.copy_(cpu_tensor) needs to hit the functionalization kernel
  // to sync xla_tensor, but not cpu_tensor.
  if (!at::functionalization::impl::isFunctionalTensor(t)) {
      return;
  }
  auto functional_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
  functional_impl->sync_();
}
void sync(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    sync(*t);
  }
}
void sync(const c10::List<Tensor> t_list) {
  for (const auto i : c10::irange(t_list.size())) {
    sync(t_list[i]);
  }
}
void sync(const at::TensorList t_list) {
  for (auto t: t_list) {
    sync(t);
  }
}
void sync(const c10::List<c10::optional<Tensor>> t_list) {
  for (const auto i : c10::irange(t_list.size())) {
    sync(t_list[i]);
  }
}

void maybe_add_update(Tensor& self) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  auto functional_base_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  functional_base_impl->maybe_add_update();
}

void set_view_meta(const at::Tensor& out, const at::Tensor& t, functionalization::ViewMeta meta, int64_t out_idx) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(out));
  auto out_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(out);
  if (out_idx != 0) {
      // Note [out_idx in ViewMeta]
      // When a view op outputs multiple tensors, each output needs its own separate ViewMeta.
      // Each ViewMeta also tracks the index of the particular output tensor, which is needed in the reverse function.
      out_impl->set_view_meta(t, functionalization::ViewMeta(meta.forward_fn, meta.reverse_fn, out_idx));
  } else {
      out_impl->set_view_meta(t, meta);
  }
}

void set_view_meta(const c10::List<at::Tensor> outs, const at::Tensor& t, functionalization::ViewMeta meta) {
  for (const auto i : c10::irange(outs.size())) {
    set_view_meta(outs[i], t, meta, i);
  }
}

void set_view_meta(const std::vector<at::Tensor> outs, const at::Tensor& t, functionalization::ViewMeta meta) {
  for (const auto i : c10::irange(outs.size())) {
    set_view_meta(outs[i], t, meta, i);
  }
}

void mutate_view_meta(const at::Tensor& self, functionalization::ViewMeta meta) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  self_impl->mutate_view_meta(meta);
}

// See Note [Propagating strides in the functionalization pass]
void set_strides(const Tensor& out, const Tensor& reference_out) {
  out.unsafeGetTensorImpl()->set_sizes_and_strides(reference_out.sizes(), reference_out.strides());
  out.unsafeGetTensorImpl()->set_storage_offset(reference_out.storage_offset());
}

void set_strides(const std::vector<Tensor>& outs, const std::vector<Tensor>& reference_outs) {
  TORCH_INTERNAL_ASSERT(outs.size() == reference_outs.size());
  for (const auto i : c10::irange(reference_outs.size())) {
    set_strides(outs[i], reference_outs[i]);
  }
}


} // namespace impl
} // namespace functionalization
} // namespace at
