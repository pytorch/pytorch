
#include <ATen/FunctionalTensorWrapper.h>

#include <ATen/FunctionalInverses.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>

#include <c10/util/irange.h>

namespace at {

void FunctionalTensorWrapper::set_constructor_metadata() {
  TORCH_INTERNAL_ASSERT(value_.defined());
  // Note: "level" is a concept that we don't know how to compute in core.
  // For now I'm retroactively setting this in functorch,
  // but once Open Multiple Dispatch lands we should be able to calculate this in core.
  level_ = -1;
  // shallow_copy_from overwrites the storage and dispatch keyset...
  auto functional_storage = storage_;
  shallow_copy_from(value_.getIntrusivePtr());
  storage_ = functional_storage;
  storage_access_should_throw_ = false;
  key_set_ = c10::DispatchKeySet(c10::DispatchKey::Functionalize) | value_.key_set();
}

FunctionalTensorWrapper::FunctionalTensorWrapper(const Tensor& value)
  : c10::TensorImpl(
      c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(value)),
      c10::DispatchKeySet(DispatchKey::Functionalize) | value.key_set(),
      value.dtype()
    ),
    value_(value)
{
  set_constructor_metadata();
}

// Note [Functionalization: Alias Removal]
// When someone calls a view() op during the functionalization pass, e.g. 'b = a.view(...)',
// we link `b` and `a` to a shared Alias object to preserve the aliasing relationship.
//
// How do we do that?
//
// Every FunctionalTensorWrapper contains a dummy FunctionalStorageImpl, which subclasses from c10::StorageImpl.
// It doesn't contain any data (similar to MetaTensor storage), but it contains an Alias object that knows about the base tensor.
// When a tensor is created through a view operation, both the new and old tensor point to the same FunctionalStorageImpl.
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
// It also has downsides though: repeatedly applying mutations to the same view without syncing
// will silently use up more and more memory as more mutations are queued up.
//
// Corresponding diagram:
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
// This constructor is only used by view ops.
// - view_value: The output tensor that we need to wrap.
// - base: The "base" of the view that `view_value` was generated from.
// See Note [Functionalization: Alias Removal Part 2] for more details on the mutation replay logic.
FunctionalTensorWrapper::FunctionalTensorWrapper(const Tensor& view_value, const FunctionalTensorWrapper* base, functionalization::ViewMeta meta)
  : c10::TensorImpl(
      c10::DispatchKeySet(DispatchKey::Functionalize),
      view_value.dtype(),
      view_value.device()
    ),
    value_(view_value)
{
  set_constructor_metadata();
  // Copy the original tensor's ViewMeta vector and push the current one.
  if (base->view_metas_.size() > 0) {
      view_metas_ = base->view_metas_;  // copy
  }
  view_metas_.push_back(meta);
  storage_ = base->storage_; // alias this tensor's storage with the base tensor's
}

functionalization::FunctionalStorageImpl* FunctionalTensorWrapper::functional_storage_impl() const {
  return static_cast<functionalization::FunctionalStorageImpl*>(storage_.unsafeGetStorageImpl());
}

void FunctionalTensorWrapper::commit_update() {
  auto storage_impl = functional_storage_impl();
  storage_impl->add_update(value_, view_metas_);
  // Invariant: commit_update() is called during an inplace operation.
  // Tensor inputs to the operation are synced before runnig the op,
  // so the current tensor must be up-to-date with its alias at this point.
  generation_ = storage_impl->generation();
}

bool FunctionalTensorWrapper::is_up_to_date() const {
  auto alias_generation = functional_storage_impl()->generation();
  return generation_ == alias_generation;
}

// See Note [Functionalization Pass - Inplace View Ops]
void FunctionalTensorWrapper::mutate_view_meta(at::functionalization::ViewMeta meta) {
  view_metas_.push_back(meta);
  // Note [Functionalization Pass - Inplace View Ops]
  // So, these ops are special - they're mutation AND view ops. They get special codegen.
  // An example is transpose_, e.g. `a.transpose_()`
  // Calling transpose_() should ensure that a gets an alias, and append the new ViewMeta to a's current list of ViewMetas.
  // We also need to force a sync (even if a is already up to date), because a's underlying tensor hasn't actually
  // been updated to reflect the new view yet.
  regenerate_from_base();
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


void FunctionalTensorWrapper::sync_() {
  if (is_up_to_date()) {
    return;
  }
  apply_updates();
  regenerate_from_base();
}

void FunctionalTensorWrapper::regenerate_from_base() {
  at::AutoDispatchSkipFunctionalize guard;
  auto storage_impl = functional_storage_impl();
  auto t = storage_impl->base();
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  // Reapply views to get the viewed tensor from the base in alias_
  for (auto& view_meta: view_metas_) {
    t = view_meta.forward_fn(t, view_meta.out_index);
  }
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  replace_(t);
  generation_ = storage_impl->generation();
}

void FunctionalTensorWrapper::apply_updates() {
  // Apply all updates on alias_
  auto storage_impl = functional_storage_impl();
  storage_impl->apply_updates();
}

const char* FunctionalTensorWrapper::tensorimpl_type_name() const {
    return "FunctionalTensorWrapper";
}

namespace functionalization {
namespace impl {

Tensor to_functional_tensor(const Tensor& tensor) {
  // Note [Wrapped Numbers <> Functionalization]
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      return tensor;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isFunctionalTensor(tensor));
  return at::detail::make_tensor<FunctionalTensorWrapper>(tensor);
}
std::vector<Tensor> to_functional_tensor(ITensorList t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_functional_tensor(t_list[i]));
  }
  return outputs;
}

Tensor from_functional_tensor(const Tensor& tensor) {
  // Note [Wrapped Numbers <> Functionalization]
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      return tensor;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(tensor));
  auto impl = unsafeGetFunctionalWrapper(tensor);
  return impl->value();
}
c10::optional<Tensor> from_functional_tensor(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    return c10::make_optional<Tensor>(from_functional_tensor(*t));
  }
  return c10::nullopt;
}
std::vector<Tensor> from_functional_tensor(ITensorList t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(from_functional_tensor(t_list[i]));
  }
  return outputs;
}
c10::List<c10::optional<Tensor>> from_functional_tensor(const c10::List<c10::optional<Tensor>>& t_list) {
  c10::List<c10::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(from_functional_tensor(t_list[i]));
  }
  return outputs;
}
std::vector<at::OptionalTensorRef> from_functional_tensor(IOptTensorRefList t_list) {
  std::vector<at::OptionalTensorRef> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    auto opt = (t_list[i].has_value()) ?
        from_functional_tensor(*t_list[i]) : OptionalTensorRef{};
    outputs.push_back(opt);
  }
  return outputs;
}

void sync(const Tensor& t) {
  if (t.unsafeGetTensorImpl()->is_wrapped_number()) {
    // Note [Wrapped Numbers <> Functionalization]
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
void sync(ITensorList t_list) {
  for (const auto& t : t_list) {
    sync(t);
  }
}
void sync(IOptTensorRefList t_list) {
  for (const auto& t : t_list) {
    if (t.has_value()) {
      sync(*t);
    }
  }
}


Tensor create_functional_tensor_with_view_meta(const at::Tensor& view_to_wrap, const at::Tensor& base, functionalization::ViewMeta meta, int64_t out_idx) {
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(view_to_wrap));
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(base));
  auto functional_base_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(base);
  if (out_idx != 0) {
    // Note [out_idx in ViewMeta]
    // When a view op outputs multiple tensors, each output needs its own separate ViewMeta.
    // Each ViewMeta also tracks the index of the particular output tensor, which is needed in the reverse function.
    meta = meta.to_out_idx(out_idx);
  }
  return at::detail::make_tensor<FunctionalTensorWrapper>(view_to_wrap, functional_base_impl, meta);
}

std::vector<Tensor> create_functional_tensor_with_view_meta(ITensorList view_to_wrap, const at::Tensor& base, functionalization::ViewMeta meta) {
  std::vector<Tensor> outputs(view_to_wrap.size());
  for (const auto i : c10::irange(view_to_wrap.size())) {
    outputs[i] = create_functional_tensor_with_view_meta(view_to_wrap[i], base, meta, i);
  }
  return outputs;
}

void mutate_view_meta(const at::Tensor& self, functionalization::ViewMeta meta) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  self_impl->mutate_view_meta(meta);
}

// Note [Propagating strides in the functionalization pass]
// In order to properly compute stride information, the functionalization pass
// calls each {view} reference implementations with meta tensors.
// The output meta tensor's stride info serves as a reference for what the correct strides should be.
void set_sizes_strides_offset(const Tensor& out, const Tensor& reference_out) {
  out.unsafeGetTensorImpl()->set_sizes_and_strides(reference_out.sizes(), reference_out.strides());
  out.unsafeGetTensorImpl()->set_storage_offset(reference_out.storage_offset());
}

void set_sizes_strides_offset(const std::vector<Tensor>& outs, const std::vector<Tensor>& reference_outs) {
  TORCH_INTERNAL_ASSERT(outs.size() == reference_outs.size());
  for (const auto i : c10::irange(reference_outs.size())) {
    set_sizes_strides_offset(outs[i], reference_outs[i]);
  }
}


} // namespace impl
} // namespace functionalization
} // namespace at
