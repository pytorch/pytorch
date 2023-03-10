
#include <ATen/FunctionalTensorWrapper.h>

#include <ATen/FunctionalInverses.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_propagate_xla_data.h>
#include <ATen/ops/_to_copy.h>
#endif

namespace at {

void FunctionalTensorWrapper::set_constructor_metadata() {
  TORCH_INTERNAL_ASSERT(value_.defined());
  // Note: "level" is a concept that we don't know how to compute in core.
  // For now I'm retroactively setting this in functorch,
  // but once Open Multiple Dispatch lands we should be able to calculate this in core.
  level_ = -1;
  // mirror all of the generic tensor metadata onto the wrapper
  copy_generic_tensor_metadata(value_.getIntrusivePtr().get(), this);
  refresh_numel();
  refresh_contiguous();
  storage_access_should_throw_ = false;
  // In general, the sizes/stride metadata on a tensor can change as it is mutated,
  // and these changes need to be reflected in the metadata of the wrapper.
  set_allow_tensor_metadata_change(true);
  key_set_ = c10::DispatchKeySet(c10::DispatchKey::Functionalize) | value_.key_set();
  // All of the keys corresponding to functorch transforms should not be copied over.
  // Functorch transforms all have their own wrapper tensors (e.g. BatchedTensorImpl) which expect
  // to participate in the functorch transforms.
  key_set_ = key_set_ - c10::functorch_transforms_ks - c10::python_ks;
  // We override a bunch of _custom(), so make sure they get called
  // TODO: metadata copying may not actually be necessary then
  set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);
  set_custom_device(true);
}

FunctionalTensorWrapper::FunctionalTensorWrapper(const Tensor& value)
  : c10::TensorImpl(
      c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(value)),
      c10::DispatchKeySet(DispatchKey::Functionalize) | value.key_set(),
      value.dtype()
    ),
    value_(value)
{
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(value_));
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  set_constructor_metadata();
}

void FunctionalTensorWrapper::freeze_storage() const {
  functional_storage_impl()->freeze();
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
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(value_));
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  set_constructor_metadata();
  // Copy the original tensor's ViewMeta vector and push the current one.
  if (!base->view_metas_.empty()) {
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
  // As an optimization, we used to mark the tensor here as "up-to-date",
  // That way, code like:
  //   x = torch.ones(1'000'000)
  //   x[0].add_(1)
  // doesn't result in an unnecessary materialization of the base.
  // This optimization results in the slice temporarily haven't incorrect
  // stride/storage_offset though, and DCE should handle that optimization anyway.
  // generation_ = storage_impl->generation();
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
  at::AutoDispatchSkipFunctionalize guard;
  value_ = meta.forward_fn(value_, meta.out_index);
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
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
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  // out= ops are allowed to resize the output tensors, mutating both the data and metadata of the tensor.
  // We need to propagate that metadata mutation to the wrapper (new size).
  auto sizes_ = value_.sym_sizes();
  auto strides_ = value_.sym_strides();
  auto storage_offset_ = value_.sym_storage_offset();
  set_sizes_and_strides(sizes_, strides_, storage_offset_);
  if (dtype() != value_.unsafeGetTensorImpl()->dtype() || layout() != value_.unsafeGetTensorImpl()->layout()) {
    // .to() should not re-entrantly go through functionalization.
    at::AutoDispatchSkipFunctionalize guard;
    // and we want _to_copy() to show up in the graph, not the composite .to() operator
    // (this can happen if autograd has already run by the time we enter this code)
    value_ = at::_to_copy(value_, c10::TensorOptions().dtype(dtype()).layout(layout()));
    TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  }
}

void FunctionalTensorWrapper::maybe_replace_storage(const Tensor& other) {
  // Note [resize_() in functionalization pass]
  // resize_() is a special operator in functionalization because it can reallocate its underlying storage.
  // This function is only ever called in the case that resize_() needs to reallocate its storage to a larger size.
  //
  // However, functionalization currently bans the following code:
  //   a = torch.ones(2)
  //   b = a.view(2)
  //   b.resize_(4) # b is a view tensor, that we are trying to increase the storage size of
  //
  // Why is this code difficult to handle?
  // The functionalization pass currently keeps aliases in sync by making the following assumptions:
  // - The “base” tensor always refers to “all of the data”
  // - Whenever you have b = view_op(a), “b” should always refer to a subset of “a”s memory.
  //
  // The code above breaks that assumption b.resize_(4) actually needs to update "a"
  // to tell it that it is now actually some slice of a pre-existing larger storage.
  // We're also no longer re-generate "b" fully from "a" anymore, since "a" refers to a slice of "b"'s data.
  //
  // This is probably fixable in theory, but:
  // - the fix would likey complicated the functionalization logic quite a bit.
  // - the primary use case for resize_() today is resizing zero-sized tensors in out= variants of operators
  // - resize_() also can give you weird results today if you try to resize_() a weirdly strided tensor.
  //
  // Given all of the above, for now we're just banning the above usage.
  TORCH_CHECK(storage().use_count() == 1, "Attempted to resize a view tensor to a larger size. This is not allowed in the functionalization pass");
  TORCH_CHECK(view_metas_.empty(), "Attempted to resize a view tensor to a larger size. This is not allowed in the functionalization pass");
  // If this tensor is not a view (and has no outstanding views taken out on it),
  // Then it's safe to throw out the old storage and replace it with the new, larger one.
  storage_ = c10::Storage(c10::make_intrusive<functionalization::FunctionalStorageImpl>(other));
  value_ = other;
  TORCH_INTERNAL_ASSERT(!value_.key_set().has(c10::DispatchKey::Functionalize));
  generation_ = 0;
  // And update the metadata on the wrapper to reflect the new sizes and strides
  set_sizes_and_strides(value_.sizes(), value_.strides());
  refresh_numel();
  // (Technically we should be guaranteed that the tensor was already contiguous,
  // since it's guaranteed not to have been a view. Doesnt hurt to run though)
  refresh_contiguous();
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

bool FunctionalTensorWrapper::apply_updates() {
  // Apply all updates on alias_
  auto storage_impl = functional_storage_impl();
  return storage_impl->apply_updates();
}

const char* FunctionalTensorWrapper::tensorimpl_type_name() const {
    return "FunctionalTensorWrapper";
}

template <typename VariableVersion>
c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach_core(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  if (key_set_.has(DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    auto r = pyobj_slot_.load_pyobj_interpreter()->detach(this);
    if (r) {
      r->set_version_counter(std::forward<VariableVersion>(version_counter));
      r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      return r;
    }
  }

  auto impl = c10::make_intrusive<FunctionalTensorWrapper>(value_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::forward<VariableVersion>(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->level_ = level_;
  impl->generation_ = generation_;
  impl->view_metas_ = view_metas_;
  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

c10::intrusive_ptr<TensorImpl> FunctionalTensorWrapper::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      std::move(version_counter), allow_tensor_metadata_change);
}

c10::Device FunctionalTensorWrapper::device_custom() const {
  return value_.unsafeGetTensorImpl()->device();
}
at::IntArrayRef FunctionalTensorWrapper::sizes_custom() const {
  return value_.unsafeGetTensorImpl()->sizes();
}
at::IntArrayRef FunctionalTensorWrapper::strides_custom() const {
  return value_.unsafeGetTensorImpl()->strides();
}
int64_t FunctionalTensorWrapper::dim_custom() const {
  return value_.unsafeGetTensorImpl()->dim();
}
int64_t FunctionalTensorWrapper::numel_custom() const {
  return value_.unsafeGetTensorImpl()->numel();
}
bool FunctionalTensorWrapper::is_contiguous_custom(at::MemoryFormat memory_format) const {
  return value_.unsafeGetTensorImpl()->is_contiguous(memory_format);
}
c10::SymIntArrayRef FunctionalTensorWrapper::sym_sizes_custom() const {
  return value_.unsafeGetTensorImpl()->sym_sizes();
}
c10::SymIntArrayRef FunctionalTensorWrapper::sym_strides_custom() const {
  return value_.unsafeGetTensorImpl()->sym_strides();
}
c10::SymInt FunctionalTensorWrapper::sym_size_custom(int64_t d) const {
  return value_.unsafeGetTensorImpl()->sym_size(d);
}
c10::SymInt FunctionalTensorWrapper::sym_storage_offset_custom() const {
  return value_.unsafeGetTensorImpl()->sym_storage_offset();
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
c10::optional<Tensor> to_functional_tensor(const c10::optional<Tensor>& tensor) {
  if (tensor.has_value()) {
    return c10::make_optional<Tensor>(to_functional_tensor(*tensor));
  }
  return c10::nullopt;
}
c10::List<c10::optional<Tensor>> to_functional_tensor(const c10::List<c10::optional<Tensor>>& t_list) {
  c10::List<c10::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(to_functional_tensor(t_list[i]));
  }
  return outputs;
}
std::vector<Tensor> to_functional_tensor(ITensorListRef t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    outputs.push_back(to_functional_tensor(tensor));
  }
  return outputs;
}

Tensor from_functional_tensor(const Tensor& tensor, bool assert_functional) {
  // Note [Wrapped Numbers <> Functionalization]
  if (!tensor.defined() || tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      return tensor;
  }
  if (isFunctionalTensor(tensor)) {
    auto impl = unsafeGetFunctionalWrapper(tensor);
    return impl->value();
  } else {
    // If the current tensor is not functional, then raise an error
    // if assert_functional is true. Otherwise, return the input.
    TORCH_INTERNAL_ASSERT(!assert_functional)
    return tensor;
  }
}
c10::optional<Tensor> from_functional_tensor(const c10::optional<Tensor>& t, bool assert_functional) {
  if (t.has_value()) {
    return c10::make_optional<Tensor>(from_functional_tensor(*t, assert_functional));
  }
  return c10::nullopt;
}
std::vector<Tensor> from_functional_tensor(ITensorListRef t_list) {
  std::vector<Tensor> outputs;
  outputs.reserve(t_list.size());
  for (const auto& tensor : t_list) {
    // from_functional_tensor(Tensor) has asserts to make sure you don't accidentally call
    // it on a non-functional input,
    // but from_functional_tensor(TensorList) can recieve a list containing both
    // functional and non-functional tensors.
    // Example of when that can happen: torch.cat(function_input_tensor, global_state_tensor).
    // When that happens, we're okay with only unwrapping the functional tensors.
    outputs.push_back(from_functional_tensor(tensor, /*assert_functional=*/false));
  }
  return outputs;
}
c10::List<c10::optional<Tensor>> from_functional_tensor(const c10::List<c10::optional<Tensor>>& t_list) {
  c10::List<c10::optional<Tensor>> outputs;
  outputs.reserve(t_list.size());
  for (const auto i : c10::irange(t_list.size())) {
    outputs.push_back(from_functional_tensor(t_list[i], /*assert_functional=*/false));
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
void sync(ITensorListRef t_list) {
  for (const auto& t : t_list) {
    sync(t);
  }
}
void sync(const c10::List<c10::optional<Tensor>>& t_list) {
  for (const auto i : c10::irange(t_list.size())) {
    sync(t_list[i]);
  }
}

void replace_(const Tensor& functional_tensor, const Tensor& other) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  unsafeGetFunctionalWrapper(functional_tensor)->replace_(other);
}

void replace_(const ITensorListRef functional_tensor, ITensorListRef other) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_tensor.size() == other.size());
  auto functional_tensor_it = functional_tensor.begin();
  auto other_it = other.begin();
  for (const auto i : c10::irange(functional_tensor.size())) {
    (void)i; // Suppress unused variable warning
    replace_(*functional_tensor_it++, *other_it++);
  }
}

void propagate_xla_data(const Tensor& functional_tensor, const Tensor& other) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  if (functional_tensor.key_set().has(c10::DispatchKey::XLA)) {
    at::_propagate_xla_data(at::functionalization::impl::unsafeGetFunctionalWrapper(functional_tensor)
        ->value(), other);
  }
}

void propagate_xla_data(const ITensorListRef functional_tensor, ITensorListRef other) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_tensor.size() == other.size());
  auto functional_tensor_it = functional_tensor.begin();
  auto other_it = other.begin();
  for (const auto i : c10::irange(functional_tensor.size())) {
    (void)i; // Suppress unused variable warning
    propagate_xla_data(*functional_tensor_it++, *other_it++);
  }
}

void commit_update(const Tensor& functional_tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isFunctionalTensor(functional_tensor));
  unsafeGetFunctionalWrapper(functional_tensor)->commit_update();
}

void commit_update(ITensorListRef functional_tensor) {
  for (const auto& t : functional_tensor) {
    commit_update(t);
  }
}

bool isFunctionalTensor(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Functionalize);
}

bool isFunctionalTensor(const c10::optional<Tensor>& t) {
  if (t.has_value()) {
    return isFunctionalTensor(*t);
  } else {
    return false;
  }
}

bool isFunctionalTensor(const c10::List<c10::optional<Tensor>>& t_list) {
  if (t_list.empty()) return false;
  auto functional_count = 0;
  for (const auto i : c10::irange(t_list.size())) {
    if (!t_list[i].has_value() || !t_list[i]->defined()) continue;
    if (isFunctionalTensor(t_list[i])) {
      ++functional_count;
    }
  }
  return functional_count > 0;
}

template <typename T>
bool isFunctionalTensorIListRef(c10::IListRef<T> list) {
  if (list.size() == 0) return false;
  auto functional_count = 0;
  for (const auto& tensor : list) {
    if (!tensor.defined()) continue;
    if (isFunctionalTensor(tensor)) {
      ++functional_count;
    }
  }
  return functional_count > 0;
}

bool isFunctionalTensor(ITensorListRef list) {
  return isFunctionalTensorIListRef(list);
}

void freeze_functional_tensor(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(tensor));
  auto functional_base_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
  functional_base_impl->freeze_storage();
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

std::vector<Tensor> create_functional_tensor_with_view_meta(ITensorListRef view_to_wrap, const at::Tensor& base, functionalization::ViewMeta meta) {
  std::vector<Tensor> outputs(view_to_wrap.size());
  int64_t i = 0;
  for (const auto& tensor : view_to_wrap) {
    outputs[i] = create_functional_tensor_with_view_meta(tensor, base, meta, i);
    i++;
  }
  return outputs;
}

void mutate_view_meta(const at::Tensor& self, functionalization::ViewMeta meta) {
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self));
  auto self_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(self);
  self_impl->mutate_view_meta(std::move(meta));
}

// Note [Propagating strides in the functionalization pass]
// In order to properly compute stride information, the functionalization pass
// calls each {view} reference implementations with meta tensors.
// The output meta tensor's stride info serves as a reference for what the correct strides should be.
void set_sizes_strides_offset(const Tensor& out, const Tensor& reference_out) {
  out.unsafeGetTensorImpl()->set_sizes_and_strides(reference_out.sym_sizes(), reference_out.sym_strides(), reference_out.sym_storage_offset());
}

void set_sizes_strides_offset(const std::vector<Tensor>& outs, const std::vector<Tensor>& reference_outs) {
  TORCH_INTERNAL_ASSERT(outs.size() == reference_outs.size());
  for (const auto i : c10::irange(reference_outs.size())) {
    set_sizes_strides_offset(outs[i], reference_outs[i]);
  }
}

thread_local bool _functionalizationReapplyViews;

bool getFunctionalizationReapplyViewsTLS() {
  return _functionalizationReapplyViews;
}
void setFunctionalizationReapplyViewsTLS(bool reapply_views) {
  _functionalizationReapplyViews = reapply_views;
}

} // namespace impl


// Given an **out-of-place** op that might internally call view/inplace ops,
// This function will "functionalize" it.
// That is, it will call the operator, but removing any intermediate views/mutations
// that are performed inside of it.
// This is useful for LTC/XLA, which would like to re-use some of our composite kernels
// from pytorch core but not have to worry about the view ops that they might call.
// e.g. at::block_diag
void functionalize_op_helper(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;
  auto arguments = torch::jit::last(stack, num_arguments);

  // Wrap all tensor-like inputs into FunctionalTensorWrappers.
  // When we re-invoke the dispatcher, this will automatically enable the functionalization pass.
  for (uint64_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (t.defined()) {
        TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t),
          "The composite op functionalization fallback expects its inputs all not to be functional tensors");
        auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(t));
        (*stack)[arguments_begin + idx] = t_new;
      }
    } else if (ivalue.isTensorList()) {
      auto tensors = ivalue.toTensorList();
      TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(tensors),
        "The composite op functionalization fallback expects its inputs all not to be functional tensors");
      auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(tensors));
      (*stack)[arguments_begin + idx] = t_new;
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList();
      TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(opt_tensors),
        "The composite op functionalization fallback expects its inputs all not to be functional tensors");
      auto t_new = c10::IValue(at::functionalization::impl::to_functional_tensor(opt_tensors));
      (*stack)[arguments_begin + idx] = t_new;
    }
  }

  {
    // Today when you call at::empty(device=lazy), the lazy backend decides whether or not to wrap
    // the output in a functional tensor based on TLS.
    // In this code, we're re-entrantly entering functionalization in the same call-stack,
    // so we need to manually fix up TLS as if it hadn't already been called.
    auto curr_tls = c10::impl::tls_local_dispatch_key_set();
    auto tls_reenable_functionalize = c10::impl::PODLocalDispatchKeySet();
    tls_reenable_functionalize.set_included(curr_tls.included_);
    tls_reenable_functionalize.set_excluded(curr_tls.excluded_.remove(c10::DispatchKey::Functionalize));
    c10::impl::ForceDispatchKeyGuard guard_(tls_reenable_functionalize);
    // So, we should probably provide a way to directly call a kernel registered to
    // the `CompositeExplicitAutograd` key.
    // We can't do that today, so this should be a reasonably good proxy
    // (It won't work in cases where an op has both a CompositeExplicitAutograd kernel
    // AND a dedicated meta kernel, but that probably shouldn't ever happen).
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::Meta), stack);
  }

  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  auto returns = torch::jit::last(stack, num_returns);

  for (const auto idx : c10::irange(num_returns)) {
    const auto& ivalue = returns[idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (!t.defined()) continue;
      at::functionalization::impl::sync(t);
      auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(t));
      (*stack)[returns_begin + idx] = t_new;
    } else if (ivalue.isTensorList()) {
      auto tensors = ivalue.toTensorList();
      at::functionalization::impl::sync(tensors);
      auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(tensors));
      (*stack)[returns_begin + idx] = t_new;
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList();
      at::functionalization::impl::sync(opt_tensors);
      auto t_new = c10::IValue(at::functionalization::impl::from_functional_tensor(opt_tensors));
      (*stack)[returns_begin + idx] = t_new;
    }
  }
}



} // namespace functionalization
} // namespace at
