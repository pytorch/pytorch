
#pragma once

#include <ATen/ArrayRef.h>
#include <ATen/core/IList.h>
#include <ATen/FunctionalStorageImpl.h>

#include <c10/core/DispatchKey.h>

namespace at {

// Note [Functionalization Pass In Core]
// The Functionalization pass is used to remove aliasing from a pytorch program.
//
// This is useful for backends that don't support aliasing, like XLA and Vulkan.
// It's also necessary in order to remove mutation from a program, which is needed in Functorch.
//
// Consider this program:
// a = torch.ones(...)
// b = a.view(...)
// b.add_(1)
//
// In this program, b is meant to alias with a due to the use of view(). At the end of the program, both a and b are full of 2's.
// However, backends that don't support aliasing aren't able to correctly implement the view() operator.
// Instead, they can opt into the Functionalization pass, which will sit between the user and the backend,
// and provide the necessary aliasing logic.
//
// The functionalization pass will turn the above program into a slightly different program that has the same semantics,
// transparently to the user, that backends like XLA/Vulkan are able to implement
// a = torch.ones(...)
// b = a.view_copy(...)  # view() replaced with view_copy(). Backends like XLA/Vulkan can implement this!
// b.add_(1)
// a.add_(1)  # Our functionalization pass machinery knows that a and b are aliased - it applies b's mutation to a too.
//
// So, how does the functionalization pass keep track of which tensors are aliased?
// The pass works by wrapping EVERY tensor in the program inside of a FunctionalTensorWrapper, which knows about its alias'd tensors.
//
// See Note [Functionalization: Alias Removal] for details on the aliasing machinery.
// See Note [Functionalization: Mutation Removal] for details on mutation removal.

struct TORCH_API FunctionalTensorWrapper : public c10::TensorImpl {
  explicit FunctionalTensorWrapper(const Tensor& value);
  // Additional constructor to create a FunctionalTensorWrapper directly from an underlying tensor that was created from a view.
  // For example, the code b = a.view1() will generate a constructor call to FunctionalTensorWrapper(b, a, view1_meta)
  explicit FunctionalTensorWrapper(const Tensor& view_value, const FunctionalTensorWrapper* base, functionalization::ViewMeta meta);

  // Get the underlying, actual tensor, that doesn't know anything about functionalization.
  const Tensor& value() const { return value_; };
  // The concept of "level" is only ever important to functorch; it's exposed here
  // as more of a hook for functorch to use.
  int64_t level() const { return level_; };
  void set_level(int64_t level) { level_ = level; }

  // Sync's the underlying tensor with its alias, if it's out of date. This involves two steps:
  // 1) Apply any pending updates/mutations to the alias
  // 2) Replay the views (if any) to regenerate the current tensor off of the updated alias.
  void sync_();
  // Performs step (1) of the sync. This is its own public API because it's needed by view_inplace ops like transpose_.
  // See Note [Functionalization Pass - Inplace View Ops]
  void regenerate_from_base();
  // Performs step (2) of the sync. This is its own public API because it's needed by functorch.
  // functorch wants to make sure that all input tensors to a functionalized program have been properly synced
  // so it can properly propagate mutations to inputs.
  // It can't just call sync_(), because the FunctionalTensorWrapper will look like it has no aliases and sync_ will be a noop.
  // We use the reference count on storage_ to determine if the wrapper is aliased, and by the time functorch
  // is ready to propagate updates to inputs, any intermediate views of the input created by the program will have been deallocated.
  void apply_updates();
  // Takes the current state of value_ and snapshots it, sending it as a pending update to the alias.
  void commit_update();
  // When any tensor is mutated, the tensor increments its alias's "generation".
  // Separately, each tensor maintains its own "generation" counter, which is used to determine if it's up-to-date with its alias.
  // The act of syncing a tensor will set a tensor's generation equal to its alias's generation.
  bool is_up_to_date() const;
  // Every FunctionalTensorWrapper contains a vector<ViewMeta> objects describing the series of view ops that ran
  // to generate the current tensor from the base tensor.
  // This method is used by inplace-view ops like transpose_.
  // It appends a ViewMeta to the existing stack, and refreshes the tensor by replaying the views off of the alias.
  void mutate_view_meta(at::functionalization::ViewMeta meta);

  // The functionalization pass can be used to remove mutations.
  // It does so by replacing any mutation op with it's corresponding out-of-place op, followed by a call to replace_().
  // e.g:
  //
  // a.add_(1)
  //
  // will turn into:
  //
  // tmp = a.add(1)
  // a.replace_(tmp)
  //
  // replace_() swaps out the wrapped tensor, value_, with tmp.
  void replace_(const Tensor& other);

  ~FunctionalTensorWrapper() override = default;

 private:
  const char* tensorimpl_type_name() const override;
  void set_constructor_metadata();
  functionalization::FunctionalStorageImpl* functional_storage_impl() const;

  // Note that value is not taken by reference: internally, the wrapper will change the value tensor that it points to over time.
  Tensor value_;
  int64_t level_;

  size_t generation_ = 0;
  std::vector<at::functionalization::ViewMeta> view_metas_;
};

// Utility functions for the functionalization pass.

namespace functionalization {
namespace impl {

TORCH_API inline FunctionalTensorWrapper* unsafeGetFunctionalWrapper(const Tensor& tensor) {
  auto functional_impl = static_cast<FunctionalTensorWrapper*>(tensor.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  return functional_impl;
}

TORCH_API inline bool isFunctionalTensor(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Functionalize);
}

TORCH_API Tensor to_functional_tensor(const Tensor& tensor);
TORCH_API std::vector<Tensor> to_functional_tensor(ITensorList t_list);

TORCH_API Tensor from_functional_tensor(const Tensor& tensor);
TORCH_API c10::optional<Tensor> from_functional_tensor(const c10::optional<Tensor>& t);
TORCH_API c10::List<c10::optional<Tensor>> from_functional_tensor(const c10::List<c10::optional<Tensor>>& t_list);
TORCH_API std::vector<Tensor> from_functional_tensor(ITensorList t_list);

TORCH_API void sync(const at::Tensor& t);
TORCH_API void sync(const c10::optional<Tensor>& t);
TORCH_API void sync(const c10::List<c10::optional<Tensor>> t_list);
TORCH_API void sync(ITensorList t_list);

Tensor create_functional_tensor_with_view_meta(const Tensor& view_to_wrap, const Tensor& base, functionalization::ViewMeta meta, int64_t out_idx = 0);
std::vector<Tensor> create_functional_tensor_with_view_meta(ITensorList view_to_wrap, const Tensor& base, functionalization::ViewMeta meta);

void mutate_view_meta(const Tensor& self, functionalization::ViewMeta meta);

void set_sizes_strides_offset(const Tensor& out, const Tensor& meta_out);
void set_sizes_strides_offset(const std::vector<Tensor>& outs, const std::vector<Tensor>& meta_outs);

} // namespace impl
} // namespace functionalization
} // namespace at
