
#pragma once

#include <ATen/ArrayRef.h>
#include <ATen/core/List.h>
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
  explicit FunctionalTensorWrapper(Tensor value);

  const Tensor& value() const { return value_; };
  int64_t level() const { return level_; };
  void set_level(int64_t level) { level_ = level; }

  void sync_(bool force_sync = false);
  void maybe_add_update();
  bool is_aliased() const;
  bool is_up_to_date() const;
  void set_view_meta(const Tensor& other, at::functionalization::ViewMeta meta);
  void mutate_view_meta(at::functionalization::ViewMeta meta);

  // Describe how to re-use a tensor in the functionalization pass.
  void replace_(const Tensor& other);

 private:
  const char* tensorimpl_type_name() const override;

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

TORCH_API Tensor wrapFunctionalTensor(const Tensor& tensor);
TORCH_API TensorList wrapFunctionalTensor(const c10::List<Tensor>& t_list);
TORCH_API std::vector<Tensor> wrapFunctionalTensor(const std::vector<Tensor>& t_list);
TORCH_API TensorList wrapFunctionalTensor(const TensorList& t_list);

TORCH_API Tensor unwrapFunctionalTensor(const Tensor& tensor);
TORCH_API c10::optional<Tensor> unwrapFunctionalTensor(const c10::optional<Tensor>& t);
TORCH_API c10::List<Tensor> unwrapFunctionalTensor(const c10::List<Tensor> t_list);
TORCH_API c10::List<c10::optional<Tensor>> unwrapFunctionalTensor(const c10::List<c10::optional<Tensor>> t_list);
TORCH_API TensorList unwrapFunctionalTensor(const TensorList& tensors);

TORCH_API void sync(const at::Tensor& t);
TORCH_API void sync(const c10::optional<Tensor>& t);
TORCH_API void sync(const c10::List<Tensor> t_list);
TORCH_API void sync(const at::TensorList t_list);
TORCH_API void sync(const c10::List<c10::optional<Tensor>> t_list);

void maybe_add_update(Tensor& self);

void set_view_meta(const Tensor& out, const Tensor& t, functionalization::ViewMeta meta, int64_t out_idx = 0);
void set_view_meta(const c10::List<Tensor> outs, const Tensor& t, functionalization::ViewMeta meta);
void set_view_meta(const std::vector<Tensor> outs, const Tensor& t, functionalization::ViewMeta meta);

void mutate_view_meta(const Tensor& self, functionalization::ViewMeta meta);

void set_strides(const Tensor& out, const Tensor& meta_out);
void set_strides(const std::vector<Tensor>& outs, const std::vector<Tensor>& meta_outs);

} // namespace impl
} // namespace functionalization
} // namespace at
