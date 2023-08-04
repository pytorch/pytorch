
#pragma once

#include <ATen/ArrayRef.h>
#include <ATen/FunctionalStorageImpl.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/List.h>
#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <c10/core/DispatchKey.h>

namespace at {

// Note [Functionalization Pass In Core]
// The Functionalization pass is used to remove aliasing from a pytorch program.
//
// This is useful for backends that don't support aliasing, like XLA and Vulkan.
// It's also necessary in order to remove mutation from a program, which is
// needed in Functorch.
//
// Consider this program:
// a = torch.ones(...)
// b = a.view(...)
// b.add_(1)
//
// In this program, b is meant to alias with a due to the use of view(). At the
// end of the program, both a and b are full of 2's. However, backends that
// don't support aliasing aren't able to correctly implement the view()
// operator. Instead, they can opt into the Functionalization pass, which will
// sit between the user and the backend, and provide the necessary aliasing
// logic.
//
// The functionalization pass will turn the above program into a slightly
// different program that has the same semantics, transparently to the user,
// that backends like XLA/Vulkan are able to implement a = torch.ones(...) b =
// a.view_copy(...)  # view() replaced with view_copy(). Backends like
// XLA/Vulkan can implement this! b.add_(1) a.add_(1)  # Our functionalization
// pass machinery knows that a and b are aliased - it applies b's mutation to a
// too.
//
// So, how does the functionalization pass keep track of which tensors are
// aliased? The pass works by wrapping EVERY tensor in the program inside of a
// FunctionalTensorWrapper, which knows about its alias'd tensors.
//
// See Note [Functionalization: Alias Removal] for details on the aliasing
// machinery. See Note [Functionalization: Mutation Removal] for details on
// mutation removal.
struct TORCH_API FunctionalTensorWrapper : public c10::TensorImpl {
  explicit FunctionalTensorWrapper(const Tensor& value);
  // Additional constructor to create a FunctionalTensorWrapper directly from an
  // underlying tensor that was created from a view. For example, the code b =
  // a.view1() will generate a constructor call to FunctionalTensorWrapper(b, a,
  // view1_meta)
  explicit FunctionalTensorWrapper(
      const Tensor& view_value,
      const FunctionalTensorWrapper* base,
      functionalization::ViewMeta meta);

  // Get the underlying, actual tensor, that doesn't know anything about
  // functionalization.
  const Tensor& value() const {
    return value_;
  };
  // The concept of "level" is only ever important to functorch; it's exposed
  // here as more of a hook for functorch to use.
  int64_t level() const {
    return level_;
  };
  void set_level(int64_t level) {
    level_ = level;
  }
  bool has_metadata_mutation() const {
    return has_metadata_mutation_;
  };

  // Sync's the underlying tensor with its alias, if it's out of date. This
  // involves two steps: 1) Apply any pending updates/mutations to the alias 2)
  // Replay the views (if any) to regenerate the current tensor off of the
  // updated alias.
  void sync_();
  // Performs step (1) of the sync. This is its own public API because it's
  // needed by view_inplace ops like transpose_. See Note [Functionalization
  // Pass - Inplace View Ops]
  void regenerate_from_base();
  // Performs step (2) of the sync. This is its own public API because it's
  // needed by functorch. functorch wants to make sure that all input tensors to
  // a functionalized program have been properly synced so it can properly
  // propagate mutations to inputs. It can't just call sync_(), because the
  // FunctionalTensorWrapper will look like it has no aliases and sync_ will be
  // a noop. We use the reference count on storage_ to determine if the wrapper
  // is aliased, and by the time functorch is ready to propagate updates to
  // inputs, any intermediate views of the input created by the program will
  // have been deallocated. This function also returns whether or not the base
  // actually had any updates to apply.
  bool apply_updates();
  // Takes the current state of value_ and snapshots it, sending it as a pending
  // update to the alias.
  void commit_update();
  // When any tensor is mutated, the tensor increments its alias's "generation".
  // Separately, each tensor maintains its own "generation" counter, which is
  // used to determine if it's up-to-date with its alias. The act of syncing a
  // tensor will set a tensor's generation equal to its alias's generation.
  bool is_up_to_date() const;
  // Freezes the storage of this tensor, preventing subsequent mutations
  void freeze_storage() const;
  // Every FunctionalTensorWrapper contains a vector<ViewMeta> objects
  // describing the series of view ops that ran to generate the current tensor
  // from the base tensor. This method is used by inplace-view ops like
  // transpose_. It appends a ViewMeta to the existing stack, and refreshes the
  // tensor by replaying the views off of the alias.
  void mutate_view_meta(at::functionalization::ViewMeta meta);

  // The functionalization pass can be used to remove mutations.
  // It does so by replacing any mutation op with it's corresponding
  // out-of-place op, followed by a call to replace_(). e.g:
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

  // See Note[resize_() in functionalization pass]
  void maybe_replace_storage(const Tensor& other);

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  ~FunctionalTensorWrapper() override = default;

  // FunctionalTensorWrapper overrides all custom size/stride function,
  // so that if the inner tensor has a custom implementation
  // we make sure to call that implementation.
  at::IntArrayRef sizes_custom() const override;
  at::IntArrayRef strides_custom() const override;
  int64_t dim_custom() const override;
  int64_t numel_custom() const override;
  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;
  c10::SymIntArrayRef sym_sizes_custom() const override;
  c10::SymInt sym_size_custom(int64_t d) const override;
  c10::SymIntArrayRef sym_strides_custom() const override;
  c10::SymInt sym_storage_offset_custom() const override;
  c10::Device device_custom() const override;

 private:
  const char* tensorimpl_type_name() const override;
  void set_constructor_metadata();
  functionalization::FunctionalStorageImpl* functional_storage_impl() const;

  // This is used to re-implement shallow_copy_and_detach for
  // FunctionalTensorWrapper. The implementation is identical, but we just need
  // to return a subclass instead of a plain TensorImpl.
  // TODO: maybe it's possible to arrange for that to happen automatically
  // without an override here?
  template <typename VariableVersion>
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach_core(
      VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;

  // Note that value is not taken by reference: internally, the wrapper will
  // change the value tensor that it points to over time.
  Tensor value_;
  int64_t level_;
  bool has_metadata_mutation_ = false;

  size_t generation_ = 0;
  std::vector<at::functionalization::ViewMeta> view_metas_;
};

// Utility functions for the functionalization pass.

namespace functionalization {
namespace impl {

TORCH_API inline FunctionalTensorWrapper* unsafeGetFunctionalWrapper(
    const Tensor& tensor) {
  auto functional_impl =
      static_cast<FunctionalTensorWrapper*>(tensor.unsafeGetTensorImpl());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(functional_impl != nullptr);
  return functional_impl;
}

TORCH_API bool isFunctionalTensor(const at::Tensor& tensor);
TORCH_API bool isFunctionalTensor(const c10::optional<Tensor>& t);
TORCH_API bool isFunctionalTensor(
    const c10::List<c10::optional<Tensor>>& t_list);
TORCH_API bool isFunctionalTensor(ITensorListRef list);

TORCH_API Tensor to_functional_tensor(const Tensor& tensor);
TORCH_API c10::optional<Tensor> to_functional_tensor(
    const c10::optional<Tensor>& tensor);
TORCH_API c10::List<c10::optional<Tensor>> to_functional_tensor(
    const c10::List<c10::optional<Tensor>>& t_list);
TORCH_API std::vector<Tensor> to_functional_tensor(ITensorListRef t_list);

TORCH_API void freeze_functional_tensor(const Tensor& tensor);

TORCH_API Tensor
from_functional_tensor(const Tensor& tensor, bool assert_functional = true);
TORCH_API c10::optional<Tensor> from_functional_tensor(
    const c10::optional<Tensor>& t,
    bool assert_functional = true);
TORCH_API c10::List<c10::optional<Tensor>> from_functional_tensor(
    const c10::List<c10::optional<Tensor>>& t_list);
TORCH_API std::vector<Tensor> from_functional_tensor(ITensorListRef t_list);

TORCH_API void sync(const at::Tensor& t);
TORCH_API void sync(const c10::optional<Tensor>& t);
TORCH_API void sync(const c10::List<c10::optional<Tensor>>& t_list);
TORCH_API void sync(ITensorListRef t_list);

TORCH_API void replace_(const Tensor& functional_tensor, const Tensor& other);
TORCH_API void replace_(
    const ITensorListRef functional_tensor,
    ITensorListRef other);

TORCH_API void commit_update(const Tensor& functional_tensor);
TORCH_API void commit_update(ITensorListRef functional_tensor);

// These two methods are XLA-specific logic and are no-ops
// for the normal functionalization flow.
TORCH_API void propagate_xla_data(
    const Tensor& functional_tensor,
    const Tensor& other);
TORCH_API void propagate_xla_data(
    const ITensorListRef functional_tensor,
    ITensorListRef other);

Tensor create_functional_tensor_with_view_meta(
    const Tensor& view_to_wrap,
    const Tensor& base,
    functionalization::ViewMeta meta,
    int64_t out_idx = 0);
std::vector<Tensor> create_functional_tensor_with_view_meta(
    ITensorListRef view_to_wrap,
    const Tensor& base,
    functionalization::ViewMeta meta);

void mutate_view_meta(const Tensor& self, functionalization::ViewMeta meta);

void set_sizes_strides_offset(const Tensor& out, const Tensor& meta_out);
void set_sizes_strides_offset(
    const std::vector<Tensor>& outs,
    const std::vector<Tensor>& meta_outs);

//  ~~~~~ TLS used in functionalization ~~~~~

TORCH_API bool getFunctionalizationReapplyViewsTLS();
TORCH_API void setFunctionalizationReapplyViewsTLS(bool reapply_views);

class TORCH_API FunctionalizationReapplyViewsGuard {
 public:
  FunctionalizationReapplyViewsGuard(bool reapply_views)
      : prev_(getFunctionalizationReapplyViewsTLS()) {
    setFunctionalizationReapplyViewsTLS(reapply_views);
  }

  ~FunctionalizationReapplyViewsGuard() {
    setFunctionalizationReapplyViewsTLS(prev_);
  }

  FunctionalizationReapplyViewsGuard(
      const FunctionalizationReapplyViewsGuard&) = delete;
  FunctionalizationReapplyViewsGuard operator=(
      const FunctionalizationReapplyViewsGuard&) = delete;
  FunctionalizationReapplyViewsGuard(FunctionalizationReapplyViewsGuard&&) =
      delete;
  FunctionalizationReapplyViewsGuard operator=(
      FunctionalizationReapplyViewsGuard&&) = delete;

 private:
  bool prev_;
};

} // namespace impl

// Helper function to call an out-of-place composite aten kernel that may use
// mutations / views internally, and functionalize them.
TORCH_API void functionalize_op_helper(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

template <class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _functionalize_aten_op final {};

template <class Op, bool symint, class ReturnType, class... ParameterTypes>
struct _functionalize_aten_op<Op, symint, ReturnType(ParameterTypes...)> final {
  static ReturnType call(
      typename c10::maybe_keep_symint<symint, ParameterTypes>::type... args) {
    using FuncType = ReturnType(
        typename c10::maybe_keep_symint<symint, ParameterTypes>::type...);
    auto op = c10::Dispatcher::singleton()
                  .findSchemaOrThrow(
                      (const char*)Op::name, (const char*)Op::overload_name)
                  .typed<FuncType>();

    return c10::impl::BoxedKernelWrapper<FuncType>::call(
        c10::BoxedKernel::makeFromFunction<functionalize_op_helper>(),
        op,
        // BoxedKernelWrapper knows to ignore this keyset argument,
        // because functionalize_op_helper doesn't take in a DispatchKeySet
        c10::DispatchKeySet(),
        args...);
  }
};

template <class Op>
using functionalize_aten_op =
    _functionalize_aten_op<Op, false, typename Op::schema>;

template <class Op>
using functionalize_aten_op_symint =
    _functionalize_aten_op<Op, true, typename Op::schema>;

} // namespace functionalization
} // namespace at
