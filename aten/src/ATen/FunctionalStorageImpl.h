#pragma once

#include <ATen/Tensor.h>

#include <utility>

namespace at::functionalization {

// See Note [Functionalization Pass In Core]

enum class InverseReturnMode {
  /// Specifies that functional inverses should always return a view.
  AlwaysView,
  /// Specifies that functional inverses should always return a non-view / copy.
  NeverView,
  /// Specifies that functional inverses should return a view unless a (copying)
  /// scatter
  /// inverse exists, in which case that will be used instead.
  /// This avoids as_strided() calls that can be difficult for subclasses to
  /// handle.
  ViewOrScatterInverse,
};

#define FUNCTIONALIZATION_VIEWMETA_NAME(TYPE) \
  static const char* name() {                 \
    return #TYPE;                             \
  }

#define FUNCTIONALIZATION_VIEWMETA_SERIALIZABLE_TUPLE(...) \
  using SerializableTuple = std::tuple<__VA_ARGS__>;

// ViewMeta is a class used by the functionalization pass to navigate between
// a base tensor and a view tensor.
// For example, if I call `b = a.view1(...)`
// the functionalization pass will generate and store a ViewMeta specialization
// for `view1` operation on b that looks like:
//
// struct TORCH_API view1_ViewMeta : public ViewMeta {
//   FUNCTIONALIZATION_VIEWMETA_NAME(view1_ViewMeta);
//   FUNCTIONALIZATION_VIEWMETA_SERIALIZABLE_TUPLE(
//       bool /* reapply_views */,
//       const std::vector<int64_t>&);
//
//   view1_ViewMeta(const SerializableTuple& tpl)
//       : view1_ViewMeta(std::get<0>(tpl), std::get<1>(tpl)) {}
//
//   view1_ViewMeta(bool reapply_views, const std::vector<int64_t>& size)
//       : ViewMeta(/*has_symbolic_inputs=*/false),
//         reapply_views(reapply_views),
//         size(size) {}
//
//   Tensor forward(const Tensor& base) override {
//       return base.view1(...);
//   }
//
//   Tensor reverse(const Tensor& base, const Tensor& mutated_view) override {
//       return at::functionalization::impl::view1_inverse(base, mutated_view,
//       ...);
//   }
//
//   SerializableTuple to_serializable_tuple() {
//     return std::make_tuple(reapply_views, size);
//   }
//
//   bool reapply_views;
//   std::vector<int64_t> size;
// };
//
// The forward function describes how to replay view1 on a tensor.
//
// The reverse function describes how, given a tensor that is already a view,
// how to get the corresponding base tensor. See Note [Functionalization Pass:
// View Inverses] for details.
//
// `SerializedTuple` is a typedef that defines an `std::tuple<...>` type
// representing the `ViewMeta` instance state. Methods that take in/return such
// a type are used for supporting pickle serialization.
struct ViewMeta {
  ViewMeta(
      bool has_symbolic_inputs,
      bool is_multi_output = false,
      bool is_as_strided = false,
      int64_t out_idx = 0)
      : out_index(out_idx),
        is_multi_output(is_multi_output),
        is_as_strided(is_as_strided),
        has_symbolic_inputs(has_symbolic_inputs) {}

  virtual ~ViewMeta() {}

  virtual Tensor forward(const Tensor& base) = 0;
  virtual Tensor reverse(const Tensor& base, const Tensor& mutated_view) = 0;

  // See Note [out_idx in ViewMeta]
  int64_t out_index;

  // Tells us if this is a multi-output view
  bool is_multi_output;

  bool is_as_strided;

  // Tells us if this view operation has any symbolic inputs
  bool has_symbolic_inputs;

  // Returns a new ViewMeta with the same forward/reverse
  // functions, but a new out index.
  //
  // This method should be implemented by those `ViewMeta` that have more than
  // one output.
  virtual std::shared_ptr<ViewMeta> to_out_index(int64_t out_index) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "ViewMeta::to_out_index not implemented. ",
        "Likely because there's only one output.");
  }
};

// FunctionalStorageImpl is a subclass of StorageImpl used by the
// functionalization pass. It has no underlying data (similar to meta storage).
// It also knows how to reflect mutations to tensors in the absence of a valid
// data pointer.
//
// A storage represents the state shared by (potentially multiple) views of the
// same tensor. For example, in the following code:
//
// b = a.view1(...)
// c = b.view2(...)
// b.add_(1)
// --> storage.add_update(b, {view1_meta})
//
// The call to add_(1) will result in a call to alias.add_update(b,
// {view1_meta}), queueing up the mutation from b onto the alias. Later, suppose
// c is used in an expression (e.g. you try to print c, or pass it to an
// operator). Doing so will involve "syncing" c. First we apply any pending
// updates to the alias, and then we regenerate c by replaying its views off of
// the updated alias. E.g:
//
// print(str(c))
// --> c.sync_()
//     --> alias.apply_updates() // after this, the alias will be updated to
//     reflect the mutation to b
struct TORCH_API FunctionalStorageImpl : public c10::StorageImpl {
 public:
  struct Update {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const at::Tensor new_val;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::shared_ptr<ViewMeta>> view_metas;
  };

  explicit FunctionalStorageImpl(const Tensor& value);

  void add_update(
      const Tensor& updated_val,
      const std::vector<std::shared_ptr<ViewMeta>>& view_metas);
  bool apply_updates();
  const Tensor& base() {
    return base_;
  }
  size_t generation() const {
    return generation_;
  }
  void freeze() {
    frozen_ = true;
  }

  c10::SymInt get_storage_size(bool before) {
    if (before) {
      return original_storage_size_;
    } else {
      return curr_storage_size_;
    }
  }

  ~FunctionalStorageImpl() override = default;

  uint64_t mutation_counter() {
    return mutation_counter_;
  }
  void mark_mutation() {
    mutation_counter_++;
  }
  void mark_mutation_during_no_grad_or_inference_mode() {
    mutation_counter_during_no_grad_or_inference_mode_++;
  }
  void mark_mutation_hidden_from_autograd() {
    mutation_counter_hidden_from_autograd_++;
  }

  bool are_all_mutations_under_no_grad_or_inference_mode() const {
    auto non_autograd_mutations =
        mutation_counter_during_no_grad_or_inference_mode_ +
        mutation_counter_hidden_from_autograd_;
    // The <= is because both counters will technically be incremented, if we
    // perform e.g. a triton kernel mutation under no_grad
    return mutation_counter_ <= non_autograd_mutations;
  }

  bool are_all_mutations_hidden_from_autograd() const {
    // mutations under no_grad / inference_mode are technically not hidden from
    // autograd - they change the version counter
    return mutation_counter_ <= mutation_counter_hidden_from_autograd_;
  }

  void mark_inductor_storage_resize(c10::SymInt new_size) {
    inductor_storage_resized_ = true;
    curr_storage_size_ = std::move(new_size);
    inductor_storage_resized_counter_++;
  }

  bool was_inductor_storage_resized() {
    return inductor_storage_resized_;
  }

  uint64_t inductor_storage_resized_counter() {
    return inductor_storage_resized_counter_;
  }

 private:
  // NB: base_ should always point to a tensor BELOW the current
  // functionalization layer. This is mainly to avoid reference cycles. e.g.
  // given `b = a.view(...)` Both a.storage_ and b.storage_ are a
  // FunctionStorageImpl containing an Walualias, with contains a Tensor
  // `base_`. In this case (where a and b are FunctionalTensorWrapper's), base_
  // should point not to a, but to a's unwrapped value, a.value_` See Note
  // [Functionalization: Walualias Removal] for a diagram that shows this
  // visually.
  at::Tensor base_;
  std::vector<Update> updates_;
  // generation_ gets incremented every time a mutation is queued onto the
  // alias. It is used to determine if a given tensor is "up to date", or if it
  // needs to be regenerated from the alias.
  size_t generation_ = 0;
  // If frozen, no more mutations are allowed on this storage.  Once frozen, a
  // storage cannot be unfrozen.
  bool frozen_ = false;

  // These mutation counters are bumped on the storage
  // whenever a FunctionalTensorWrapper experiences a mutation.
  // When the mutation is under no_grad, or comes from a triton kernel, we also
  // bump the corresponding during_no_grad or hidden_from_autograd counters. Why
  // do we need to detect these two situations separately from "normal" input
  // mutations? (1) "normal" input mutations can mutate autograd metadata like
  // .grad_fn,
  //     in which case they need to be replayed outside of the compiled graph
  // (2) "no_grad" input mutations are generally safe to keep in the graph (and
  // compile),
  //     but they bump the tensor's VC, so we need to mark_dirty() on the inputs
  //     in torch.compile
  // (3) mutations that are fully hidden from autograd (e.g. from a triton
  // kernel)
  //     do not mutate any autograd state, and be fully kept in the graph
  // When we detect that an input was mutated, we need to be able to tell if:
  // (1) all of the mutations were from triton kernels
  // (2) all of the mutations were under no_grad
  uint64_t mutation_counter_during_no_grad_or_inference_mode_ = 0;
  uint64_t mutation_counter_ = 0;
  uint64_t mutation_counter_hidden_from_autograd_ = 0;

  // Used to tell if:
  // (1) There were any storage resizes on a graph input
  // (2) The original/curr storage size tell us if these resizes result in a nop
  bool inductor_storage_resized_ = false;
  uint64_t inductor_storage_resized_counter_ = 0;
  c10::SymInt original_storage_size_;
  c10::SymInt curr_storage_size_;
};

} // namespace at::functionalization
