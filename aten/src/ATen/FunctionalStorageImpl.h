#pragma once

#include <ATen/Tensor.h>

namespace at::functionalization {

// See Note [Functionalization Pass In Core]

// ViewMeta is a class used by the functionalization pass to navigate between
// a base tensor and a view tensor.
// For example, if I call `b = a.view1(...)`
// the functionalization pass will generate and store a ViewMeta on b that looks
// like:
//
// ViewMeta(
//   [<captures>](const Tensor& base, int64_t mutated_view_idx) {
//     return base.view1(...);
//   },
//   [<captures>](const at::Tensor& base, const at::Tensor& mutated_view,
//   int64_t mutated_view_idx) -> at::Tensor {
//     return at::functionalization::impl::view1_inverse(base, mutated_view,
//     ...);
//   }
//
// The forward_fn lambda describes how to replay view1 on a tensor.
//
// The reverse_fn lambda describes how, given a tensor that is already a view,
// how to get the corresponding base tensor. See Note [Functionalization Pass:
// View Inverses] for details.
struct ViewMeta {
  ViewMeta(
      std::function<Tensor(const Tensor&, int64_t)> forward,
      std::function<Tensor(const Tensor&, const Tensor&, int64_t)> reverse,
      int64_t out_idx = 0)
      : forward_fn(std::move(forward)),
        reverse_fn(std::move(reverse)),
        out_index(out_idx) {}

  std::function<Tensor(const Tensor&, int64_t)> forward_fn;
  std::function<Tensor(const Tensor&, const Tensor&, int64_t)> reverse_fn;
  // See Note [out_idx in ViewMeta]
  int64_t out_index;

  // Returns a copy of the current ViewMeta, if out_idx matches the current
  // out_index. Otherwise, returns a new ViewMeta with the same forward/reverse
  // functions, but a new out index.
  ViewMeta to_out_idx(int64_t out_idx);
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
    const at::Tensor new_val;
    const std::vector<ViewMeta> view_metas;
  };

  explicit FunctionalStorageImpl(const Tensor& value);

  void add_update(
      const Tensor& updated_val,
      const std::vector<ViewMeta>& view_metas);
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

  ~FunctionalStorageImpl() override = default;

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
};

} // namespace at::functionalization
