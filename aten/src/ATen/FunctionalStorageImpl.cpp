#include <ATen/FunctionalStorageImpl.h>

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>
#include <vector>

namespace at {
namespace functionalization {

ViewMeta ViewMeta::to_out_idx(int64_t out_idx) {
  if (out_idx == this->out_index) return *this;
  return ViewMeta(forward_fn, reverse_fn, out_idx);
}

Alias::Alias(const at::Tensor& base) {
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(base));
  base_ = base;
}

const at::Tensor& Alias::base() const {
  return base_;
}

void Alias::add_update(const at::Tensor& updated_val, const std::vector<ViewMeta>& metas) {
  updates_.push_back({updated_val, metas});
  generation_++;
}

// Note [Functionalization: Alias Removal Part 2]
// See Note [Functionalization: Alias Removal] for more details.
// This function applies a single update from one of the views to the Alias object.
// We start out with <original_base> and <mutated_view>, and our goal is to end up with <mutated_base>.
// Consider this program:
//
// base = ...
// a = base.view1()
// b = a.view2()
// c = b.view3()
// c.add_(3)
//
// Then the functionalization pass will queue an update as follows:
//
// update.new_val = c  # the updated value of c
// update.view_metas = [view1_meta, view2_meta, view3_meta]
//
// Syncing any of a, b or c will eventually call apply_update() on the alias, and the following will run:
//
// tmp_values = [base, a, b]  # NB: c is not necessary
// t = update.new_val
// t = view3_inverse(b, t, 0)  # 0 is output index, these are all single output views so it's 0
// t = view2_inverse(a, t, 0)
// t = view1_inverse(base, t, 0)  # t now represents the updated alias.
// alias.base_ = t
const Tensor apply_update(const Alias::Update& update, const Tensor& base) {
  at::Tensor t = update.new_val;
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  if (update.view_metas.size() == 0) return t;

  std::vector<at::Tensor> tmp_values({base});
  for (size_t i = 0; i < update.view_metas.size() - 1; ++i) {
    at::Tensor next_view = update.view_metas[i].forward_fn(tmp_values.back(), update.view_metas[i].out_index);
    // NB: We only actually need tmp_values for ops like select/slice/diagonal/squeeze/as_strided
    // All of these ops require additional information to recover the sizes of the original tensor.
    // If need to, we could probably apply this optimization and only bother computing tmp_values
    // for those necessary view ops.
    tmp_values.push_back(std::move(next_view));
  }
  for(int i = update.view_metas.size()-1; i >= 0; --i) {
    int64_t out_idx = update.view_metas[i].out_index;
    // Each view inverse is implemented in ViewInverses.cpp.
    t = update.view_metas[i].reverse_fn(tmp_values[i], t, out_idx);
  }
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  return t;
}

bool Alias::apply_updates() {
  // N.B:none of the tensors used in this function should be FunctionalTensorWrappers at this point.
  // The only reason we currently need the TLS exclude guard here is because of functorch's DynamicLayer stack.
  // It adds the Functionalize key into TLS before redispatching to the functionalization kernels,
  // which means that we need to explicitly exclude it here before doing any other work underneath the pass.
  at::AutoDispatchSkipFunctionalize guard;
  bool any_updates = updates_.size() > 0;
  for (auto& update_data: updates_) {
    base_ = apply_update(update_data, base_);
  }
  updates_.clear();
  return any_updates;
}

// Should we just add a sym_numel() so we don't need to do this?
c10::SymInt compute_nbytes(c10::SymIntArrayRef sym_sizes, caffe2::TypeMeta dtype) {
  SymInt size_bytes = dtype.itemsize();
  for (auto s : sym_sizes) {
    size_bytes = size_bytes * s;
  }
  return size_bytes;
}

FunctionalStorageImpl::FunctionalStorageImpl(const Tensor& value)
  : c10::StorageImpl(
      c10::StorageImpl::use_byte_size_t(),
      compute_nbytes(value.sym_sizes(), value.dtype()),
      DataPtr{nullptr, value.device()},
      // Using a null allocator, since FunctionalTensorImpl's aren't resizeable.
      nullptr,
      /*resizeable=*/false
    ),
    alias_(Alias(value))
  {}

void FunctionalStorageImpl::add_update(const Tensor& updated_val, const std::vector<ViewMeta>& view_metas) {
  alias_.add_update(updated_val, view_metas);
}

bool FunctionalStorageImpl::apply_updates() {
  return alias_.apply_updates();
}

const Tensor& FunctionalStorageImpl::base() {
  return alias_.base();
}

size_t FunctionalStorageImpl::generation() const {
  return alias_.generation();
}

} // namespace functionalization
} // namespace at
