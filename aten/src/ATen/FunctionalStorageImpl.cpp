#include <ATen/FunctionalStorageImpl.h>

#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>
#include <vector>

namespace at::functionalization {

ViewMeta ViewMeta::to_out_idx(int64_t out_idx) {
  if (out_idx == this->out_index) return *this;
  return ViewMeta(forward_fn, reverse_fn, is_multi_output, out_idx);
}

// Note [Functionalization: Alias Removal Part 2]
// See Note [Functionalization: Alias Removal] for more details.
// This function applies a single update from one of the views to the StorageImpl.
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
// Syncing any of a, b or c will eventually call apply_update() on the storage, and the following will run:
//
// tmp_values = [base, a, b]  # NB: c is not necessary
// t = update.new_val
// t = view3_inverse(b, t, 0)  # 0 is output index, these are all single output views so it's 0
// t = view2_inverse(a, t, 0)
// t = view1_inverse(base, t, 0)  # t now represents the updated storage.
// storage.base_ = t
static const Tensor apply_update(const FunctionalStorageImpl::Update& update, const Tensor& base) {
  at::Tensor t = update.new_val;
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  if (update.view_metas.empty()) return t;

  std::vector<at::Tensor> tmp_values({base});
  tmp_values.reserve(update.view_metas.size());
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


static c10::SymInt get_nbytes(const Tensor& value) {
  // The functionalization story when wrapping tensors that don't have storage
  // is a bit wonky, but fortunately for some models (e.g., dlrm) we never
  // actually perform mutations on these tensors, so you never really get
  // called out on it.  For now, functionalization still creates "storages"
  // for these tensors (which is wrong), but we don't give them any space.
  // A more proper fix would be to have a SparseFunctionalTensorWrapper that
  // models sparse correctly.
  if (value.is_sparse()) {
    return 0;
  }
  if (value.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    // Today, the two implementations of SymInt are in Python (proxy tensor),
    // and lazy tensor (LTC/XLA).
    // LTC hasn't implemented SymInt support yet though
    // Once it does, we should remove this check.
    if (value.key_set().has(c10::DispatchKey::Python)) {
      return value.storage().sym_nbytes();
    }
    return at::detail::computeStorageNbytes(value.sym_sizes(), value.sym_strides(), value.dtype().itemsize(), value.sym_storage_offset());
  }
  // XLA storage objects also do not properly track nbytes.
  return at::detail::computeStorageNbytes(value.sizes(), value.strides(), value.dtype().itemsize(), value.storage_offset());
}

FunctionalStorageImpl::FunctionalStorageImpl(const Tensor& base)
  : c10::StorageImpl(
      c10::StorageImpl::use_byte_size_t(),
      get_nbytes(base),
      DataPtr{nullptr, base.device()},
      GetAllocator(kMeta),
      /*resizable=*/true
    ),
    base_(base)
  {
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(base_));
}

void FunctionalStorageImpl::add_update(const Tensor& updated_val, const std::vector<ViewMeta>& metas) {
  TORCH_CHECK(!frozen_, "cannot mutate tensors with frozen storage");
  updates_.push_back({updated_val, metas});
  generation_++;
}

bool FunctionalStorageImpl::apply_updates() {
  // N.B:none of the tensors used in this function should be FunctionalTensorWrappers at this point.
  // The only reason we currently need the TLS exclude guard here is because of functorch's DynamicLayer stack.
  // It adds the Functionalize key into TLS before redispatching to the functionalization kernels,
  // which means that we need to explicitly exclude it here before doing any other work underneath the pass.
  at::AutoDispatchSkipFunctionalize guard;
  bool any_updates = !updates_.empty();
  for (auto& update_data: updates_) {
    base_ = apply_update(update_data, base_);
  }
  updates_.clear();
  return any_updates;
}

} // namespace at::functionalization
