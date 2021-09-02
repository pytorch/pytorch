#include <ATen/FunctionalStorageImpl.h>

#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>
#include <vector>

namespace at {
namespace functionalization {

Alias::Alias(const at::Tensor& base) {
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(base));
  base_ = base;
}

const at::Tensor& Alias::base() const {
  return base_;
}

// metas is taken by value on purpose - we want to copy the vector.
void Alias::add_update(const at::Tensor& updated_val, std::vector<ViewMeta> metas) {
  updates_.push_back({updated_val, metas});
  generation_++;
}

// Note [Functionalization: Alias Removal Part 2]
// See Note [Functionalization: Alias Removal] for more details.
// This function applies a single update from one of the views to the Alias object.
// We start out with <original_base> and <mutated_view>, and our goal is to end up with <mutated_base>.
void Alias::apply_update(const Update& update) {
  at::AutoDispatchBelowFunctionalize guard;
  at::Tensor t = update.new_val;
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  std::vector<at::Tensor> tmp_values({base_});
  // First, we replay each view on <original_base>.
  // We only actually need this for ops like select/slice/diagonal, which creates a view that's a subset of the original tensor.
  // e.g.:
  // a = torch.ones(...)
  // b = a[0]
  // c = a.permute(...)
  // d = c[0]
  // d.add_(1)
  // In order to get the mutated version of a, we need to know the intermediates b and c.
  for (size_t i = 0; i < update.view_metas.size() - 1; ++i) {
    at::Tensor next_view = update.view_metas[i].forward_fn(tmp_values.back(), update.view_metas[i].out_index);
    tmp_values.push_back(std::move(next_view));
  }

  // Next, starting with <mutated_view>, we apply the inverse of each view to it in reverse order,
  // Eventually ending up with <mutated_base>.
  for(int i = update.view_metas.size()-1; i >= 0; --i) {
    int64_t out_idx = update.view_metas[i].out_index;
    // Each view inverse is implemented in ViewInverses.cpp.
    t = update.view_metas[i].reverse_fn(tmp_values[i], t, out_idx);
  }
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  base_ = t;
}

Tensor Alias::sync_update_operations() {
  for (auto& update_data: updates_) {
    apply_update(update_data);
  }
  updates_.clear();
  return base_;
}

FunctionalStorageImpl::FunctionalStorageImpl(c10::Device device, int64_t numel, caffe2::TypeMeta dtype)
  : c10::StorageImpl(
      c10::StorageImpl::use_byte_size_t(),
      numel * dtype.itemsize(),
      DataPtr{nullptr, device},
      // Using a null allocator, since FunctionalTensorImpl's aren't resizeable.
      nullptr,
      /*resizeable=*/false
    )
  {}

bool FunctionalStorageImpl::is_aliased() const {
    return (bool) alias_;
}

bool FunctionalStorageImpl::maybe_add_update(const Tensor& updated_val, std::vector<ViewMeta>& view_metas) {
  // If the mutated tensor doesn't have an alias, we don't need to do anything
  if (is_aliased()) {
    alias_->add_update(updated_val, view_metas);
    return true;
  }
  return false;
}

Tensor FunctionalStorageImpl::sync_update_operations() {
  TORCH_INTERNAL_ASSERT(is_aliased());
  return alias_->sync_update_operations();
}

void FunctionalStorageImpl::set_alias(const Tensor& alias) {
  TORCH_INTERNAL_ASSERT(!is_aliased());
  alias_ = std::make_unique<at::functionalization::Alias>(alias);
}

size_t FunctionalStorageImpl::generation() const {
  if (is_aliased()) {
    return alias_->generation();
  }
  return 0;
}

} // namespace functionalization
} // namespace at
