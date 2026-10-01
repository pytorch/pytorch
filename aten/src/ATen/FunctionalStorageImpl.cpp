#include <ATen/FunctionalStorageImpl.h>

#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/util/Exception.h>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/copy.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/view_copy.h>
#endif

namespace at::functionalization {

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
//
// Note [Functionalization: mutations through expand()]
// The view inverses assume every element of a view is its own memory location. expand() breaks
// that: several elements of the expanded view alias one element of its input, and expand_inverse
// can't tell which of the aliases were written. Instead, for the views from the first expand() on,
// we replay them on a tensor holding each element's position in the expand's input, and scatter
// the mutated values straight back to those positions.
static Tensor apply_update_through_expand(
    const std::vector<std::shared_ptr<ViewMeta>>& view_metas,
    size_t expand_idx,
    const std::vector<at::Tensor>& tmp_values,
    const Tensor& new_val) {
  const auto& input = tmp_values[expand_idx];
  bool same_dtype = new_val.scalar_type() == input.scalar_type();
  for (size_t i = expand_idx + 1; i < tmp_values.size(); ++i) {
    same_dtype &= tmp_values[i].scalar_type() == input.scalar_type();
  }
  TORCH_CHECK(same_dtype, "Functionalization does not support a mutation through a dtype-changing view of an expand()");
  // We run below functionalization, and some backends (e.g. lazy) have no view ops.
  auto reshape = [](const Tensor& t, c10::SymIntArrayRef sizes) {
    return at::view_copy_symint(t.contiguous(), sizes);
  };
  auto long_options = input.options().dtype(at::kLong);
  auto positions = reshape(at::arange(c10::Scalar(input.sym_numel()), long_options), input.sym_sizes());
  // Lazy tensors are always contiguous, and the lazy backend can't build strided tensors.
  bool contiguous = input.is_contiguous_or_false();
  if (!contiguous) {
    // Give positions the same strides as input so that view() replays exactly as it did on input.
    positions =
        at::copy(at::empty_strided_symint(input.sym_sizes(), input.sym_strides(), long_options), positions);
  }
  for (size_t i = expand_idx; i < view_metas.size(); ++i) {
    positions = view_metas[i]->forward(positions);
  }
  c10::SymInt numel = positions.sym_numel();
  c10::List<std::optional<at::Tensor>> indices;
  indices.push_back(reshape(positions, numel));
  auto flat = at::index_put(reshape(input, input.sym_numel()), indices, reshape(new_val, numel));
  auto result = reshape(flat, input.sym_sizes());
  // Preserve input's strides: views replayed from the updated base must stay valid.
  return contiguous ? result : at::copy(input, result);
}

static const Tensor apply_update(const FunctionalStorageImpl::Update& update, const Tensor& base) {
  at::Tensor t = update.new_val;
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(t));
  if (update.view_metas.empty()) { return t; }

  std::vector<at::Tensor> tmp_values({base});
  tmp_values.reserve(update.view_metas.size());
  for (size_t i = 0; i < update.view_metas.size() - 1; ++i) {
    at::Tensor next_view = update.view_metas[i]->forward(tmp_values.back());
    // NB: We only actually need tmp_values for ops like select/slice/diagonal/squeeze/as_strided
    // All of these ops require additional information to recover the sizes of the original tensor.
    // If need to, we could probably apply this optimization and only bother computing tmp_values
    // for those necessary view ops.
    tmp_values.push_back(std::move(next_view));
  }
  size_t num_inverted = update.view_metas.size();
  for (size_t i = 0; i < update.view_metas.size(); ++i) {
    if (update.view_metas[i]->is_expand) {
      t = apply_update_through_expand(update.view_metas, i, tmp_values, update.new_val);
      num_inverted = i;
      break;
    }
  }
  for(int64_t i = static_cast<int64_t>(num_inverted) - 1; i >= 0; --i) {
    // Each view inverse is implemented in ViewInverses.cpp.
    t = update.view_metas[i]->reverse(tmp_values[i], t);
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
  if (value.is_sparse() || at::sparse_csr::is_sparse_compressed(value)) {
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
    return at::detail::computeStorageNbytes(value.sym_sizes(), value.sym_strides(),static_cast<int64_t>(value.dtype().itemsize()), value.sym_storage_offset());
  }
  // XLA storage objects also do not properly track nbytes.
  return static_cast<int64_t>(at::detail::computeStorageNbytes(value.sizes(), value.strides(), value.dtype().itemsize(), value.storage_offset()));
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
  // SparseTensorImpl has no storage, so we cannot query its nbytes.
  // (original_storage_size is only used for storage resizing in fsdp anyway, which does not apply to sparse)
  // Same for XLA
  if (base.unsafeGetTensorImpl()->has_storage() && data_ptr().device().type() != c10::DeviceType::XLA) {
    original_storage_size_ = base.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl()->sym_nbytes();
  } else {
    original_storage_size_ = -1;
  }
  curr_storage_size_ = original_storage_size_;
  TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(base_));
}

void FunctionalStorageImpl::add_update(const Tensor& updated_val, const std::vector<std::shared_ptr<ViewMeta>>& metas) {
  TORCH_CHECK(!frozen_, "cannot mutate tensors with frozen storage");

  if (metas.size() > 1) {
    for (size_t i = 1; i < metas.size(); ++i) {
      // Skipping this check for XLA. Would be good to add it back, but it is failing XLA CI
      TORCH_CHECK(updated_val.device().type() == c10::DeviceType::XLA || !metas[i]->is_as_strided,
"During torch.compile, encountered a mutation on a view chain of length ", metas.size(), ", where view ", i,
" was an as_strided() call. as_strided() is non-compositional, and therefore is not possible to functionalize properly today,"
"so this behavior is banned in compile. As a workaround, you can either remove the mutation from the model code, or you "
"can insert a graph break right before the mutation with torch._dynamo.graph_break(). If you would like this behavior to "
"work properly, please comment on https://github.com/pytorch/pytorch/issues/104505.");
    }
  }
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
