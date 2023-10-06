#include <ATen/core/Tensor.h>
#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/LegacyVmapTransforms.h>

#ifdef AT_PER_OPERATOR_HEADERS
#include <ATen/ops/_add_batch_dim_native.h>
#include <ATen/ops/_remove_batch_dim_native.h>
#endif

namespace at { namespace native {

// Adds a batch dimension to the tensor `self` out-of-place
Tensor _add_batch_dim(const Tensor& self, int64_t batch_dim, int64_t level) {
  return addBatchDim(self, level, batch_dim);
}

static bool has_level(const Tensor& self, int64_t level) {
  const auto* batched = maybeGetBatchedImpl(self);
  if (!batched) {
    return false;
  }
  auto bdims = batched->bdims();
  auto* it = std::find_if(bdims.begin(), bdims.end(), [&](const BatchDim& bdim) {
    return bdim.level() == level;
  });
  return it != bdims.end();
}

// Returns a Tensor with batch dim with level `level` turned into a regular dimension,
// as well as a logical dim index of where said dimension is in the returned tensor.
// A call to this function is always followed by a call to `movedim`.
//
// Preconditions: A BatchDim with level `level` must exist inside `batched`.
//
// The reason why we want to return the index of where said dimension is in the returned
// tensor is because we want to keep track of which dimension used to be the batch
// dimension so that we can move it to the correct logical dimension specified by
// `out_dims` in vmap. For example, if we had
// >>> x = torch.randn(2, 3, 5)
// >>> vmap(lambda x: x, in_dims=0, out_dims=1)(x)
// then right when we are about to exit the vmap block, x is a BatchedTensor with a
// batch dimension at (physical) index 0. Note that the batch dimension doesn't
// always have to exist at (physical) index 0. When we undo the batch dimension,
// we want to move it to dimension 1 (as specified by out_dims). So we return the
// index at which the batch dim appears so that we can move it to the correct place.
// later down the line via a call to `movedim`.
static std::pair<Tensor,int64_t> remove_existing_batch_dim(
    const BatchedTensorImpl* batched, int64_t level) {
  auto bdims = batched->bdims();
  if (bdims.size() == 1) {
    TORCH_INTERNAL_ASSERT(bdims[0].level() == level);
    return std::make_pair(batched->value(), bdims[0].dim());
  }
  BatchDims new_bdims;
  int64_t newly_exposed_physical_dim = -1;
  new_bdims.reserve(bdims.size() - 1);
  for (const auto& bdim : bdims) {
    if (bdim.level() == level) {
      newly_exposed_physical_dim = bdim.dim();
    } else {
      new_bdims.push_back(bdim);
    }
  }
  // Because a BatchDim with level `level` must exist inside `batched,
  // we should have found a `newly_exposed_logical_dim`.
  TORCH_INTERNAL_ASSERT(newly_exposed_physical_dim != -1);
  int64_t num_batch_dims_before_newly_exposed_physical_dim = std::count_if(
      new_bdims.begin(), new_bdims.end(),
      [&](const BatchDim& bdim) {
        return bdim.dim() < newly_exposed_physical_dim;
      });
  int64_t newly_exposed_logical_dim =
      newly_exposed_physical_dim - num_batch_dims_before_newly_exposed_physical_dim;
  auto result_tensor = makeBatched(batched->value(), std::move(new_bdims));
  return std::make_pair(std::move(result_tensor), newly_exposed_logical_dim);
}

// at::movedim but may return the original tensor if dst is the same as src.
static Tensor maybe_movedim(const Tensor& self, int64_t src, int64_t dst) {
  auto logical_dim = self.dim();
  src = maybe_wrap_dim(src, logical_dim);
  dst = maybe_wrap_dim(dst, logical_dim);
  if (src == dst) {
    return self;
  }
  return self.movedim(src, dst);
}

// Removes the batch dim with level `level` from `self`. If this causes the
// last batch dim to be removed from a BatchedTensor, then this returns a
// regular Tensor.
//
// If the `level` of the batch dim to remove does not exist in `self`, then we
// add the batch dim in. This can happen if `self` didn't interact with a tensor
// inside the vmap level, for example,
//     self = torch.randn(3)
//     y = torch.randn(5)
//     out = vmap(lambda x: vmap(lambda y: x)(y))(self)
//     assert out.shape == (3, 5)
// Inside the inner vmap, `x` is a BatchedTensor with a single batch dimension
// corresponding to the *outer* vmap level and it doesn't have any dimensions that
// correspond to the inner vmap level so we need to create one for the user.
//
// `out_dim` controls where we should put the batch dimension in the output tensor.
Tensor _remove_batch_dim(const Tensor& self, int64_t level, int64_t batch_size, int64_t out_dim) {
  if (!has_level(self, level)) {
    auto self_sizes = self.sizes();
    VmapDimVector expanded_sizes(self_sizes.begin(), self_sizes.end());
    expanded_sizes.insert(expanded_sizes.begin() + out_dim, batch_size);
    return self.expand(expanded_sizes);
  }

  // Must be batched if has_level(self, /*any_level*/)
  const auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched != nullptr);

  Tensor self_without_bdim;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t newly_exposed_logical_dim;
  std::tie(self_without_bdim, newly_exposed_logical_dim) = remove_existing_batch_dim(batched, level);
  return maybe_movedim(self_without_bdim, newly_exposed_logical_dim, out_dim);
}

} // namespace native
} // namespace at
