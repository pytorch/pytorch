#include <ATen/BatchedTensorImpl.h>
#include <ATen/VmapTransforms.h>

namespace at { namespace native {

// Adds a batch dimension to the tensor `self` out-of-place
Tensor _add_batch_dim(const Tensor& self, int64_t batch_dim, int64_t level) {
  return addBatchDim(self, level, batch_dim);
}

static bool has_level(const Tensor& self, int64_t level) {
  const auto* batched = maybeGetBatched(self);
  if (!batched) {
    return false;
  }
  auto bdims = batched->bdims();
  auto* it = std::find_if(bdims.begin(), bdims.end(), [&](const BatchDim& bdim) {
    return bdim.level() == level;
  });
  return it != bdims.end();
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
  TORCH_INTERNAL_ASSERT(out_dim == 0);
  if (!has_level(self, level)) {
    auto self_sizes = self.sizes();
    VmapDimVector expanded_sizes(self_sizes.begin(), self_sizes.end());
    expanded_sizes.insert(expanded_sizes.begin() + out_dim, batch_size);
    return self.expand(expanded_sizes);
  }

  const auto* batched = maybeGetBatched(self);
  TORCH_INTERNAL_ASSERT(batched != nullptr);
  auto bdims = batched->bdims();
  if (bdims.size() == 1) {
    return batched->value();
  }
  BatchDims new_bdims;
  new_bdims.reserve(bdims.size() - 1);
  std::copy_if(bdims.begin(), bdims.end(), std::back_inserter(new_bdims),
               [&](const BatchDim& bdim) { return bdim.level() != level; });
  return makeBatched(batched->value(), std::move(new_bdims));
}

} // namespace native
} // namespace at
