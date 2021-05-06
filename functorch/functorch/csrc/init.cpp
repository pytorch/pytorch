#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>

#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/VmapTransforms.h>
#include <functorch/csrc/PythonKey.h>
#include <functorch/csrc/BatchedFallback.h>
#include <functorch/csrc/BatchRulesHelper.h>

namespace at {
namespace functorch {

static bool has_level(const Tensor& self, int64_t level) {
  const auto* batched = maybeGetBatchedImpl(self);
  if (!batched) {
    return false;
  }
  auto bdims = batched->bdims();
  return bdims.back().level() >= level;
}

Tensor _add_batch_dim(const Tensor& self, int64_t batch_dim, int64_t level) {
  return addBatchDim(self, level, batch_dim);
}

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

// Poor man's version of np.moveaxis. Moves the dimension at `dst` to `src`
// while preserving the order of other existing dimensions.
// We should probably add np.moveaxis (it is more general) to PyTorch. (#36048)
// When we do, replace the following with it.
static Tensor _movedim(const Tensor& self, int64_t src, int64_t dst) {
  auto logical_dim = self.dim();
  src = maybe_wrap_dim(src, logical_dim);
  dst = maybe_wrap_dim(dst, logical_dim);
  if (src == dst) {
    return self;
  }
  VmapDimVector permutation;
  permutation.reserve(logical_dim);
  for (int64_t dim = 0; dim < logical_dim; dim++) {
    if (dim == src) {
      continue;
    }
    permutation.push_back(dim);
  }
  permutation.insert(permutation.begin() + dst, src);
  return self.permute(permutation);
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
    auto result = self.expand(expanded_sizes);
    return result;
  }

  // Must be batched if has_level(self, /*any_level*/)
  const auto* batched = maybeGetBatchedImpl(self);
  TORCH_INTERNAL_ASSERT(batched != nullptr);

  Tensor self_without_bdim;
  int64_t newly_exposed_logical_dim;
  std::tie(self_without_bdim, newly_exposed_logical_dim) = remove_existing_batch_dim(batched, level);
  auto result = _movedim(self_without_bdim, newly_exposed_logical_dim, out_dim);
  return result;
}

Tensor _wrap_for_grad(const Tensor& self, int64_t level) {
  // NB: different behavior inside??
  // return self;
  // TORCH_INTERNAL_ASSERT(!maybeGetTensorWrapper(self));
  // TORCH_INTERNAL_ASSERT(self.has_storage());
  return makeTensorWrapper(self, level);
}

Tensor _unwrap_for_grad(const Tensor& self, int64_t level) {
  auto* result = maybeGetTensorWrapper(self);
  if (!result) {
    return self;
  }
  TORCH_INTERNAL_ASSERT(result->level().has_value());
  if (result->level() == level) {
    return result->value();
  }
  return self;
}

int64_t dlevel(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    return 0;
  }
  if (!wrapped->is_alive()) {
    return -1;
  }
  return wrapped->level().value();
}

bool dump_tensor(const Tensor& self) {
  dumpTensorCout(self);
  return true;
}

int64_t _grad_increment_nesting() {
  return initAndPushDynamicLayer(at::DispatchKey::Autograd);
}

int64_t _grad_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  TORCH_INTERNAL_ASSERT(layer.key() == DispatchKey::Autograd);
  return layer.layerId();
}

int64_t _vmap_increment_nesting() {
  return initAndPushDynamicLayer(kBatchedKey);
}

int64_t _vmap_decrement_nesting() {
  auto layer = popDynamicLayerAndDeleteMetadata();
  TORCH_INTERNAL_ASSERT(layer.key() == kBatchedKey);
  return layer.layerId();
}


} // namespace functorch
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_add_batch_dim", &at::functorch::_add_batch_dim, "add batch dim");
  m.def("_remove_batch_dim", &at::functorch::_remove_batch_dim, "remove batch dim");
  m.def("_vmap_increment_nesting", &at::functorch::_vmap_increment_nesting, "remove batch dim");
  m.def("_vmap_decrement_nesting", &at::functorch::_vmap_decrement_nesting, "remove batch dim");
  m.def("_grad_increment_nesting", &at::functorch::_grad_increment_nesting, "remove batch dim");
  m.def("_grad_decrement_nesting", &at::functorch::_grad_decrement_nesting, "remove batch dim");
  m.def("_wrap_for_grad", &at::functorch::_wrap_for_grad, "add batch dim");
  m.def("_unwrap_for_grad", &at::functorch::_unwrap_for_grad, "add batch dim");
  m.def("_set_vmap_fallback_warning_enabled", &at::functorch::setVmapFallbackWarningEnabled, "Set vmap fallback warnings");
  m.def("dlevel", &at::functorch::dlevel, "add batch dim");
  m.def("dump_tensor", &at::functorch::dump_tensor, "add batch dim");
  m.def("reshape_dim_into", &at::functorch::reshape_dim_into);
  m.def("reshape_dim_outof", &at::functorch::reshape_dim_outof);

  m.def(
      "addPythonKey",
      &at::functorch::addPythonKey,
      py::return_value_policy::copy); // not sure if needed - cargo cult
  m.def("removePythonKey", &at::functorch::removePythonKey);
  m.def("hasPythonKey", &at::functorch::hasPythonKey);
}
