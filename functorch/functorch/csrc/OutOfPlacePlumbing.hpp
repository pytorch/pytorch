#include <functorch/csrc/OutOfPlacePlumbing.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/BatchedTensorImpl.h>

#include <ATen/Tensor.h>

namespace at { namespace functorch {

static Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level) {
  if (bdim.has_value()) {
    return makeBatched(tensor, {{level, bdim.value()}});
  }
  return tensor;
}

static std::tuple<Tensor, optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    return {tensor, nullopt};
  }
  TORCH_INTERNAL_ASSERT(batched->bdims().size() == 1);
  auto batched_level = batched->bdims().back().level();
  if (batched_level == level) {
    auto bdim = batched->bdims().back().dim();
    return {batched->value(), bdim};
  }
  return {tensor, nullopt};
}

}}
