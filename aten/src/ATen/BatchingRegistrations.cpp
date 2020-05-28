#include <ATen/BatchedTensorImpl.h>
#include <torch/library.h>
#include <ATen/BatchingUtils.h>
#include <ATen/ATen.h>

namespace at {

// Note: [How to write batching rules]
//
// Whenever you do a vmap, on a vmap'ed tensor, the dimension that is being vmap'ed
// gets recorded as a "batch dimension". Tensors that are not being vmap'ed over
// behave as if they had a broadcasting batch dimension added to them.
//
// A batching rule function implements the logic of how to call an operator on
// some inputs that have one or more batch dimensions.
//
// ==========================================
// When and why should I add a batching rule?
// ==========================================
// When you are adding a new operator, you'll need to add a batching rule so
// that vmap can work efficiently with said operator. If you do not, we'll attempt
// to generate a slow fallback for the batching rule (TODO). In the future,
// if the operator is composite (with respect to autograd), then we'll offer a
// mechanism to register a fallthrough where we just run the composite op directly
// on the BatchedTensor (TODO).
//
// ====================================
// How to implement a new batching rule
// ====================================
// The signature of a batching rule should look like exactly like the C++ signature
// of its operator.
// `
// Most batching rules will look like the following:
// 1. Move all batch dims to the front of the input tensors
// 2. If there are multiple tensors, align their batch dimensions by level
// 3. Once the tensors are aligned, figure out how to call the at:: op that
//    corresponds to the operator with additional batch dims
// The first two steps can be handled via a call to `materializeBatchDimsAtFront`.
// (TODO: there's a multi-tensor variant).
//
// There are a couple of cases for that last step:
// 1. For operators like `add`, that support multiple broadcasting dimensions,
//    we can call the op directly.
// 2. For operators like `conv2d`, that only take in a single batch dim, we can
//    flatten the batch dims into a single dimension, call conv2d, and then
//    unflatten the batch dims of the output. Note that this only works for
//    the `input` tensor to conv2d; if `weights` is batched, then go to case 3.
// 3. For operators where the above two approaches don't work, we need to
//    register and write a new custom operator that handles the batching.
//    If this is too much work, call the slow fallback in the batching rule.
//
// ============================
// Future API changes and plans
// ============================
// The API for writing a batching rule isn't stable. In the future, we'd like
// to think about the problem of translating these batching rules to TorchScript.
// Ideally batching rules in eager mode vs TorchScript would look pretty similar,
// if not use the same mechanism. In order to accomplish that we might have to
// do some refactoring.

Tensor sum_batching_rule(const Tensor& self, IntArrayRef dims, bool keepdim, optional<ScalarType> dtype) {
  Tensor self_;
  std::bitset<kVmapNumLevels> levels;
  std::tie(self_, levels) = materializeBatchDimsAtFront(self);
  auto num_bdims = levels.count();
  auto self_dim = self.dim();
  auto actual_dims = transformIntVector(
      dims,
      [&](int64_t dim) {
        return maybe_wrap_dim(dim, self_dim) + num_bdims;
      });
  return makeBatchedFromLevels(at::sum(self_, actual_dims, keepdim, dtype), levels);
}

void batchedTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "NYI: Calling ", op.schema().name(), " inside of vmap");
}

TORCH_LIBRARY_IMPL(_, Batched, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorFallback>());
}

TORCH_LIBRARY_IMPL(aten, Batched, m) {
  m.impl_UNBOXED("sum.dim_IntList", sum_batching_rule);
}

} // namespace at
