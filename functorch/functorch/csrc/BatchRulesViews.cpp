#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

// Note [Adding vmap support for an operator]
// Hey there! So you have an operator and you want to get it to work with vmap.
// For example, let's say you just invented the `sum.int` operator and want to make
// it so that the following works.
// >>> tensor = torch.randn(B, 3)
// >>> vmap(torch.sum, (0, None))(tensor, 0)` works
// There are three main ways to do so.
//
// Note [Writing batch rule for out-of-place operators]
// If your operator is out-of-place, you can write a batch rule for it.
// The batch rule defines how to perform the operator on inputs where each
// Tensor input may have an additional dimension that is being vmapped over.
// We refer to this dimension as the *batch dimension* or bdim for short.
//
// For example, let's consider writing a batch rule for
// `Tensor sum(const Tensor& self, int64_t dim)`. The signature of the
// batch rule has an additional optional<int64_t> argument after each
// Tensor argument and return. So, in this case, the batch rule has signature
//   tuple<Tensor,optional<int64_t>> sum_batch_rule(
//       const Tensor& self, optional<int64_t> self_bdim, int64_t dim);
//
// The vmap call above invokes the batch rule with `self = tensor`,
// `self_bdim = 0`, and `dim = 0`. Note that there are **no BatchedTensors**
// involved in this case; there exists some plumbing that automatically unwraps
// BatchedTensors before calling the batch rule.
// 
// To write the logic of the batch rule: think about the semantics of the
// `sum` operation if `self` had an additional dimension (indicated by self_bdim):
// - If `self_bdim` is null, then we just do `result = self.sum(dim)` as usual
// - If `self_bdim` is not-null, then we need to modify `dim`. `dim` is equal
//   to whatever the user passed in (0 in this case), but we should actually
//   perform the reduction over dimension 1 and do `result = self.sum(1)`
//   because dim 0 is being vmapped over.
// Finally, we return the result as well as a new bdim
// - If `self_bdim` is null, then there's no batch dim in the result.
// - If `self_bdim` is not-null, then we return where the bdim is.
//   Since we invoked `result = self.sum(1)`, the bdim is still at dim 0.
//
// Now that we have written `sum_batch_rule`, we have to register it inside a
// TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
//     ...
//     VMAP_SUPPORT("sum.int", sum_batch_rule);
//     ...
//   }
//  
// Note [Reusing batch rules to add vmap support for a complicated operator]
// Can't figure out how to write a batch rule for a big operation? If the
// operation can be expressed as a composition of other operations that do have
// batch rules, then that is another way to add vmap support. For example,
// consider the following schema 
//   func: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1)
// and assume we already have batching rules for basic arithmetic operators.
// 
// To add vmap support, define a decomposition using the same signature:
//   Tensor addcmul_decomp(const Tensor& self, const Tensor& tensor1,
//                         const Tensor& tensor2, const Scalar& value) {
//     auto product = torch.mul(tensor1, tensor2);
//     return torch.add(self, product, value);
//   }
// And register it inside a TORCH_LIBRARY_IMPL block:
//   TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
//     ...
//     m.impl("addcmul", addcmul_decomp);
//     ...
//   }
//
// Note [Writing batch rule for in-place operators]
// TODO: This is kinda complicated. Saving this for a future date.

std::tuple<Tensor, optional<int64_t>> flatten_batch_rule(
    const Tensor& self, 
    optional<int64_t> self_bdim,
    int64_t start_dim, int64_t end_dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  start_dim = getPhysicalDim(self_, self_bdim.has_value(), start_dim);
  end_dim = getPhysicalDim(self_, self_bdim.has_value(), end_dim);
  return { at::flatten(self_, start_dim, end_dim), valIfNonempty(self_bdim, 0) };
}

std::tuple<Tensor,optional<int64_t>> unsqueeze_batch_rule(
    const Tensor& self,
    optional<int64_t> self_bdim,
    int64_t dim) {
  auto self_ = moveBatchDimToFront(self, self_bdim); 
  auto rank = rankWithoutBatchDim(self, self_bdim);
  dim = maybe_wrap_dim(dim, rank + 1);
  if (self_bdim) {
    dim += 1;
  }
  return { self_.unsqueeze(dim), valIfNonempty(self_bdim, 0) };
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("flatten.using_ints", flatten_batch_rule);
  VMAP_SUPPORT("unsqueeze", unsqueeze_batch_rule);
}

}}
