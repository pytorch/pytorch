#include <ATen/BatchingRules.h>
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
// TODO(rzou): A lot of the description references things that don't exist yet.
// Fill them in/change them as necessary.
//
// ==========================================
// When and why should I add a batching rule?
// ==========================================
// When you are adding a new operator, you'll need to add a batching rule so
// that vmap can work efficiently with said operator. If you do not, we'll attempt
// to generate a slow fallback for the batching rule. If the operator is composite
// (with respect to autograd), then... TODO(rzou): there should be a mechanism to
// register a fallthrough where we just run the composite op directly on the
// BatchedTensor.
//
// =======================
// Batching rule signature
// =======================
// The signature of a batching rule should look like the C++ signature of its
// operator, but with an additional BatchDimsRef argument added after each Tensor
// argument and additional BatchDims argument added after each Tensor return.
// The BatchDimsRef/BatchDims holds the metadata for which dimensions in its
// respective Tensor are batch dimensions. For example:
// `
// Operator declaration:
//   Tensor add(const Tensor& self, const Tensor& other, Scalar alpha);
//
// Batching rule declaration:
//   std::pair<Tensor,BatchDims> add_batching_rule(
//       const Tensor& self, BatchDimsRef self_bdims,
//       const Tensor& other, BatchDimsRef other_bdims,
//       Scalar alpha);
//
// ====================================
// How to implement a new batching rule
// ====================================
// Most batching rules will look like the following:
// - Move all batch dims to the front of the input tensors
// - If there are multiple tensors, align their batch dimensions by level
// - Once the tensors are aligned, figure out how to call the at:: op that
//   corresponds to the operator with additional batch dims
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

std::pair<Tensor,BatchDims> sum_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims, bool keepdim, c10::optional<ScalarType> dtype) {
  // NB: We don't really need to move the batch dims to the front.
  // One alternative way to do this is to keep them where they are and compute
  // the required `dims` to reduce over. However, assuming that the batch
  // dims are at front greatly simplifies the `dims` calculation and moving
  // them there is relatively cheap.
  auto self_ = moveBatchDimsToFront(self, self_bdims);
  auto result_bdims = moveBatchDimsToFront(self_bdims);
  auto tensor_dims = self_.dim() - self_bdims.size();

  // Real dims to reduce over
  std::vector<int64_t> actual_dims;
  actual_dims.reserve(dims.size());
  for (int64_t dim : dims) {
    dim = maybe_wrap_dim(dim, tensor_dims);
    actual_dims.push_back(dim + self_bdims.size());
  }

  auto result = at::sum(self_, actual_dims, keepdim, dtype);
  return { result, result_bdims };
}

}

