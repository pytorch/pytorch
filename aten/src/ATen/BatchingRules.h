#include <ATen/BatchingUtils.h>

namespace at {

/*
 * This file contains custom-defined batching rules for operators.
 * The batching rules are hooked up to operations on Tensors backed
 * by BatchedTensorImpl. See aten/src/ATen/BatchingRegistrations.cpp
 * for the glue that actually registers a batching rule to BatchedTensor.
 *
 * NB: For now, we'd like to maintain a split between Batching Rules and
 * BatchedTensorImpl registrations; i.e., the Batching Rules should not depend
 * on BatchedTensorImpl directly. The two main reasons for this is:
 * - I'd like to swap out some of the batching rules with TorchScript-defined
 *   batching rules and perform some benchmarking at some point. Keeping them
 *   BatchedTensorImpl-agnostic means the translation can almost be 1:1.
 * - We might want to be able to register batching rules as operators in the future
 *   so they can be overriden by alternate backends.
 */

std::pair<Tensor,BatchDims> sum_batching_rule(
    const Tensor& self, BatchDimsRef self_bdims,
    IntArrayRef dims, bool keepdim, c10::optional<ScalarType> dtype);

}
