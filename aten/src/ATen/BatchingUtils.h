#include <ATen/BatchedTensorImpl.h>

namespace at {

/*
 * Utility functions used to implement batching rules.
 *
 * NB: All of these do NOT accept Tensors backed by BatchedTensor, unless
 * otherwise specified. These APIs usually operate on "unpacked BatchedTensors",
 * i.e. a (value Tensor, BatchDims) pair. This is for performance reasons:
 * we do not want to always wrap and unwrap BatchedTensors; we try to
 * only unwrap once per input tensor per operator and wrap once per output
 * tensor per operator.
 */ 

// If the input is a Tensor backed with a BatchedTensorImpl, then
// this function returns the underlying Tensor and BatchDims.
// If the input is a Tensor backed with regular TensorImpl, then
// this function returns the tensor and empty BatchDims.
TORCH_API std::pair<Tensor, BatchDimsRef> unpackBatched(const Tensor& self);

// Moves the specified BatchDims to the front of `self`, ordered by their level.
// Returns a view of the original tensor if any dims were moved; otherwise
// returns the original tensor.
//
// For example:
//   moveBatchDimsToFront(ones(2, 3, 5), [(lvl=1, dim=2), (lvl=2, dim=1)])
// would return a permuted view of size [5, 3, 2].
TORCH_API Tensor moveBatchDimsToFront(const Tensor& self, BatchDimsRef bdims);

// Reindexes batch dims (out-of-place) assuming they appear at the front of
// a tensor.
// For example:
//   moveBatchDimsToFront([(lvl=1, dim=2), (lvl=3, dim=1)])
// returns:
//   [(lvl=1, dim=0), (lvl=3, dim=0)]
TORCH_API BatchDims moveBatchDimsToFront(BatchDimsRef bdims);

}
