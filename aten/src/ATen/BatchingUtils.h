#include <ATen/BatchedTensorImpl.h>

namespace at {

/*
 * Utility functions used to implement batching rules.
 */ 

// Creates a bitset of all the levels present in `bdims`.
//
// For example:
//   createLevelsBitset([(lvl=1, dim=2), (lvl=3, dim=1)]) -> 1010000...
std::bitset<kVmapNumLevels> createLevelsBitset(BatchDimsRef bdims);

// Produces a new vector of the same size as `arr` where each element of is
// created by calling `func` on  each element of `arr` in order.
// Equivalent to `output = map(func, arr)` in Python.
std::vector<int64_t> transformIntVector(IntArrayRef arr, std::function<int64_t(int64_t)> func);

// Takes a BatchedTensor or a regular Tensor and permutes all of the batch dims of
// the underlying tensor (if they exist) to the front, in the order of their level `.
// Returns the (possibly) permuted tensor, which is a regular Tensor(!!)
// (not a BatchedTensor), and a bitset of the levels that that correspond to the
// levels present in the batch dims.
//
// For example:
//     materializeBatchDimsAtFront(ones({2, 3})) -> ones({2, 3}), 0
// and
//     materializeBatchDimsAtFront(makeBatched(
//         ones({2, 3, 4}), {{/lvl*/=1, dim=1}, {/*lvl*/=3, dim=0}}
// returns
//     tensor = ones({3, 2, 4}), levels = 0101000...
// (so the dimension with level 1 is at the front of the tensor, followed by
// the dimension with level 3).
//
// This transformation is central to the semantics of BatchedTensor. Most
// (unary) batching rules will materialize the batch dimensions at the front
// of the tensor and then perform some operations on the returned tensor.
TORCH_API std::pair<Tensor,std::bitset<kVmapNumLevels>>
materializeBatchDimsAtFront(const Tensor& tensor);

}
