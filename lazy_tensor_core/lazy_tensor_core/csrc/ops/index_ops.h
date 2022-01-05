// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containg kLong or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]

#pragma once

#include <ATen/core/Tensor.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {

struct CanonicalIndexInfo {
  at::Tensor base;
  std::vector<at::Tensor> indices;
  // The permutation to be applied to the result. This is needed for indexed
  // updates, since a permutation is applied to the base to bring non-null
  // indices to front. This is the inverse of that permutation.
  std::vector<int64_t> result_permutation;
  // The dimension number at which indexing starts.
  int64_t start_dim = 0;
};

// Transform the given base and indices to a form supported by the torch::lazy::LazyTensor
// index implementation. Input indices are reordered so that non-null indices
// are first and the tail of null indices is dropped. The dimensions of the base
// are reordered to be consistent with this reordering.
CanonicalIndexInfo GetCanonicalIndexInfo(
    const at::Tensor& base,
    const c10::List<c10::optional<at::Tensor>>& orig_indices);

// Expands a rank <= 1 tensor to rank 1, if necessary.
torch::lazy::Value EnsureRank1(const torch::lazy::Value& index);

torch::lazy::NodePtr IndexFill(const torch::lazy::LazyTensor& base, int64_t dim, const torch::lazy::LazyTensor& index,
                               const at::Scalar& value);

torch::lazy::NodePtr IndexFill(const torch::lazy::LazyTensor& base, int64_t dim,
                               const torch::lazy::LazyTensor& index,
                               const torch::lazy::LazyTensor& value);

torch::lazy::Value IndexAdd(const torch::lazy::LazyTensor& base, int64_t dim,
                            const torch::lazy::LazyTensor& index, const torch::lazy::LazyTensor& source);

torch::lazy::Value IndexCopy(const torch::lazy::LazyTensor& base, int64_t dim,
                             const torch::lazy::LazyTensor& index, const torch::lazy::LazyTensor& source);

}  // namespace torch_lazy_tensors
