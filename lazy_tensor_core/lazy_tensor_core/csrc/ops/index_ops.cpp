#include "lazy_tensor_core/csrc/ops/index_ops.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/ts_backend/ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/ts_backend/ops/expand.h>
#include <torch/csrc/lazy/ts_backend/ops/scalar.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include "lazy_tensor_core/csrc/ops/index_along_dim.h"
#include "lazy_tensor_core/csrc/ops/index_get.h"
#include "lazy_tensor_core/csrc/ops/index_put.h"
#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace {
using namespace torch_lazy_tensors::ir;

void CheckIndexTensorTypes(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  for (const c10::optional<at::Tensor>& tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      at::ScalarType scalar_type = tensor->scalar_type();
      if (scalar_type != at::kLong && scalar_type != at::kByte &&
          scalar_type != at::kBool) {
        LOG(ERROR) << "Tensors used as indices must be long, byte or boolean "
                      "tensors, found scalar type: "
                   << scalar_type;
      }
    }
  }
}

// Expands byte tensors (masks) into the equivalent indexing by LongTensors.
// This is a version of at::native::expandByteTensors with style adjustments.
std::vector<at::Tensor> ExpandByteTensors(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  std::vector<at::Tensor> result;
  for (const c10::optional<at::Tensor>& index : indices) {
    if (index.has_value() && (index->scalar_type() == at::kByte ||
                              index->scalar_type() == at::kBool)) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self.
      for (int64_t j = 0; j < index->dim(); j++) {
        int64_t src_idx = result.size() + j;
        CHECK_EQ(index->size(j), self.size(src_idx))
            << "The shape of the mask " << index->sizes() << " at index " << j
            << " does not match the shape of the indexed tensor "
            << self.sizes() << " at index " << src_idx;
      }
      // Replace with nonzeros.
      auto nonzero = index->nonzero();
      for (int64_t j = 0; j < index->dim(); j++) {
        result.emplace_back(nonzero.select(1, j));
      }
    } else {
      result.emplace_back(index.value_or(at::Tensor()));
    }
  }
  return result;
}

struct IndexAdjacencyInfo {
  bool contiguous_non_null = false;
  int64_t start_dim = 0;
};

// Checks whether all the non-null tensors are adjacent, in which case we must
// not permute the base and instead treat the null tensors prefix as a no-op.
// Replicates the behavior of at::native::hasContiguousSubspace and also returns
// the position of the first non-null index.
IndexAdjacencyInfo GetIndexAdjacencyInfo(at::TensorList indices) {
  auto is_defined = [](const at::Tensor& tensor) { return tensor.defined(); };
  auto is_null = [](const at::Tensor& tensor) { return !tensor.defined(); };
  auto start = std::find_if(indices.begin(), indices.end(), is_defined);
  auto stop = std::find_if(indices.rbegin(), indices.rend(), is_defined);
  auto it = std::find_if(start, stop.base(), is_null);
  int64_t start_dim = std::distance(indices.begin(), start);
  return {it == stop.base(), start_dim};
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor and
// the reordered indices. For example:
//  TransposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b}
//
// This is a simplified version of at::native::transposeToFront which better
// fits our requirements.
CanonicalIndexInfo TransposeToFront(at::Tensor base, at::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposed_indices;
  size_t base_rank = base.dim();
  dims.reserve(base_rank);
  CHECK_LE(indices.size(), base_rank);
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposed_indices.emplace_back(indices[i]);
    }
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
    }
  }
  for (size_t i = indices.size(); i < base_rank; ++i) {
    dims.push_back(i);
  }
  IndexAdjacencyInfo adjacency_info = GetIndexAdjacencyInfo(indices);
  if (adjacency_info.contiguous_non_null) {
    return {base, std::move(transposed_indices),
            torch::lazy::Iota<int64_t>(base_rank),
            adjacency_info.start_dim};
  }
  return {base.permute(dims), std::move(transposed_indices),
          torch::lazy::InversePermutation(torch::lazy::ToI64Vector(dims)), 0};
}

torch::lazy::NodePtr IndexFillOp(const torch::lazy::Value& buffer, int64_t dim,
                    const torch::lazy::Value& index,
                    const torch::lazy::Value& value) {
  torch::lazy::Value index_rank1 = EnsureRank1(index);
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::IndexAlongDim>(
      torch::lazy::OpKind(at::aten::index_fill), buffer, index_rank1, value,
      dim);
  torch::lazy::TsNodeSetShapeDeferred(
      node, [&]() { return compiler::InferShape(node.get()); });
  return node;
}

torch::lazy::NodePtr IndexAddOp(const torch::lazy::Value& buffer, int64_t dim,
                                const torch::lazy::Value& index,
                                const torch::lazy::Value& source) {
  torch::lazy::Value index_rank1 = EnsureRank1(index);
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::IndexAlongDim>(
      torch::lazy::OpKind(at::aten::index_add), buffer, index_rank1, source,
      dim);
  torch::lazy::TsNodeSetShapeDeferred(
      node, [&]() { return compiler::InferShape(node.get()); });
  return node;
}

torch::lazy::NodePtr IndexCopyOp(const torch::lazy::Value& buffer, int64_t dim,
                                 const torch::lazy::Value& index,
                                 const torch::lazy::Value& source) {
  torch::lazy::Value index_rank1 = EnsureRank1(index);
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::IndexAlongDim>(
      torch::lazy::OpKind(at::aten::index_copy), buffer, index_rank1, source,
      dim);
  torch::lazy::TsNodeSetShapeDeferred(
      node, [&]() { return compiler::InferShape(node.get()); });
  return node;
}

}  // namespace

CanonicalIndexInfo GetCanonicalIndexInfo(
    const at::Tensor& base,
    const c10::List<c10::optional<at::Tensor>>& orig_indices) {
  CheckIndexTensorTypes(orig_indices);
  // First expand ByteTensor (boolean masks) into 1 or more LongTensors, then
  // broadcast all index tensors together.
  auto indices = at::expand_outplace(ExpandByteTensors(base, orig_indices));
  // If the non-null indices are not all adjacent, transpose base and indices
  // together so that they're adjacent at the front.
  CanonicalIndexInfo canonical_index_info = TransposeToFront(base, indices);
  // Ensure indices are on the same device as the base.
  for (size_t i = 0; i < canonical_index_info.indices.size(); i++) {
    if (canonical_index_info.indices[i].device() != base.device()) {
      canonical_index_info.indices[i] =
          canonical_index_info.indices[i].to(base.device());
    }
  }
  return canonical_index_info;
}

torch::lazy::Value EnsureRank1(const torch::lazy::Value& index) {
  CHECK_LE(torch::lazy::GetShapeFromTsValue(index).dim(), 1);
  return torch::lazy::GetShapeFromTsValue(index).dim() == 0
             ? torch::lazy::MakeNode<torch::lazy::Expand>(
                   index, std::vector<int64_t>{1},
                   /*is_scalar_expand=*/false)
             : index;
}

torch::lazy::NodePtr IndexFill(const torch::lazy::LazyTensor& base, int64_t dim, const torch::lazy::LazyTensor& index,
                  const at::Scalar& value) {
  CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long, but it is "
      << index.dtype();
  CHECK_LE(index.shape().Get().dim(), 1)
      << "Fill index is supposed to be a vector";
  return IndexFillOp(
      base.GetIrValue(), dim, index.GetIrValue(),
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForScalar(
          value, base.shape().Get().scalar_type(), base.GetDevice()));
}

torch::lazy::NodePtr IndexFill(const torch::lazy::LazyTensor& base, int64_t dim,
                               const torch::lazy::LazyTensor& index,
                               const torch::lazy::LazyTensor& value) {
  CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long, but it is "
      << index.dtype();
  CHECK_LE(index.shape().Get().dim(), 1)
      << "Fill index is supposed to be a vector";
  CHECK_EQ(value.shape().Get().dim(), 0)
      << "Fill only supports a 0-dimensional value tensor";
  return IndexFillOp(base.GetIrValue(), dim, index.GetIrValue(),
                     value.GetIrValue());
}

torch::lazy::Value IndexAdd(const torch::lazy::LazyTensor& base, int64_t dim,
                            const torch::lazy::LazyTensor& index, const torch::lazy::LazyTensor& source) {
  CHECK(index.dtype() == at::ScalarType::Long ||
        index.dtype() == at::ScalarType::Int)
      << "Add index is expected to be of scalar type Long or scalar type Int, "
         "but it is "
      << index.dtype();
  CHECK_LE(index.shape().Get().dim(), 1)
      << "Add index is supposed to be a vector";
  return IndexAddOp(base.GetIrValue(), dim, index.GetIrValue(),
                    source.GetIrValue());
}

torch::lazy::Value IndexCopy(const torch::lazy::LazyTensor& base, int64_t dim,
                             const torch::lazy::LazyTensor& index,
                             const torch::lazy::LazyTensor& source) {
  CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Copy index is expected to be of scalar type Long, but it is "
      << index.dtype();
  CHECK_LE(index.shape().Get().dim(), 1)
      << "Copy index is supposed to be a vector";
  return IndexCopyOp(base.GetIrValue(), dim, index.GetIrValue(),
                     source.GetIrValue());
}

}  // namespace torch_lazy_tensors
