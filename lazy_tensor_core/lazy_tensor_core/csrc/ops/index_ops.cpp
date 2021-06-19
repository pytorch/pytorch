#include "lazy_tensor_core/csrc/ops/index_ops.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/index_along_dim.h"
#include "lazy_tensor_core/csrc/ops/index_get.h"
#include "lazy_tensor_core/csrc/ops/index_put.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/permutation_util.h"

namespace torch_lazy_tensors {
namespace {

void CheckIndexTensorTypes(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  for (const c10::optional<at::Tensor>& tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      at::ScalarType scalar_type = tensor->scalar_type();
      if (scalar_type != at::kLong && scalar_type != at::kByte &&
          scalar_type != at::kBool) {
        LTC_ERROR() << "Tensors used as indices must be long, byte or boolean "
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
        LTC_CHECK_EQ(index->size(j), self.size(src_idx))
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
  lazy_tensors::int64 start_dim = 0;
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
  lazy_tensors::int64 start_dim = std::distance(indices.begin(), start);
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
  LTC_CHECK_LE(indices.size(), base_rank);
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
            lazy_tensors::util::Iota<lazy_tensors::int64>(base_rank),
            adjacency_info.start_dim};
  }
  return {base.permute(dims), std::move(transposed_indices),
          lazy_tensors::InversePermutation(Helpers::I64List(dims)), 0};
}

// Wraps index tensors once into the [0, dim_size) interval, where dim_size is
// the size of the current indexed dimension.
std::vector<LazyTensor> WrapIndicesOnce(
    const LazyTensor& base, lazy_tensors::Span<const LazyTensor> indices,
    int start_dim) {
  std::vector<LazyTensor> canonical_indices;
  auto base_shape_ref = base.shape();
  LTC_CHECK_LE(indices.size(), base_shape_ref.get().rank());
  for (size_t dim_idx = 0; dim_idx < indices.size(); ++dim_idx) {
    const LazyTensor& dim_index = indices[dim_idx];
    int64_t dim_size = base_shape_ref.get().dimensions(dim_idx + start_dim);
    LazyTensor wrapped_dim_index = LazyTensor::Create(
        dim_index.GetIrValue() +
            LazyTensor::GetIrValueForScalar(dim_size, dim_index.shape(),
                                            base.GetDevice()),
        base.GetDevice());
    LazyTensor wrap_cond =
        LazyTensor::lt(indices[dim_idx], at::Scalar(int64_t(0)));
    canonical_indices.push_back(
        LazyTensor::where(wrap_cond, wrapped_dim_index, dim_index));
  }
  return canonical_indices;
}

ir::NodePtr IndexFillOp(const ir::Value& buffer, lazy_tensors::int64 dim,
                        const ir::Value& index, const ir::Value& value) {
  ir::Value index_rank1 = EnsureRank1(index);
  ir::NodePtr node = ir::MakeNode<ir::ops::IndexAlongDim>(
      ir::OpKind(at::aten::index_fill), buffer, index_rank1, value, dim);
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

ir::NodePtr IndexAddOp(const ir::Value& buffer, lazy_tensors::int64 dim,
                       const ir::Value& index, const ir::Value& source) {
  ir::Value index_rank1 = EnsureRank1(index);
  ir::NodePtr node = ir::MakeNode<ir::ops::IndexAlongDim>(
      ir::OpKind(at::aten::index_add), buffer, index_rank1, source, dim);
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
  return node;
}

ir::NodePtr IndexCopyOp(const ir::Value& buffer, lazy_tensors::int64 dim,
                        const ir::Value& index, const ir::Value& source) {
  ir::Value index_rank1 = EnsureRank1(index);
  ir::NodePtr node = ir::MakeNode<ir::ops::IndexAlongDim>(
      ir::OpKind(at::aten::index_copy), buffer, index_rank1, source, dim);
  node->SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(node.get()); });
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

ir::Value EnsureRank1(const ir::Value& index) {
  LTC_CHECK_LE(index->shape().rank(), 1);
  return index->shape().rank() == 0
             ? ir::MakeNode<ir::ops::Expand>(
                   index, std::vector<lazy_tensors::int64>{1},
                   /*is_scalar_expand=*/false)
             : index;
}

LazyTensor IndexByTensors(const LazyTensor& base,
                          lazy_tensors::Span<const LazyTensor> indices,
                          lazy_tensors::int64 start_dim) {
  if (indices.empty()) {
    return base;
  }
  auto canonical_indices = WrapIndicesOnce(base, indices, start_dim);
  lazy_tensors::int64 indices_rank =
      canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single gather.
  LazyTensor indices_nd = LazyTensor::stack(canonical_indices, indices_rank);
  return LazyTensor::Create(
      ir::MakeNode<ir::ops::IndexGet>(base.GetIrValue(),
                                      indices_nd.GetIrValue(), start_dim),
      base.GetDevice(), base.dtype());
}

ir::Value IndexPutByTensors(
    const LazyTensor& base, lazy_tensors::Span<const LazyTensor> indices,
    lazy_tensors::int64 start_dim, const LazyTensor& values, bool accumulate,
    lazy_tensors::Span<const lazy_tensors::int64> result_permutation) {
  if (indices.empty()) {
    return base.GetIrValue();
  }
  auto canonical_indices = WrapIndicesOnce(base, indices, start_dim);
  lazy_tensors::int64 indices_rank =
      canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single scatter.
  LazyTensor indices_nd = LazyTensor::stack(canonical_indices, indices_rank);
  return ir::MakeNode<ir::ops::Permute>(
      ir::MakeNode<ir::ops::IndexPut>(base.GetIrValue(),
                                      indices_nd.GetIrValue(), start_dim,
                                      values.GetIrValue(), accumulate),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(result_permutation));
}

ir::NodePtr IndexFill(const LazyTensor& base, lazy_tensors::int64 dim,
                      const LazyTensor& index, const at::Scalar& value) {
  LTC_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long, but it is "
      << index.dtype();
  LTC_CHECK_LE(index.shape().get().rank(), 1)
      << "Fill index is supposed to be a vector";
  return IndexFillOp(
      base.GetIrValue(), dim, index.GetIrValue(),
      LazyTensor::GetIrValueForScalar(value, base.shape().get().element_type(),
                                      base.GetDevice()));
}

ir::NodePtr IndexFill(const LazyTensor& base, lazy_tensors::int64 dim,
                      const LazyTensor& index, const LazyTensor& value) {
  LTC_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long, but it is "
      << index.dtype();
  LTC_CHECK_LE(index.shape().get().rank(), 1)
      << "Fill index is supposed to be a vector";
  LTC_CHECK_EQ(value.shape().get().rank(), 0)
      << "Fill only supports a 0-dimensional value tensor";
  return IndexFillOp(base.GetIrValue(), dim, index.GetIrValue(),
                     value.GetIrValue());
}

ir::Value IndexAdd(const LazyTensor& base, lazy_tensors::int64 dim,
                   const LazyTensor& index, const LazyTensor& source) {
  LTC_CHECK(index.dtype() == at::ScalarType::Long ||
            index.dtype() == at::ScalarType::Int)
      << "Add index is expected to be of scalar type Long or scalar type Int, "
         "but it is "
      << index.dtype();
  LTC_CHECK_LE(index.shape().get().rank(), 1)
      << "Add index is supposed to be a vector";
  return IndexAddOp(base.GetIrValue(), dim, index.GetIrValue(),
                    source.GetIrValue());
}

ir::Value IndexCopy(const LazyTensor& base, lazy_tensors::int64 dim,
                    const LazyTensor& index, const LazyTensor& source) {
  LTC_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Copy index is expected to be of scalar type Long, but it is "
      << index.dtype();
  LTC_CHECK_LE(index.shape().get().rank(), 1)
      << "Copy index is supposed to be a vector";
  return IndexCopyOp(base.GetIrValue(), dim, index.GetIrValue(),
                     source.GetIrValue());
}

}  // namespace torch_lazy_tensors
