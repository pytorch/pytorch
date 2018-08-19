// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value)
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


#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/ExpandUtils.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace at { namespace native {

[[noreturn]]
static void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  std::stringstream ss;
  ss << "The shape of the mask " << mask.sizes() << " at index " << maskIdx;
  ss << " does not match the shape of the indexed tensor " << self.sizes();
  ss << " at index " << idx;
  throw std::runtime_error(ss.str());
}

static void checkIndexTensorTypes(TensorList indices) {
  for (auto& tensor : indices) {
    if (tensor.defined()) {
      auto& type = tensor.type();
      auto scalarType = type.scalarType();
      if (scalarType != kLong && scalarType != kByte) {
        throw std::runtime_error("tensors used as indices must be long or byte tensors");
      }
    }
  }
}

static std::vector<std::tuple<Tensor, bool>>
expandByteTensors(const Tensor & self, TensorList indices) {
  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
  std::vector<std::tuple<Tensor, bool>> result;
  for (auto & index : indices) {
    if (index.type().scalarType() == kByte) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self
      for (int64_t j = 0; j < index.dim(); j++) {
        int64_t srcIdx = result.size() + j;
        if (index.size(j) != self.size(srcIdx)) {
          invalid_mask(self, srcIdx, index, j);
        }
      }
      // Replace with nonzeros
      auto nonzero = index.nonzero();
      for (int64_t j = 0; j < index.dim(); j++) {
        result.emplace_back(nonzero.select(1, j), false);
      }
    } else {
      result.emplace_back(index, true);
    }
  }
  return result;
}

static bool hasContiguousSubspace(const std::vector<std::tuple<Tensor, bool>> &tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const std::tuple<Tensor, bool> & tensor){
    return std::get<0>(tensor).defined();
  };
  auto isNull = [](const std::tuple<Tensor, bool> & tensor){
    return !std::get<0>(tensor).defined();
  };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
//  transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static std::tuple<Tensor, std::vector<std::tuple<Tensor, bool>>>
transposeToFront(Tensor self, const std::vector<std::tuple<Tensor, bool>> &indices) {
  std::vector<int64_t> dims;
  std::vector<std::tuple<Tensor, bool>> transposedIndices;
  dims.reserve(self.dim());
  for (int64_t i = 0; i < self.dim(); i++) {
    const Tensor &t = std::get<0>(indices[i]);
    if (t.defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(t, std::get<1>(indices[i]));
    }
  }
  for (int64_t i = 0; i < self.dim(); i++) {
    if (!std::get<0>(indices[i]).defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(Tensor(), std::get<1>(indices[i]));
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

// Unsqueezes src `before` times at the front and `after` times at the end
static Tensor unsqueezeN(const Tensor & src, int64_t before, int64_t after) {
  auto srcSizes = src.sizes();
  auto nDim = src.dim();
  std::vector<int64_t> sizes(nDim + before + after, 1);
  for (int64_t i = 0; i < nDim; i++) {
    sizes[i + before] = srcSizes[i];
  }
  return src.view(sizes);
}

Tensor _wrap_index_once(const Tensor & self, int64_t dim, int64_t dim_size) {
  if (self.numel() != 0) {
    auto max_idx = self.max().toCLong();
    auto min_idx = self.min().toCLong();
    if (max_idx >= dim_size) {
      AT_ERROR("index ", max_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
    if (min_idx < -dim_size) {
      AT_ERROR("index ", min_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
  }
  return self.remainder(dim_size);
}

static Tensor computeLinearIndex(const Tensor & src,
const std::vector<std::tuple<Tensor, bool>> &indices) {
  auto strides = computeLinearStride(src);
  Type& longType = src.type().toScalarType(kLong);

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1;
  for (int64_t i = 0; i < src.dim(); i++) {
    const Tensor &t = std::get<0>(indices[i]);
    bool needs_wrapdim = std::get<1>(indices[i]);
    if (t.defined()) {
      // Cast index to the longType matching src's backend
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = ((needs_wrapdim ? t._wrap_index_once(i, src.size(i)) : t)
                      * strides[i]).toType(longType);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
      }
    } else if (linearIndex.defined()) {
      emptyAfter++;
      nElemAfter *= src.size(i);
    } else {
      emptyBefore++;
      nElemBefore *= src.size(i);
    }
  }

  // Compute the linear indices for the parts of the tensor not being indexed
  Tensor beforeIndex;
  if (emptyBefore > 0) {
    auto index = at::arange(0, nElemBefore, longType) * strides[emptyBefore - 1];
    index = index.view(src.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(index, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    auto index = at::arange(0, nElemAfter, longType);
    index = index.view(src.sizes().slice(src.dim() - emptyAfter, emptyAfter));
    afterIndex = unsqueezeN(index, linearIndex.dim() + emptyBefore, 0);
  }

  // Sum with broadcasting to compute the full index
  linearIndex = unsqueezeN(linearIndex, emptyBefore, emptyAfter);
  if (beforeIndex.defined()) {
    linearIndex = linearIndex + beforeIndex;
  }
  if (afterIndex.defined()) {
    linearIndex = linearIndex + afterIndex;
  }
  return linearIndex;
}

static void _expand_outplace(std::vector<std::tuple<Tensor, bool>> &tensors) {
  std::vector<Tensor> _tensors;
  for (auto &t : tensors) {
    _tensors.push_back(std::get<0>(t));
  }
  _tensors = expand_outplace(_tensors);
  for (int i = 0; i < tensors.size(); i++) {
    std::get<0>(tensors[i]) = _tensors[i];
  }
}

static std::tuple<Tensor, Tensor> makeLinearIndex(Tensor self, TensorList orig) {
  checkIndexTensorTypes(orig);
  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices = expandByteTensors(self, orig);
  // next broadcast all index tensors together
  _expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back(Tensor(), false);
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  auto linearIndex = computeLinearIndex(self, indices);
  return std::make_tuple(self, linearIndex);
}

Tensor index(const Tensor & self, TensorList indices) {
  if (indices.size() > (size_t)self.dim()) {
   AT_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }

  Tensor src, linearIndex;
  std::tie(src, linearIndex) = makeLinearIndex(self, indices);
  return src.take(linearIndex);
}

Tensor index_put(const Tensor & self, TensorList indices, const Tensor & value) {
  if (indices.size() > (size_t)self.dim()) {
   AT_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }

  Tensor src, linearIndex, expandedValue;
  std::tie(src, linearIndex) = makeLinearIndex(self, indices);
  std::tie(expandedValue) = expand_inplace(linearIndex, value);
  Tensor dst = src.clone();
  return dst.put_(linearIndex, expandedValue);
}

Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & value) {
  if (indices.size() > (size_t)self.dim()) {
   AT_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }

  Tensor src, linearIndex, expandedValue;
  std::tie(src, linearIndex) = makeLinearIndex(self, indices);
  std::tie(expandedValue) = expand_inplace(linearIndex, value);
  return src.put_(linearIndex, expandedValue);
}

Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  dim = maybe_wrap_dim(dim, self.dim());

  if (index.dim() >= 2) {
   AT_ERROR(
        "index_copy_(): Index should have dimension 1 or 0 (got ", index.dim(), ")");
  }
  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
   AT_ERROR(
        "index_copy_(): When source is scalar, index should have one element (got ", numIndices, ")");
  }
  if (index.type().scalarType() != ScalarType::Long) {
   AT_ERROR("index_copy_(): Expected LongTensor for index");
  }

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(selfSlicedSizes.begin(), selfSlicedSizes.end(),
                  sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension " << dim;
    ss << " and source slice shape: " << sourceSlicedSizes << " at dimension 0.";
    throw std::runtime_error(ss.str());
  }
  if (source.dim() > 0 && numIndices != source.size(dim)) {
     AT_ERROR(
          "index_copy_(): Number of indices (", numIndices, ") should be equal to source.size(dim) (", source.size(dim), ")");
  }

  return self._indexCopy_(dim, index, source);
}

}} // at::native
