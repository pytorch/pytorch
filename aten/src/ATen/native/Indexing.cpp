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

static std::vector<Tensor> expandByteTensors(const Tensor & self, TensorList indices) {
  // Expands byte tensors (masks) into the equivalent indexing by LongTensors
  std::vector<Tensor> result;
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
      auto is_empty = nonzero.numel() == 0;
      for (int64_t j = 0; j < index.dim(); j++) {
        if (is_empty) {
          // We can't call select on an empty tensor so we just create an empty
          // tensor.
          result.emplace_back(nonzero.type().tensor());
        } else {
          result.emplace_back(nonzero.select(1, j));
        }
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}

static bool hasContiguousSubspace(TensorList tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
  auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
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
static std::tuple<Tensor, std::vector<Tensor>>
transposeToFront(Tensor self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (int64_t i = 0; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (int64_t i = 0; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contigous
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

static Tensor computeLinearIndex(const Tensor & src, TensorList indices) {
  auto strides = computeLinearStride(src);
  Type& longType = src.type().toScalarType(kLong);

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1;
  for (int64_t i = 0; i < src.dim(); i++) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's backend
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (indices[i] * strides[i]).toType(longType);
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
    auto index = at::arange(longType, 0, nElemBefore) * strides[emptyBefore - 1];
    index = index.view(src.sizes().slice(0, emptyBefore));
    beforeIndex = unsqueezeN(index, 0, linearIndex.dim() + emptyAfter);
  }
  Tensor afterIndex;
  if (emptyAfter > 0) {
    auto index = at::arange(longType, 0, nElemAfter);
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

static bool hasEmptyTensor(TensorList tensors) {
  for (auto& tensor : tensors) {
    if (tensor.defined() && tensor.numel() == 0) {
      return true;
    }
  }
  return false;
}

static std::tuple<Tensor, Tensor> makeLinearIndex(Tensor self, TensorList orig) {
  checkIndexTensorTypes(orig);
  // first expand ByteTensor (boolean masks) into 1 or more LongTensors
  auto indices = expandByteTensors(self, orig);
  if (hasEmptyTensor(indices)) {
    return std::make_tuple(self, self.type().toScalarType(kLong).tensor());
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
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
   AT_ERROR("too many indices for tensor of dimension %d (got %d)",
      (int)self.dim(), (int)indices.size());
  }

  Tensor src, linearIndex;
  std::tie(src, linearIndex) = makeLinearIndex(self, indices);
  return src.take(linearIndex);
}

Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & value) {
  if (indices.size() > (size_t)self.dim()) {
   AT_ERROR("too many indices for tensor of dimension %d (got %d)",
      (int)self.dim(), (int)indices.size());
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
        "index_copy_(): Index should have dimension 1 or 0 (got %d)",
        (int)index.dim());
  }
  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
   AT_ERROR(
        "index_copy_(): When source is scalar, index should have one element (got %d)",
        (int)numIndices);
  }
  if (source.dim() > 0 && numIndices != source.size(dim)) {
   AT_ERROR(
        "index_copy_(): Number of indices (%d) should be equal to source.size(dim) (%d)",
        (int)numIndices, (int)source.size(dim));
  }
  if (index.type().scalarType() != ScalarType::Long) {
   AT_ERROR("index_copy_(): Expected LongTensor for index");
  }

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = std::vector<int64_t>(self.sizes());
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = std::vector<int64_t>(source.sizes());
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin());
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

  return self._indexCopy_(dim, index, source);
}

}} // at::native
