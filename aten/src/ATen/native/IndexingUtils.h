#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

[[noreturn]]
static void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
  TORCH_CHECK_INDEX(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
  " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
}


static std::vector<Tensor> expandTensors(const Tensor & self, TensorList indices) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (const auto & index : indices) {
    if (index.scalar_type() == kByte || index.scalar_type() == kBool) {
      if (index.scalar_type() == kByte) {
        TORCH_WARN("indexing with dtype torch.uint8 is now deprecated," \
        " please use a dtype torch.bool instead.");
      }
      // The sizes of the ByteTensor mask or bool tensor must match the sizes of the
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
        result.emplace_back(nonzero.select(1, j));
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}


static void checkIndexTensorTypes(TensorList indices) {
  for (auto& tensor : indices) {
    if (tensor.defined()) {
      auto scalarType = tensor.scalar_type();
      if (scalarType != kLong && scalarType != kByte && scalarType != kBool) {
          AT_INDEX_ERROR("tensors used as indices must be long, byte or bool tensors");
      }
    }
  }
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
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static std::tuple<Tensor, std::vector<Tensor>>
transposeToFront(Tensor self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

inline std::tuple<Tensor, std::vector<Tensor>, std::vector<int64_t>>
transposeToFrontAndInvPerm(Tensor self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<int64_t> invPerm;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  invPerm.resize(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    invPerm[dims[i]] = i;
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices), std::move(invPerm));
}

struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);

  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};


}}
