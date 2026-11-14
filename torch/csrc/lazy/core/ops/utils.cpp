#include <algorithm>

#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch::lazy {

bool StrideIsSupported(c10::ArrayRef<int64_t> stride) {
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<int64_t> GetArrayStridePermutation(c10::ArrayRef<int64_t> stride) {
  std::vector<int64_t> permutation = Iota<int64_t>(stride.size());
  std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b) {
    return stride[a] > stride[b];
  });
  return permutation;
}

Shape MakeDiagonalShape(
    const Shape& shape,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  std::vector<int64_t> dimensions;
  for (const auto dim : c10::irange(shape.dim())) {
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.size(dim));
    }
  }
  int64_t dsize = 0;
  if (offset >= 0) {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1), shape.size(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1) + offset, shape.size(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return Shape(shape.scalar_type(), dimensions);
}

Shape MakePermuteShape(
    const Shape& source_shape,
    c10::ArrayRef<int64_t> permutation) {
  return Shape(
      source_shape.scalar_type(),
      PermuteDimensions(permutation, source_shape.sizes()));
}

Shape MakeSelectShape(
    const Shape& shape,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride) {
  int64_t effective_stride = GetStride(start, end, stride);
  Shape select_shape(shape);
  select_shape.set_size(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

int64_t GetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    TORCH_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

// This is almost like at::inferSqueezeGeometry, but that requires a Tensor
// input and also computes new strides.  This logic seems correct.
std::vector<int64_t> BuildSqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions;
  for (const auto i : c10::irange(dimensions.size())) {
    int64_t dim = dimensions[i];
    if (dim != 1 ||
        (static_cast<int64_t>(i) != squeeze_dim && squeeze_dim >= 0)) {
      output_dimensions.push_back(dim);
    }
  }
  return output_dimensions;
}

std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions(
      dimensions.cbegin(), dimensions.cend());
  output_dimensions.insert(output_dimensions.begin() + squeeze_dim, 1);
  return output_dimensions;
}

} // namespace torch::lazy
