#include <ATen/ExpandUtils.h>

namespace at {

std::vector<int64_t> infer_size(IntArrayRef a, IntArrayRef b) {
  size_t dimsA = a.size();
  size_t dimsB = b.size();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizes(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;
    ptrdiff_t dimB = dimsB - 1 - offset;
    int64_t sizeA = (dimA >= 0) ? a[dimA] : 1;
    int64_t sizeB = (dimB >= 0) ? b[dimB] : 1;

    AT_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);

      // 1s map to the other size (even 0).
      expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
  int64_t ndim = sizes.size();
  int64_t tensor_dim = tensor_sizes.size();

  if (tensor_dim == 0) {
    std::vector<int64_t> expandedStrides(ndim, 0);
    return std::tuple<std::vector<int64_t>, std::vector<int64_t>>(
        sizes.vec(), expandedStrides);
  }
  std::vector<int64_t> expandedSizes(ndim);
  std::vector<int64_t> expandedStrides(ndim);

  // create a new geometry for the tensors
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_dim - 1 - offset;
    int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
    int64_t stride = (dim >= 0) ? tensor_strides[dim]
                                : expandedSizes[i + 1] * expandedStrides[i + 1];
    int64_t targetSize = sizes[i];
    if (targetSize == -1) {
      AT_CHECK(
          dim >= 0,
          "The expanded size of the tensor (",
          targetSize,
          ") isn't allowed in a leading, non-existing dimension ",
          i);
      targetSize = size;
    }
    if (size != targetSize) {
      AT_CHECK(
          size == 1,
          "The expanded size of the tensor (",
          targetSize,
          ") must match the existing size (",
          size,
          ") at non-singleton dimension ",
          i,
          ".  Target sizes: ",
          sizes,
          ".  Tensor sizes: ",
          tensor_sizes);
      size = targetSize;
      stride = 0;
    }
    expandedSizes[i] = size;
    expandedStrides[i] = stride;
  }
  return std::tuple<std::vector<int64_t>, std::vector<int64_t>>(
      expandedSizes, expandedStrides);
}

} // namespace at
