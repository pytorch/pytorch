#include <ATen/ExpandUtils.h>

namespace at {

// NOTE: are_expandable did a similar check, please keep them sync if change is needed
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

    TORCH_CHECK(
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
      TORCH_CHECK(
          dim >= 0,
          "The expanded size of the tensor (",
          targetSize,
          ") isn't allowed in a leading, non-existing dimension ",
          i);
      targetSize = size;
    }
    if (size != targetSize) {
      TORCH_CHECK(
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

// This function returns a dense and non-overlapping strides, which keeps the same memory
// layout as the input tensor, based on the shape of input tensor.
// If the input tensor is a dense and non-overlapping tensor, the output strides will
// be the same as input tensor's strides. Otherwise, the input tensor's strides will be
// sorted (see note inside the code), the output strides will be computed based on the
// sorted input tensor's strides and shape, so the input tensor's memory layout is preserved
// as well.
std::vector<int64_t> infer_dense_strides(const Tensor& tensor) {

  // Note: numel() == 0 is also treated as contiguous which is in the scope of
  //       non overlapping and dense, so we return the strides as it as well
  if (tensor.is_non_overlapping_and_dense()) {
    IntArrayRef strides = tensor.strides();
    return std::vector<int64_t>(strides.begin(), strides.end());
  }

  IntArrayRef ori_sizes(tensor.sizes());
  IntArrayRef ori_strides(tensor.strides());

  size_t ndim = tensor.dim();
  DimVector perm(ndim);

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm.rbegin(), perm.rend(), 0);

  // The following sorting algorithm has exactly the same behavior as TensorIterator
  // This is to make sure we have the same stride propagation everywhere.

  // return -1 if dim0 should come before dim1
  // return  1 if dim0 should come after dim1
  // return  0 if comparison is ambiguous
  auto should_swap = [&](size_t dim0, size_t dim1) {
    int64_t stride0 = ori_strides[dim0];
    int64_t stride1 = ori_strides[dim1];

    // if any stride is 0, treat it as ambiguous comparison to
    // keep the same behavior as TensorIterator
    if (stride0 == 0 || stride1 == 0) {
      return 0;
    }

    if (stride0 < stride1) {
      return -1;
    }
    if (stride0 > stride1) {
      return 1;
    }
    // for equal strides, the dimension with smaller size goes front
    if (ori_sizes[dim0] > ori_sizes[dim1]) {
      return 1;
    }
    return 0;
  };

  // Insertion sort (stable) indices in `perm` based on input tensor's stride and shape,
  // all dimensions with 0 stride won't move. This is the same behavior as TensorIterator.
  // eg. Given tensor with size/stride (6, 5, 4, 3, 2)/(6, 0, 120, 0, 1), the initial `perm`
  //     is (4, 3, 2, 1, 0) and the sorted `perm` will be (4, 3, 0, 1, 2)
  for (int i = 1; i < ndim; ++i) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; --dim0) {
      int comparison = should_swap(perm[dim0], perm[dim1]);
      if (comparison > 0) {
        std::swap(perm[dim0], perm[dim1]);
        dim1 = dim0;
      }
      else if (comparison < 0) {
        break;
      }
    }
  }

  // compute output strides which preserves the input tensor's memory layout
  std::vector<int64_t> out_strides(ndim);
  int64_t curr_stride = 1;
  for (size_t i = 0; i < ndim; ++i) {
    int64_t idx = perm[i];
    out_strides[idx] = curr_stride;
    // ori_sizes does not have 0 here since the input tensor will be contiguous otherwise,
    // the function will stoped in the `non overlapping and dense check` block
    if (ori_sizes[idx] != 1) {
      curr_stride *= ori_sizes[idx];
    }
  }
  return out_strides;
}

} // namespace at
