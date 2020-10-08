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

// Given input tensor, compute the dense strides which preserve the same memory layout as
// the input tensor with same sizes.
// Return false, if input tensor has memory overlap.
// Return true otherwise,  the calculated strides will be stored in `out_strides`.
bool infer_dense_strides(const Tensor& tensor, DimVector& out_strides) {

  // Note: numel() == 0 is treated as contiguous which is in the scope of
  //       non overlapping and dense, so we keep the strides as it
  if (tensor.is_non_overlapping_and_dense()) {
    out_strides = tensor.strides();
    return true;
  }

  if (tensor.dim() == 0) {
    out_strides.resize(0);
    return true;
  }

  IntArrayRef ori_sizes(tensor.sizes());
  IntArrayRef ori_strides(tensor.strides());
  DimVector perm(tensor.dim());

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm.rbegin(), perm.rend(), 0);

  // sort index of sizes, strides from back to front in ascending order
  std::sort(perm.begin(), perm.end(), [&](size_t dim0, size_t dim1) {
    // keep the original order if strides are the same
    if (ori_strides[dim0] == ori_strides[dim1]) {
      return dim0 > dim1;
    }
    return ori_strides[dim0] < ori_strides[dim1];
  });

  // check overlap.
  int64_t sz = perm.size() - 1;
  for (size_t i = 0; i < sz; ++i) {
    int64_t idx = perm[i];
    if (ori_strides[idx] == 0) {
      if (ori_sizes[idx] != 1) {
        return false;
      }
      continue;
    }
    // check minimal stride for non-overlapping
    if ((ori_sizes[idx] - 1) * ori_strides[idx] + 1 > ori_strides[perm[i+1]]) {
      return false;
    }
  }
  if (ori_strides[perm[sz]] == 0 && ori_sizes[perm[sz]] != 1) {
    return false;
  }

  // get dense strides in preserved memory layout
  out_strides.resize(tensor.dim());
  int64_t curr_stride = 1;
  for (size_t i = 0; i <= sz; ++i) {
    int64_t idx = perm[i];
    out_strides[idx] = curr_stride;
    if (ori_sizes[idx] != 1) {
      curr_stride *= ori_sizes[idx];
    }
  }

  return true;
}

} // namespace at
