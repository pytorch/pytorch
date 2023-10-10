#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/ExpandBase.h>

#include <c10/util/irange.h>

namespace at {
namespace internal {
TensorBase expand_slow_path(const TensorBase &self, IntArrayRef size) {
  return OptionalTensorRef(self)->expand(size);
}
} // namespace internal

namespace {
// NOTE: are_expandable did a similar check, please keep them sync if change is needed
template <typename Container, typename ArrayType>
Container infer_size_impl(ArrayType a, ArrayType b) {
  size_t dimsA = a.size();
  size_t dimsB = b.size();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;
  Container expandedSizes(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    TORCH_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);

      // 1s map to the other size (even 0).
      expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  return expandedSizes;
}
}

std::vector<int64_t> infer_size(IntArrayRef a, IntArrayRef b) {
  return infer_size_impl<std::vector<int64_t>>(a, b);
}

std::vector<SymInt> infer_size_symint(SymIntArrayRef a, SymIntArrayRef b) {
  return infer_size_impl<std::vector<SymInt>>(a, b);
}

DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b) {
  return infer_size_impl<DimVector, IntArrayRef>(a, b);
}

SymDimVector infer_size_symdimvector(SymIntArrayRef a, SymIntArrayRef b) {
  return infer_size_impl<SymDimVector, SymIntArrayRef>(a, b);
}

template<typename Container>
C10_ALWAYS_INLINE InferExpandGeometryResult<Container> inferExpandGeometryImpl(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
  int64_t ndim = sizes.size();
  int64_t tensor_dim = tensor_sizes.size();

  if (tensor_dim == 0) {
    return InferExpandGeometryResult<Container>(sizes, ndim);
  }

  InferExpandGeometryResult<Container> result(ndim);
  auto& expandedSizes = result.sizes;
  auto& expandedStrides = result.strides;

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
  return result;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
  auto result = inferExpandGeometryImpl<std::vector<int64_t>>(
      tensor_sizes, tensor_strides, sizes);
  return std::make_tuple(std::move(result.sizes), std::move(result.strides));
}

InferExpandGeometryResult<DimVector> inferExpandGeometry_dimvector(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
  return inferExpandGeometryImpl<DimVector>(
      tensor_sizes, tensor_strides, sizes);
}


// This function returns a dense and non-overlapping strides, which keeps the same layout permutation
// as the input `tensor_strides`, computed based on the input `tensor_sizes`.
// Note:
// 1. This function expects the inputs `tensor_strides` and `tensor_sizes` are non-dense or overlapping,
//    If the inputs are densed and non-overlapping, the output strides will be the same as `tensor_strides`.
//    However, this function won't check whether inputs are dense or overlapping, so the whole function will
//    still be executed even the inputs are already dense and non-overlapping, this will cause slowness.
//
//    Please verify whether the inputs are non-dense or overlapping before calling this function if possible,
//    if the inputs come from a tensor, you can check this through `is_non_overlapping_and_dense()`
//
// 2. The strides propagation rule that is used in this function is exactily the same as what is being used in
//    TensorIterator. Please refer to https://github.com/pytorch/pytorch/pull/42922 for more details

std::vector<int64_t> infer_dense_strides(IntArrayRef tensor_sizes, IntArrayRef tensor_strides) {

  TORCH_CHECK(tensor_sizes.size() == tensor_strides.size(),
    "Input sizes and strides should have same size but got ", tensor_sizes.size(), " and ", tensor_strides.size());

  size_t ndim = tensor_sizes.size();
  if (ndim == 0) {
    return {};
  }
  if (ndim == 1) {
    return {1};
  }

  std::vector<int64_t> perm(ndim);
  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm.rbegin(), perm.rend(), 0);

  // The following sorting algorithm has exactly the same behavior as TensorIterator
  // This is to make sure we have the same stride propagation everywhere.

  // return -1 if dim0 should come before dim1
  // return  1 if dim0 should come after dim1
  // return  0 if comparison is ambiguous
  auto should_swap = [&](size_t dim0, size_t dim1) {
    int64_t stride0 = tensor_strides[dim0];
    int64_t stride1 = tensor_strides[dim1];

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
    if (tensor_sizes[dim0] > tensor_sizes[dim1]) {
      return 1;
    }
    return 0;
  };

  // Insertion sort (stable) indices in `perm` based on input tensor's stride and shape,
  // all dimensions with 0 stride won't move. This is the same behavior as TensorIterator.
  // eg. Given tensor with size/stride (6, 5, 4, 3, 2)/(6, 0, 120, 0, 1), the initial `perm`
  //     is (4, 3, 2, 1, 0) and the sorted `perm` will be (4, 3, 0, 1, 2)
  for (const auto i : c10::irange(1, ndim)) {
    auto dim1 = i;
    for (const auto j : c10::irange(1, i + 1)) {
      auto dim0 = i - j;
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
  for (const auto i : c10::irange(ndim)) {
    int64_t idx = perm[i];
    out_strides[idx] = curr_stride;
    // Note: for size 0, we simply treated it as 1, it really doesn't matter here
    // since the total number of element is 0.
    if (tensor_sizes[idx] > 1) {
      curr_stride *= tensor_sizes[idx];
    }
  }
  return out_strides;
}

} // namespace at
