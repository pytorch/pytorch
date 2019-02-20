#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <THC/THCDeviceUtils.cuh> // only for THCRoundUp?
#include <THC/THCNumerics.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorMathReduce.cuh> // AddOp
#include <THC/THCThrustAllocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingSort.cuh>

namespace at {
namespace native {

namespace {


#pragma nv_exec_check_disable
template <typename scalar_t>
void sortViaThrust(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool descending) {
  // We perform a vectorized segmented sort in Thrust.
  // Say we are sorting a (2, 3) tensor. We have in flattened form:
  // values 0.4 1.2 5.3 6.2 1.3 2.3
  // indices  0   1   2   3   4   5
  // where indices is a global index (across all slices)

  // First we sort by values, globally:
  // values 6.2 5.3 2.3 1.2 1.3 0.4
  // indices  3   2   5   1   4   0

  // Then we stable sort by segment, which is index / 3:
  // values 5.3 1.2 0.4 6.2 2.3 1.3
  // indices  2   1   0   3   5   4

  // Then we translate the global index to a per-slice Lua index
  // (index % 3) + 1:
  // values 5.3 1.2 0.4 6.2 2.3 1.3
  // indices  3   2   1   1   3   2

  // This method can only work if the slice we are sorting (`dim`) is
  // innermost, and both values and indices are contiguous. We do this
  // by re-arranging the input into this form as needed, which will
  // unfortunately allocate memory if the request is not in this form.
  // Vectorized sort is slower than iterated sort if the number of
  // slices is small (since we're sorting twice, instead of invoking a
  // smaller sort `numSlices` times), but the Thrust sort
  // implementation here is a catch-all, so we're not looking for
  // efficiency, but instead correctness.
  // a copy of self that has contiguous sorted dim

  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t nDims = self.dim();

  // we use that transpose with the same coordinate twice works
  // (leaving the matrix unchanged)
  Tensor tr_contig_values =
      self.transpose(dim, -1)
          .clone(); // a copy that is of the required contiguitys
  Tensor trContigIndices = at::empty(
      tr_contig_values.sizes(), tr_contig_values.options().dtype(kLong));

  int64_t totalElements = self.numel();
  int64_t sliceSize = self.size(dim);
  int64_t sliceStride = self.stride(dim);

  auto stream = at::cuda::getCurrentCUDAStream();

  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  thrust::device_ptr<scalar_t> value_iter(tr_contig_values.data<scalar_t>());

  // Since we are composing a global index across all segments rather
  // than a per-segment index, we treat the memory as int so we don't
  // have problems sorting slices < 2^24 but where the entire tensor
  // has more than 2^24 elements
  thrust::device_ptr<int64_t> index_iter(trContigIndices.data<int64_t>());

  // Fill the indices with a global index across all slices
  thrust::counting_iterator<int64_t> count_iter(0);

  thrust::copy(
      thrust::cuda::par(allocator).on(stream),
      count_iter,
      count_iter + totalElements,
      index_iter);

  // First, we sort globally (across all slices) according to key
  // (the values we're sorting)
  if (descending) {
    thrust::stable_sort_by_key(
        thrust::cuda::par(allocator).on(stream),
        value_iter,
        value_iter + totalElements,
        index_iter,
        ThrustGTOp<scalar_t, true>());
  } else {
    thrust::stable_sort_by_key(
        thrust::cuda::par(allocator).on(stream),
        value_iter,
        value_iter + totalElements,
        index_iter,
        ThrustLTOp<scalar_t, true>());
  }

  // Then, re-sort according to slice that each index is
  // in. This completes the segment sort in Thrust, since we're
  // stably sorting here, preserving the relative order of values
  // per each slice
  thrust::stable_sort_by_key(
      thrust::cuda::par(allocator).on(stream),
      index_iter,
      index_iter + totalElements,
      value_iter,
      SliceComp(sliceSize));

  // Translate the global integer 0-based index to a per-slice real
  // Lua index
  thrust::for_each(
      thrust::cuda::par(allocator).on(stream),
      index_iter,
      index_iter + totalElements,
      GlobalIndexToPerSliceIndex(sliceSize));

  // Reverse the transposition as needed
  if (dim != nDims - 1) {
    tr_contig_values.transpose_(dim, nDims - 1);
    trContigIndices.transpose_(dim, nDims - 1);
  }

  // Then copy back to the expected output
  values.copy_(tr_contig_values);
  indices.copy_(trContigIndices);
}

Tensor& fill_slice_with_index(Tensor& t, int64_t dim_) {
  int64_t dim = maybe_wrap_dim(dim_, t.dim());
  AT_CHECK(
      t.dim() < MAX_TENSORINFO_DIMS,
      "cannot operate on tensors with more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  int64_t inElements = t.numel();
  if (inElements == 0) {
    return t;
  }
  int64_t sliceSize = t.size(dim);
  int64_t numSlices = inElements / sliceSize;

  dim3 grid;
  if (!getGridFromTiles(numSlices, grid)) {
    AT_ERROR("Slice to fill with indices is too large");
  }

  int64_t maxThreads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t numThreads = sliceSize;
  if (numThreads > maxThreads) {
    numThreads = maxThreads;
  }

  dim3 block(numThreads);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (cuda::detail::canUse32BitIndexMath(t)) {
    auto info = cuda::detail::getTensorInfo<int64_t, uint32_t>(t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    if (info.isContiguous()) {
      fillSliceWithIndex_kernel<uint32_t, -2><<<grid, block, 0, stream>>>(
          info, numSlices, sliceSize, info.strides[collapseDim]);
    } else {
      if (info.dims == 1) {
        fillSliceWithIndex_kernel<uint32_t, 1><<<grid, block, 0, stream>>>(
            info, numSlices, sliceSize, info.strides[collapseDim]);
      } else if (info.dims == 2) {
        fillSliceWithIndex_kernel<uint32_t, 2><<<grid, block, 0, stream>>>(
            info, numSlices, sliceSize, info.strides[collapseDim]);
      } else {
        fillSliceWithIndex_kernel<uint32_t, -1><<<grid, block, 0, stream>>>(
            info, numSlices, sliceSize, info.strides[collapseDim]);
      }
    }
  } else {
    auto info = cuda::detail::getTensorInfo<int64_t, uint64_t>(t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    // catch-all implementation
    fillSliceWithIndex_kernel<uint64_t, -1><<<grid, block, 0, stream>>>(
        info, numSlices, sliceSize, info.strides[collapseDim]);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return t;
}

} // namespace

std::tuple<Tensor&, Tensor&> sort_out_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool descending) {
  if (values.defined()) {
    AT_CHECK(
        self.type() == values.type(),
        "output values must be of same type as self");
    values.resize_as_(self);
  } else {
    values = at::empty_like(self);
  }
  if (indices.defined()) {
    AT_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    AT_CHECK(
        indices.device() == self.device(),
        "output indices must be on same device as self");
    indices.resize_(self.sizes());
  } else {
    indices = at::empty(self.sizes(), self.options().dtype(kLong));
  }
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.fill_(0);
    return std::forward_as_tuple(values, indices);
  }
  AT_CHECK(
      self.dim() < MAX_TENSORINFO_DIMS,
      "cannot sort tensors with more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // How large are the slices that we are sorting?
  int64_t sliceSize = self.size(dim);

  // Workaround:
  // CUDA 8 uses more shared memory than 7.5 for bitonicSortKVInPlace,
  // and so for the double word types,
  // we get "too many resources requested for launch" in the 2048 case
#if CUDA_VERSION >= 8000
  int maxSliceSize =
      (self.dtype() == kDouble || self.dtype() == kLong) ? 1024 : 2048;
#else
  int maxSliceSize = 2048;
#endif

  if (sliceSize <= maxSliceSize) {
    // Sort using our in-place k/v kernel that supports arbitrary
    // layout

    // Fill `indices` (the values) with the
    // slice-relative index.
    fill_slice_with_index(indices, dim);

    // We sort k/v pairs in-place; copy unsorted self to output
    values.copy_(self);

    AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "sort", [&] {
      sortKeyValueInplace<scalar_t>(values, indices, dim, descending);
    });

  } else {
    // Otherwise, fall back upon Thrust, which handles all other cases
    // (potentially slowly, with extra copies/memory allocations)
    AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "sort", [&] {
      sortViaThrust<scalar_t>(values, indices, self, dim, descending);
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return std::forward_as_tuple(values, indices);
}

} // namespace native
} // namespace at
