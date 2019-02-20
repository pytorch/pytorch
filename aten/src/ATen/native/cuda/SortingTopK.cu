#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <THC/THCDeviceUtils.cuh> // only for THCRoundUp?
#include <THC/THCNumerics.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorMathReduce.cuh> // AddOp

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <THC/THCThrustAllocator.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingSort.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>

namespace at {
namespace native {

namespace {

template <typename scalar_t, typename index_t, int Dim, bool Order>
__global__ void gatherTopK(
    cuda::detail::TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t outputSliceSize, // aka `k`

    index_t numInputSlices,
    index_t inputWithinSliceStride,

    cuda::detail::TensorInfo<scalar_t, index_t> topK,
    index_t numTopKSlices,
    index_t topKWithinSliceStride,

    cuda::detail::TensorInfo<int64_t, index_t> indices,
    index_t indicesWithinSliceStride) {
  // Indices are limited to integer fp precision, so counts can fit in
  // int32, regardless of index_t
  __shared__ int smem[WARP_SIZE]; // one per each warp, up to warp limit

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  index_t sliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, input);
  index_t topKSliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, topK);
  index_t indicesSliceStartIndex =
      cuda::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

  scalar_t* inputSliceStart = &input.data[sliceStartIndex];
  scalar_t* topKSliceStart = &topK.data[topKSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  scalar_t topKValue = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t,
      Order>(
      inputSliceStart,
      outputSliceSize,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &topKValue);

  // Every value that is strictly less/greater than `pattern`
  // (depending on sort dir) in sorted int format is in the top-K.
  // The top-K value itself might not be unique.
  //
  // Since there are a variable number of elements that we see that
  // are within the top-k, we don't know at what index to write out
  // the resulting values.
  // In order to get this, we perform an exclusive prefix sum of
  // `hasTopK`. This will return the resulting index into which we
  // need to write the result, if a thread has a result.

  // All threads need to participate in the loop and the prefix sum,
  // but not necessarily in the load; hence loop bounds being rounded
  // up to a multiple of the block dim.
  index_t numIterations =
      THCRoundUp(inputSliceSize, static_cast<index_t>(blockDim.x));
  index_t writeIndexStart = 0;

  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    scalar_t v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride])
                         : static_cast<scalar_t>(0);
    bool hasTopK;
    if (Order) {
      hasTopK = inRange && (THCNumerics<scalar_t>::gt(v, topKValue));
    } else {
      hasTopK = inRange && (THCNumerics<scalar_t>::lt(v, topKValue));
    }

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK) {
      int writeIndex = writeIndexStart + index;
      assert(writeIndex < outputSliceSize);

      index_t topKOffset = writeIndex * topKWithinSliceStride;
      index_t indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i + TH_INDEX_BASE; // to Lua index
    }

    writeIndexStart += carry;
  }

  // We need to fill in the rest with actual == top-K values.
  // The number that we need is outputSliceSize -
  // writeIndexStart. There might be more than that number available,
  // in which case we have to choose the first seen set. We do this
  // via a prefix sum to calculate indices for writing results.
  assert(outputSliceSize >= writeIndexStart);
  index_t topKRemaining = (outputSliceSize - writeIndexStart);

  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    scalar_t v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride])
                         : static_cast<scalar_t>(0);
    bool hasTopK = inRange && (THCNumerics<scalar_t>::eq(v, topKValue));

    int index;
    int carry;
    exclusiveBinaryPrefixScan<int, true>(
        smem, hasTopK, &index, &carry, AddOp<int>());

    if (hasTopK && index < topKRemaining) {
      int writeIndex = writeIndexStart + index;
      assert(writeIndex < outputSliceSize);

      index_t topKOffset = writeIndex * topKWithinSliceStride;
      index_t indexOffset = writeIndex * indicesWithinSliceStride;

      topKSliceStart[topKOffset] = v;
      indicesSliceStart[indexOffset] = i + TH_INDEX_BASE; // to Lua index
    }

    if (carry >= topKRemaining) {
      break;
    }

    topKRemaining -= carry;
    writeIndexStart += carry;
  }
}

struct TopKLauncher {
  int64_t k;
  bool largest;

  TopKLauncher(int64_t k, bool largest) : k(k), largest(largest) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      cuda::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      cuda::detail::TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      cuda::detail::TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t self_num_slices,
      int64_t slice_size) {
    int64_t values_num_slices = 1;
    for (int i = 0; i < values_info.dims; ++i) {
      values_num_slices *= values_info.sizes[i];
    }

    dim3 grid;
    if (!getGridFromTiles(self_num_slices, grid)) {
      AT_ERROR("Number of slices is too large");
    }

    dim3 block(
        std::min(THCRoundUp(slice_size, (int64_t)WARP_SIZE), (int64_t)1024));

    // static_cast is required to ensure that the correct type (index_t)
    // is provided to the kernel for the arguments.
    if (largest) {
      gatherTopK<scalar_t, index_t, all_dims, true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              self_info,
              static_cast<index_t>(slice_size),
              static_cast<index_t>(k),
              static_cast<index_t>(self_num_slices),
              /* The actual dimension that the k-selection is running in */
              /* may have changed from collapseDims() */
              static_cast<index_t>(self_info.strides[collapse_self_dim]),
              values_info,
              static_cast<index_t>(values_num_slices),
              static_cast<index_t>(values_info.strides[collapse_values_dim]),
              indices_info,
              static_cast<index_t>(indices_info.strides[collapse_indices_dim]));
    } else {
      gatherTopK<scalar_t, index_t, all_dims, false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              self_info,
              static_cast<index_t>(slice_size),
              static_cast<index_t>(k),
              static_cast<index_t>(self_num_slices),
              /* The actual dimension that the k-selection is running in */
              /* may have changed from collapseDims() */
              static_cast<index_t>(self_info.strides[collapse_self_dim]),
              values_info,
              static_cast<index_t>(values_num_slices),
              static_cast<index_t>(values_info.strides[collapse_values_dim]),
              indices_info,
              static_cast<index_t>(indices_info.strides[collapse_indices_dim]));
    }
  }
};

template <typename scalar_t>
void topk_cuda_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self_,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  int64_t dim = maybe_wrap_dim(dim_, self_.dim(), /*wrap_scalar=*/true);
  AT_CHECK(
      k >= 0 && k <= (self_.dim() > 0 ? self_.size(dim) : 1),
      "selected number k out of range");
  // Build the output size, which is the dim being selected set to
  // size k
  auto result_sizes = self_.sizes().vec();
  if (result_sizes.size() > 0) {
    result_sizes[dim] = k;
  } else if (k == 0) {
    result_sizes.emplace_back(0);
  }
  if (values.defined()) {
    AT_CHECK(
        self_.type() == values.type(),
        "output values must be of same type as self");
    AT_CHECK(
        values.device() == self_.device(),
        "output values must be on same device as self");
    values.resize_(result_sizes);
  } else {
    values = at::empty(result_sizes, self_.options());
  }
  if (indices.defined()) {
    AT_CHECK(
        indices.dtype() == kLong, "output indices must be of scalar type Long");
    AT_CHECK(
        indices.device() == self_.device(),
        "output indices must be on same device as self");
    indices.resize_(result_sizes);
  } else {
    indices = at::empty(result_sizes, self_.options().dtype(kLong));
  }
  if (k == 0) { // we're done already
    return;
  }
  if (self_.dim() == 0 && self_.numel() == 1) {
    values.copy_(self_);
    indices.zero_();
    return;
  }

  AT_CHECK(
      self_.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");
  Tensor self = self_.contiguous();

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(values) &&
      cuda::detail::canUse32BitIndexMath(indices)) {
    run_launcher<scalar_t, uint32_t>(
        values, indices, self, dim, TopKLauncher(k, largest));
  } else {
    run_launcher<scalar_t, uint64_t>(
        values, indices, self, dim, TopKLauncher(k, largest));
  }
  // Sort the results if the user wants them sorted, since our
  // selection routine does not ensure sorting
  if (sorted) {
    // FIXME: the k/v inplace sort along slice only works for size <=
    // 2048 at the moment
    // FIXME: 1024 seems to be the limit with newer cuda and 64 bit types
    if (k <= 2048) {
      // This avoids any memory allocations and performs all sorting
      // work inplace along the slice
      sortKeyValueInplace<scalar_t>(values, indices, dim, largest);
    } else {
      // Depend upon the backup sort that returns indices, which we
      // can use in conjunction with gather to produce the original
      // indices.
      // This is not the most efficient implementation, especially since
      // there are memory allocations performed here. If the user desires
      // greater performance, they should torch.gather() the results
      // themselves using the reported indices, providing previously
      // allocated tensors to receive the results.
      auto sorted = values.sort(dim, largest);
      auto mapped_indices = indices.gather(dim, std::get<1>(sorted));
      indices.copy_(mapped_indices);
      values.copy_(std::get<0>(sorted));
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
}


} // namespace

std::tuple<Tensor&, Tensor&> topk_out_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "topk", [&] {
    topk_cuda_template<scalar_t>(
        values, indices, self, k, dim, largest, sorted);
  });
  return std::forward_as_tuple(values, indices);
}

} // namespace native
} // namespace at
