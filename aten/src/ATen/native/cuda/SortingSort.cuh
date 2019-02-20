namespace at {
namespace native {

// Collection of kernel sort routines
template <typename scalar_t, bool handleNaN = false>
struct LTComp {
  __device__ inline bool operator()(const scalar_t& a, const scalar_t& b)
      const {
    return (handleNaN && THCNumerics<scalar_t>::isnan(b) &&
            !THCNumerics<scalar_t>::isnan(a)) ||
        THCNumerics<scalar_t>::lt(a, b);
  }
};

template <typename scalar_t, bool handleNaN = false>
struct GTComp {
  __device__ inline bool operator()(const scalar_t& a, const scalar_t& b)
      const {
    return (handleNaN && THCNumerics<scalar_t>::isnan(a) &&
            !THCNumerics<scalar_t>::isnan(b)) ||
        THCNumerics<scalar_t>::gt(a, b);
  }
};

template <typename scalar_t>
__device__ inline void swapVars(scalar_t& t1, scalar_t& t2) {
  scalar_t tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(
    K& kA,
    V& vA,
    bool& validA,
    K& kB,
    V& vB,
    bool& validB,
    bool dir,
    const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
};

template <typename Comparator, typename K>
__device__ inline void bitonicSwapValues(
    K& kA,
    bool& validA,
    K& kB,
    bool& validB,
    bool descending,
    const Comparator& comp) {
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == descending) {
    swapVars(kA, kB);
    swapVars(validA, validB);
  }
}

template <
    typename Comparator,
    typename K,
    typename V,
    typename index_t,
    int Power2SortSize>
__device__ inline void bitonicSort(
    K values[Power2SortSize],
    V indices[Power2SortSize],
    bool valid[Power2SortSize],
    const Comparator& comp) {
#pragma unroll
  for (uint32_t size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#pragma unroll
    for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();

      uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(
          values[pos],
          indices[pos],
          valid[pos],
          values[pos + stride],
          indices[pos + stride],
          valid[pos + stride],
          flag,
          comp);
    }
  }

#pragma unroll
  for (uint32_t stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    __syncthreads();

    uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(
        values[pos],
        indices[pos],
        valid[pos],
        values[pos + stride],
        indices[pos + stride],
        valid[pos + stride],
        false,
        comp);
  }

  __syncthreads();
}

template <typename Comparator, typename K, typename index_t, int Power2SortSize>
__device__ inline void bitonicSortValues(
    K values[Power2SortSize],
    bool valid[Power2SortSize],
    const Comparator& comp) {
#pragma unroll
  for (uint32_t size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#pragma unroll
    for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
      __syncthreads();

      uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwapValues<Comparator, K>(
          values[pos],
          valid[pos],
          values[pos + stride],
          valid[pos + stride],
          flag,
          comp);
    }
  }

#pragma unroll
  for (uint32_t stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    __syncthreads();

    uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwapValues<Comparator, K>(
        values[pos],
        valid[pos],
        values[pos + stride],
        valid[pos + stride],
        false,
        comp);
  }

  __syncthreads();
}

// Sorts (value, index) pairs (in different tensors) in-place; i.e.,
// modifies the input `values` and `indices`
template <
    typename scalar_t,
    int ValueDims,
    int IndexDims,
    typename Comparator,
    typename index_t,
    int Power2SortSize>
C10_LAUNCH_BOUNDS(1024)
__global__ void bitonicSortKVInPlace(
    cuda::detail::TensorInfo<scalar_t, index_t> values,
    index_t valueSlices,
    index_t valueSliceSize,
    index_t valueSliceStride,
    cuda::detail::TensorInfo<int64_t, index_t> indices,
    index_t indicesliceStride,
    Comparator comp) {
  // Find the slice of the tensor that we are sorting
  const index_t linearIndex = getLinearBlockId<index_t>();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= valueSlices) {
    return;
  }

  __shared__ scalar_t sharedValues[Power2SortSize];
  __shared__ int64_t sharedIndices[Power2SortSize];
  __shared__ bool sharedValid[Power2SortSize];

  const index_t valueStartOffset =
      cuda::detail::IndexToOffset<scalar_t, index_t, ValueDims>::get(
          linearIndex, values);
  const index_t indicestartOffset =
      cuda::detail::IndexToOffset<int64_t, index_t, ValueDims>::get(
          linearIndex, indices);

  // If the sort size is 1, the data is already sorted
  if (Power2SortSize == 1) {
    return;
  } else {
    // Otherwise, each thread is responsible for loading and storing 2
    // elements. The sort size is guaranteed to be >= 2
    const int elem1 = threadIdx.x;
    const int elem2 = threadIdx.x + (Power2SortSize / 2);

    bool valid1 = (elem1 < valueSliceSize);
    scalar_t k1 = valid1
        ? values.data[valueStartOffset + elem1 * valueSliceStride]
        : static_cast<scalar_t>(0);
    int64_t v1 = valid1
        ? indices.data[indicestartOffset + elem1 * indicesliceStride]
        : 0;

    sharedValues[elem1] = k1;
    sharedIndices[elem1] = v1;
    sharedValid[elem1] = valid1;

    bool valid2 = (elem2 < valueSliceSize);
    scalar_t k2 = valid2
        ? values.data[valueStartOffset + elem2 * valueSliceStride]
        : static_cast<scalar_t>(0);
    int64_t v2 = valid2
        ? indices.data[indicestartOffset + elem2 * indicesliceStride]
        : 0;

    sharedValues[elem2] = k2;
    sharedIndices[elem2] = v2;
    sharedValid[elem2] = valid2;

    // Sort!
    bitonicSort<Comparator, scalar_t, int64_t, index_t, Power2SortSize>(
        sharedValues, sharedIndices, sharedValid, comp);

    // elem1 and elem2 indices might be out-of-range, if the data size we are
    // sorting is smaller than half the power2 size
    if (valid1) {
      values.data[valueStartOffset + elem1 * valueSliceStride] =
          sharedValues[elem1];
      indices.data[indicestartOffset + elem1 * indicesliceStride] =
          sharedIndices[elem1];
    }

    if (valid2) {
      values.data[valueStartOffset + elem2 * valueSliceStride] =
          sharedValues[elem2];
      indices.data[indicestartOffset + elem2 * indicesliceStride] =
          sharedIndices[elem2];
    }
  }
}

template <typename scalar_t, typename index_t, int valuedims, int sortsize>
inline void launch_bitonic_sort_per_size(
    const Tensor& indices,
    int64_t dim,
    cuda::detail::TensorInfo<scalar_t, index_t>& valueinfo,
    int64_t collapseValueDim,
    bool descending) {
  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (value, indices) sort by slice segment
  int64_t valueSliceSize =
      indices.size(dim); // we know that indices.sizes() == value.sizes()
  int64_t valueSlices = indices.numel() / valueSliceSize;

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!getGridFromTiles(valueSlices, grid)) {
    AT_ERROR("Slice to sort is too large");
  }

  int blockSize = sortsize / 2;
  if (blockSize < 1) {
    blockSize = 1;
  }
  dim3 block(blockSize);

  auto indicesInfo = cuda::detail::getTensorInfo<int64_t, index_t>(indices);
  indicesInfo.reduceDim(dim);
  int64_t collapseIndicesDim = indicesInfo.collapseDims(dim);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (descending) {
    bitonicSortKVInPlace<
        scalar_t,
        valuedims,
        -1,
        GTComp<scalar_t, true>,
        index_t,
        sortsize><<<grid, block, 0, stream>>>(
        valueinfo,
        static_cast<index_t>(valueSlices),
        static_cast<index_t>(valueSliceSize),
        static_cast<index_t>(valueinfo.strides[collapseValueDim]),
        indicesInfo,
        static_cast<index_t>(indicesInfo.strides[collapseIndicesDim]),
        GTComp<scalar_t, true>());
  } else {
    bitonicSortKVInPlace<
        scalar_t,
        valuedims,
        -1,
        LTComp<scalar_t, true>,
        index_t,
        sortsize><<<grid, block, 0, stream>>>(
        valueinfo,
        valueSlices,
        static_cast<index_t>(valueSliceSize),
        static_cast<index_t>(valueinfo.strides[collapseValueDim]),
        indicesInfo,
        static_cast<index_t>(indicesInfo.strides[collapseIndicesDim]),
        LTComp<scalar_t, true>());
  }
}

template <typename scalar_t, typename index_t, int valuedims>
inline void launch_bitonic_sort(
    const Tensor& indices,
    int64_t dim,
    cuda::detail::TensorInfo<scalar_t, index_t>& valueinfo,
    int64_t collapseValueDim,
    bool descending) {
  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(indices.size(dim));
  AT_ASSERT(
      ceilPowerOf2 <=
      2048); // sortKeyValueInplace only works for sizes <= 2048 at present

  switch (ceilPowerOf2) {
    case 2048:
      launch_bitonic_sort_per_size<scalar_t, index_t, valuedims, 2048>(
          indices, dim, valueinfo, collapseValueDim, descending);
      break;
    case 1024:
    case 512:
    case 256:
      launch_bitonic_sort_per_size<scalar_t, index_t, valuedims, 1024>(
          indices, dim, valueinfo, collapseValueDim, descending);
      break;
    case 128:
    case 64:
      launch_bitonic_sort_per_size<scalar_t, index_t, valuedims, 128>(
          indices, dim, valueinfo, collapseValueDim, descending);
      break;
    case 32:
    case 16:
    case 8:
    case 4:
    case 2:
      launch_bitonic_sort_per_size<scalar_t, index_t, valuedims, 32>(
          indices, dim, valueinfo, collapseValueDim, descending);
      break;
    case 1:
      /* Nothing to do, data already sorted */
      break;
    default:
      assert(false);
  }
}

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
template <typename scalar_t>
void sortKeyValueInplace(
    const Tensor& values,
    const Tensor& indices,
    int64_t dim_,
    bool descending) {
  int64_t dim = maybe_wrap_dim(dim_, values.dim());
  AT_CHECK(
      values.sizes() == indices.sizes(),
      "values tensor must have same size as indices tensor");
  AT_CHECK(indices.dtype() == kLong, "indices tensor must be of dtype Long");
  AT_CHECK(
      values.dim() < MAX_TENSORINFO_DIMS && indices.dim(),
      "cannot sort tensors with more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  int64_t inElements = values.numel();

  if (inElements == 0) {
    return;
  }

  // The constructed key/indices tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (cuda::detail::canUse32BitIndexMath(values) &&
      cuda::detail::canUse32BitIndexMath(indices)) {
    auto valuesinfo = cuda::detail::getTensorInfo<scalar_t, uint32_t>(values);
    valuesinfo.reduceDim(dim);
    int64_t collapseValuesDim = valuesinfo.collapseDims(dim);

    if (valuesinfo.isContiguous()) {
      launch_bitonic_sort<scalar_t, uint32_t, -2>(
          indices, dim, valuesinfo, collapseValuesDim, descending);
    } else {
      switch (valuesinfo.dims) {
        case 2:
          launch_bitonic_sort<scalar_t, uint32_t, 2>(
              indices, dim, valuesinfo, collapseValuesDim, descending);
          break;
        default:
          launch_bitonic_sort<scalar_t, uint32_t, -1>(
              indices, dim, valuesinfo, collapseValuesDim, descending);
          break;
      }
    }
  } else {
    auto valuesinfo = cuda::detail::getTensorInfo<scalar_t, uint64_t>(values);
    valuesinfo.reduceDim(dim);
    int64_t collapseValuesDim = valuesinfo.collapseDims(dim);

    // int64_t case is rare, just instantiate the generic version
    launch_bitonic_sort<scalar_t, uint64_t, -1>(
        indices, dim, valuesinfo, collapseValuesDim, descending);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace native
} // namespace at
