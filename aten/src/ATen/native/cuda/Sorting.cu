#include <ATen/ATen.h>
#include <ATen/native/SortingUtils.h>
#include <assert.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <THC/THCDeviceUtils.cuh> // only for THCRoundUp?
#include <THC/THCNumerics.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorMathReduce.cuh> // AddOp
#include <THC/THCThrustAllocator.cuh>

namespace at {
namespace native {

namespace {

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
constexpr int MAX_BLOCK_SIZE = 256;

#else
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
#endif

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
constexpr int64_t MAX_GRID_SIZE = 65535LL;

bool getGridFromTiles(int64_t gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  int64_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = cuda::ATenCeilDiv(gridTiles, MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = cuda::ATenCeilDiv(gridTiles, MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = dim3(gridX, gridY, gridZ);
  return true;
}

template <typename scalar_t, bool handleNaN = false>
struct ThrustGTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && THCNumerics<scalar_t>::isnan(lhs) &&
            !THCNumerics<scalar_t>::isnan(rhs)) ||
        THCNumerics<scalar_t>::gt(lhs, rhs);
  }
};

template <typename scalar_t, bool handleNaN = false>
struct ThrustLTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && THCNumerics<scalar_t>::isnan(rhs) &&
            !THCNumerics<scalar_t>::isnan(lhs)) ||
        THCNumerics<scalar_t>::lt(lhs, rhs);
  }
};

template <typename index_t>
__device__ __forceinline__ index_t getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
      blockIdx.x;
}

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename index_t, int Dim>
__global__ void fillSliceWithIndex_kernel(
    cuda::detail::TensorInfo<int64_t, index_t> out,
    index_t totalSlices,
    index_t sliceSize,
    index_t sliceStride) {
  index_t slice = getLinearBlockId<index_t>();

  if (slice >= totalSlices) {
    return;
  }

  const uint64_t offset =
      cuda::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, out);
  int64_t* base = &out.data[offset];

  for (int64_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = i;
  }
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  SliceComp(int64_t size) : sliceSize(size) {}

  __device__ bool operator()(const int64_t& a, const int64_t& b) const {
    // Since the slices are guaranteed to be innermost,
    // the segment is just via int64_t division
    int64_t segA = a / sliceSize;
    int64_t segB = b / sliceSize;
    return segA < segB;
  }

  const int64_t sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(int64_t size) : sliceSize(size) {}

  __device__ inline void operator()(int64_t& v) const {
    v = v % sliceSize;
  }

  const int64_t sliceSize;
};

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

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
uint64_t nextHighestPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;

  return n;
}

// ****************************** Sort ***************

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

// ********************************* Topk ********************************

template <typename scalar_t>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;

  // Converts a float to an integer representation with the same
  // sorting; i.e., for floats f1, f2:
  // if f1 < f2 then convert(f1) < convert(f2)
  // We use this to enable radix selection of floating-point values.
  // This also gives a relative order for NaNs, but that's ok, as they
  // will all be adjacent
  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<uint8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(uint8_t v) {
    return v;
  }

  static inline __device__ uint8_t deconvert(RadixType v) {
    return v;
  }
};

template <>
struct TopKTypeConfig<int8_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int8_t v) {
    return 128u + v;
  }

  static inline __device__ int8_t deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct TopKTypeConfig<int16_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int16_t v) {
    assert(sizeof(short) == 2);
    return 32768u + v;
  }

  static inline __device__ int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<int32_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int32_t v) {
    assert(sizeof(int) == 4);
    return 2147483648u + v;
  }

  static inline __device__ int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(int64_t v) {
    assert(sizeof(int64_t) == 8);
    return 9223372036854775808ull + v;
  }

  static inline __device__ int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<at::Half> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(at::Half v) {
#if CUDA_VERSION >= 8000 || defined __HIP_PLATFORM_HCC__
    RadixType x = __half_as_ushort(v);
    RadixType mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
#else
    assert(false);
    return 0u;
#endif
  }

  static inline __device__ at::Half deconvert(RadixType v) {
#if CUDA_VERSION >= 8000 || defined __HIP_PLATFORM_HCC__
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    return __ushort_as_half(v ^ mask);
#else
    assert(false);
    return static_cast<at::Half>(0);
#endif
  }
};

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits>
__device__ void countRadixUsingMask(
    CountType counts[RadixSize],
    CountType* smem,
    bitwise_t desired,
    bitwise_t desiredMask,
    int radixDigitPos,
    index_t sliceSize,
    index_t withinSliceStride,
    scalar_t* data) {
  // Clear out per-thread counts from a previous round
#pragma unroll
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  // Scan over all the data. Upon a read, the warp will accumulate
  // counts per each digit in the radix using warp voting.
  for (index_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    bitwise_t val =
        TopKTypeConfig<scalar_t>::convert(doLdg(&data[i * withinSliceStride]));

    bool hasVal = ((val & desiredMask) == desired);
    bitwise_t digitInRadix =
        Bitfield<bitwise_t>::getBitfield(val, radixDigitPos, RadixBits);

#pragma unroll
    for (uint32_t j = 0; j < RadixSize; ++j) {
      bool vote = hasVal && (digitInRadix == j);
#if defined(__HIP_PLATFORM_HCC__)
      counts[j] += __popcll(WARP_BALLOT(vote));
#else
      counts[j] += __popc(WARP_BALLOT(vote, ACTIVE_MASK()));
#endif
    }
  }

  // Now, for each warp, sum values
  if (getLaneId() == 0) {
#pragma unroll
    for (uint32_t i = 0; i < RadixSize; ++i) {
      atomicAdd(&smem[i], counts[i]);
    }
  }

  __syncthreads();

  // For each thread, read in the total counts
#pragma unroll
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  __syncthreads();
}

// Over what radix we are selecting values
constexpr int RADIX_BITS = 2; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

// This finds the unique value `v` that matches the pattern
// ((v & desired) == desiredMask) in our sorted int format
template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPattern(
    scalar_t* smem,
    scalar_t* data,
    index_t sliceSize,
    index_t withinSliceStride,
    bitwise_t desired,
    bitwise_t desiredMask) {
  if (threadIdx.x < WARP_SIZE) {
    smem[threadIdx.x] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  // All threads participate in the loop, in order to sync on the flag
  index_t numIterations =
      THCRoundUp(sliceSize, static_cast<index_t>(blockDim.x));
  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    scalar_t v = inRange ? doLdg(&data[i * withinSliceStride])
                         : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = static_cast<scalar_t>(1);
      smem[1] = v; // can't use val as the flag, since it could be 0
    }

    __syncthreads();

    scalar_t found = smem[0];
    scalar_t val = smem[1];

    __syncthreads();

    // Check to see if a thread found the value
    if (THCNumerics<scalar_t>::ne(found, static_cast<scalar_t>(0))) {
      // all threads return this value
      return val;
    }
  }

  // should not get here
  assert(false);
  return static_cast<scalar_t>(0);
}

// Returns the top-Kth element found in the data using radix selection
template <typename scalar_t, typename bitwise_t, typename index_t, bool Order>
__device__ void radixSelect(
    scalar_t* data,
    index_t k,
    index_t sliceSize,
    index_t withinSliceStride,
    int* smem,
    scalar_t* topK) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
#pragma unroll
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data);

    auto found_unique = [&](int i, int count) -> bool {
      /* All threads have the same value in counts here, so all */
      /* threads will return from the function. */
      if (count == 1 && kToFind == 1) {
        /* There is a unique answer. */
        desired =
            Bitfield<bitwise_t>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The answer is now the unique element v such that: */
        /* (v & desiredMask) == desired */
        /* However, we do not yet know what the actual element is. We */
        /* need to perform a search through the data to find the */
        /* element that matches this pattern. */
        *topK = findPattern<scalar_t, bitwise_t, index_t>(
            (scalar_t*)smem,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            Bitfield<bitwise_t>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The top-Kth element v must now be one such that: */
        /* (v & desiredMask == desired) */
        /* but we haven't narrowed it down; we must check the next */
        /* least-significant digit */
        return true;
      }
      kToFind -= count;
      return false; // continue the loop
    };

    // All threads participate in the comparisons below to know the
    // final result
    if (Order) {
      // Process in descending order
#pragma unroll
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
      // Process in ascending order
#pragma unroll
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  } // end digitPos for

  // There is no unique result, but there is a non-unique result
  // matching `desired` exactly
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}

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

template <typename scalar_t, typename index_t, typename Launcher>
void run_launcher(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    Launcher l) {
  auto self_info = cuda::detail::getTensorInfo<scalar_t, index_t>(self);
  auto values_info = cuda::detail::getTensorInfo<scalar_t, index_t>(values);
  auto indices_info = cuda::detail::getTensorInfo<int64_t, index_t>(indices);

  int64_t slice_size = self.size(dim);
  /* We use these structures solely to find the offset to */
  /* each slice we are operating on */
  self_info.reduceDim(dim);
  values_info.reduceDim(dim);
  indices_info.reduceDim(dim);

  /* Collapse all other dims */
  int collapse_self_dim = self_info.collapseDims(dim);
  int collapse_values_dim = values_info.collapseDims(dim);
  int collapse_indices_dim = indices_info.collapseDims(dim);

  int64_t num_slices = 1;
  for (int i = 0; i < self_info.dims; ++i) {
    num_slices *= self_info.sizes[i];
  }

  /* This is used as a template parameter to calculate indices. */
  /* We only specialize it if all collapsed dim sizes are the */
  /* same; otherwise, we use -1 which is the specialization */
  /* parameter for arbitrary dimensions */
  int all_dims = self_info.dims;
  if (values_info.dims != all_dims || indices_info.dims != all_dims) {
    all_dims = -1;
  }

  if (all_dims == 1) {
    l.template launch<scalar_t, index_t, 1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 2) {
    l.template launch<scalar_t, index_t, 2>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 3) {
    l.template launch<scalar_t, index_t, 3>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else {
    l.template launch<scalar_t, index_t, -1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  }
}

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

// ****************************** Mode ***************

template <typename scalar_t>
struct BinaryAddOp {
  __host__ __device__ inline scalar_t operator()(
      const scalar_t a,
      const scalar_t b) {
    return THCNumerics<scalar_t>::add(a, b);
  }
};

template <>
struct BinaryAddOp<uint32_t> {
  __host__ __device__ inline uint32_t operator()(
      const uint32_t a,
      const uint32_t b) {
    return a + b;
  }
};

// Used for a segmented reduction
struct ModeUnsignedBoolPair {
  uint32_t val;
  bool flag;
};

// In the kernel below, we have a common pattern of reducing (uint32_t,
// uint32_t) pairs of data
struct ModeUnsignedPair {
  uint32_t val;
  uint32_t index;
};

template <typename scalar_t>
struct MaxReduceOp {
  __host__ __device__ inline scalar_t operator()(
      const scalar_t& a,
      const scalar_t& b) {
    return b.val > a.val ? b : a;
  }
};

template <typename scalar_t>
struct MatchReduceOp {
  __host__ __device__ inline scalar_t operator()(
      const scalar_t& a,
      const scalar_t& b) {
    return b.flag ? b : a;
  }
};

// The mode kernel has the following characteristics: It uses internal shared
// memory buffers of Power2Size, which must be greater than the number of
// elements. Additionally, there is one block for every slice to calculate the
// mode for, and in each block there is one thread for every two elements.
//
// Both sorted and positions are assumed to be contiguous Tensors with the mode
// dimension as the innermost dim, such that we can get the particular slice for
// a Tensor via its linear block dimension * the slice size.
template <typename scalar_t, uint32_t Power2Size>
__global__ void computeMode(
    scalar_t* input,
    cuda::detail::TensorInfo<scalar_t, uint32_t> values,
    cuda::detail::TensorInfo<int64_t, uint32_t> indices,
    int64_t sliceSize) {
  int tidx = threadIdx.x;
  int stidx =
      blockDim.x + threadIdx.x; // Second index this thread responsible for

  // First, we need to calculate the offset into the sorted Tensor that
  // represents the start of the slice for this block to calculate the mode for.
  // This offset is a combination of the gridIndices, and the number of elements
  // in the slice.
  uint32_t blockId = getLinearBlockId<uint32_t>();
  uint32_t linearOffset = blockId * sliceSize;

  // shmem is a dynamically sized buffer we will use throughout the kernel to
  // handle computation efficiently. The size of this shmem must be
  // sizeof(scalar_t) * Power2Size + (2 * sizeof(uint32_t) * Power2Size)
  //
  // Initially, the buffer will be organized as follows:
  //
  // [smem (slice elements) | bmem (valid indices) | <scratch space>]
  extern __shared__ char shmem[];

  // smem represents a proportion of the shared memory buffer that is used to
  // store the elements from the slice:
  scalar_t* smem = reinterpret_cast<scalar_t*>(shmem);

  // Each thread loads up to two elements from the Tensor into shared memory
  if (tidx < sliceSize) {
    smem[tidx] = input[linearOffset + tidx];
  }
  if (stidx < sliceSize) {
    smem[stidx] = input[linearOffset + stidx];
  }

  // Next, we initialize a boolean region of the buffer, offset by the loaded
  // element smem region
  bool* bmem = reinterpret_cast<bool*>(&smem[Power2Size]);

  // The first use of this region stores bmem[i] = i < sliceSize to mark the
  // valid components in the smem buffer
  bmem[tidx] = tidx < sliceSize;
  bmem[stidx] = stidx < sliceSize;
  __syncthreads(); // barrier for smem, bmem initialization

  // First, sort the input slice in ascending order. smem contains the input
  // elements, and bmem marks the valid indices
  bitonicSortValues<LTComp<scalar_t>, scalar_t, uint32_t, Power2Size>(
      smem, bmem, LTComp<scalar_t>());
  __syncthreads(); // make no assumptions that the sort syncs at end

  // The next step of our algorithm is performing a block-wide comparison of
  // neighboring elements. In particular, given an sorted input slice A, we
  // produce an output slice B, such that B[i] = 1 if A[i-i] != A[i], otherwise
  // 0.
  //
  // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
  //                 B = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  //
  // In particular, we can think of B[i] true indicating the start of a sequence
  // of equal values in the sorted list. Similarly, we will also store the
  // negation of B, which we'll call C. In particular, we can think of C[i] =
  // true iff A[i-1] == A[i] in our original sorted slice.
  //
  //                 C = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]

  // We overwrite bmem, and treat the rest of shared memory as a buffer of
  // (index, flag) pairs where the index represents values from C, and the flag
  // represents values from B.
  //
  // [smem (sorted slice) | ubpmem (index, flag pairs)]

  struct ModeUnsignedBoolPair* ubpmem =
      reinterpret_cast<struct ModeUnsignedBoolPair*>(&smem[Power2Size]);

  if (tidx == 0) {
    ubpmem[0].flag = true;
    ubpmem[0].val = 0;
  }

  // Compares elements (0, 1), (2, 3), ... and sets 1, 3, ...
  ubpmem[tidx * 2 + 1].flag = THCNumerics<scalar_t>::ne(
      smem[tidx * 2], smem[tidx * 2 + 1]); // (0, 1), (1, 2), etc.
  ubpmem[tidx * 2 + 1].val = !ubpmem[tidx * 2 + 1].flag;

  // Compares elements (1, 2), (3, 4), ... and sets 2, 4, ...
  if (((tidx + 1) * 2) < Power2Size) {
    ubpmem[(tidx + 1) * 2].flag = THCNumerics<scalar_t>::ne(
        smem[((tidx + 1) * 2) - 1], smem[(tidx + 1) * 2]);
    ubpmem[(tidx + 1) * 2].val = !ubpmem[(tidx + 1) * 2].flag;
  }
  __syncthreads(); // barrier for ubpmem initialization

  // Next, we perform a segmented prefix sum on the neighboring elements, where
  // the presence of a one indicates the start of a segment. In this case B acts
  // as the segment start flags, and C is the buffer to be summed:
  //
  // Input  (C)  = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
  // Flag   (B)  = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
  // Output (C)  = [0, 1, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 0]
  //
  // Afterwards, the (index) components of the ubpmem buffer contain the lengths
  // of the segments (minus 1), i.e. the counts of each element in the original
  // input.

  inclusivePrefixScan<
      struct ModeUnsignedBoolPair,
      struct SegmentedScanOp<
          struct ModeUnsignedBoolPair,
          BinaryAddOp<uint32_t>>,
      Power2Size>(
      ubpmem,
      SegmentedScanOp<struct ModeUnsignedBoolPair, BinaryAddOp<uint32_t>>(
          BinaryAddOp<uint32_t>()));
  // assumes scan syncs at the end

  // Next, we reinterpret the ubpmem buffer as pairs of uint32_tegers (i.e.
  // we treat the boolean flag regions as integers). We initialize these to
  // represent indices, and we'll call this buffer I
  struct ModeUnsignedPair* uupmem =
      reinterpret_cast<struct ModeUnsignedPair*>(ubpmem);

  // At this point, we need to find the maximum element in lengths buffer C.
  // This element will represent the count (-1) of the mode. Because of the
  // way we have set up the problem, the index where this mode occurs will
  // also be the location of the mode value in the sorted array, e.g.
  //
  // smem = [0, 0, 1, 1, 1, 2]
  // C    = [0, 1, 0, 1, 2, 0]
  // I    = [0, 1, 2, 3, 4, 5]
  //                     ^
  //                     maximum value, also aligned with mode = 1
  //
  // We perform a block wide max-reduction of the C buffer, but we also need the
  // indices to come along with it, so we utilize the uupmem construction.
  //
  // At the end we need to return the ModeUnsignedPair containing index = 4, val
  // = 2, which represents the max

  // In practice, we will make each thread locally reduce 2 values in its
  // registers prior to the global block-wide reduction. Note that instead of
  // tidx/stidx, we utilize tidx * 2, tidx * 2 + 1, so each thread deals with
  // adjacent elements. This is because the reduce code below relies on thread
  // elements to be adjacent.
  struct ModeUnsignedPair uup[2];
  uup[0].index = tidx * 2;
  uup[0].val = ubpmem[tidx * 2].val;
  uup[1].index = tidx * 2 + 1;
  uup[1].val = ubpmem[tidx * 2 + 1].val;
  __syncthreads();

  struct ModeUnsignedPair max = {0, 0};

  max = reduceBlockWithNThreadLocalReductions<
      struct ModeUnsignedPair,
      MaxReduceOp<struct ModeUnsignedPair>,
      2>(uupmem, uup, sliceSize, MaxReduceOp<struct ModeUnsignedPair>(), max);

  // Store the mode in shared memory for use in finding the mode in the input
  // slice
  __shared__ scalar_t mode;

  // Given the above constraints, the mode is the value at the reduced index in
  // the original sorted element buffer
  if (tidx == 0) {
    mode = smem[max.index];
  }
  __syncthreads(); // broadcast mode

  // Finally, we need to find the "an" index of the mode in the input Tensor.
  // The API does not constrain which index we pick, so it can be any of the
  // indices that contain the mode. We will do a reduction to find the index. We
  // go back to using the (index, flag) buffer arrangement. First, we mark
  // indices that are equal to the mode, i.e B[i] = true if input[i] == mode,
  // and initialize C[i] to be the index
  //
  // Again we reduce 2 elements in the thread's registers prior to the
  // block-wide reduction
  struct ModeUnsignedBoolPair ubpp[2];
  if (tidx * 2 < sliceSize) {
    ubpp[0].flag =
        THCNumerics<scalar_t>::eq(input[linearOffset + (tidx * 2)], mode);
    ubpp[0].val = tidx * 2;
  }
  if (tidx * 2 + 1 < sliceSize) {
    ubpp[1].flag =
        THCNumerics<scalar_t>::eq(input[linearOffset + (tidx * 2 + 1)], mode);
    ubpp[1].val = tidx * 2 + 1;
  }

  // Then we perform a similar reduction to the one above, except this time we
  // update the element if the element at the base position is not equal to the
  // mode and the element at the offset position is. At the end, C[0] will
  // contain an index with the mode.
  struct ModeUnsignedBoolPair match = {0, false};

  match = reduceBlockWithNThreadLocalReductions<
      struct ModeUnsignedBoolPair,
      MatchReduceOp<struct ModeUnsignedBoolPair>,
      2>(
      ubpmem,
      ubpp,
      sliceSize,
      MatchReduceOp<struct ModeUnsignedBoolPair>(),
      match);

  // Finally, we have the mode, and an index where it occurs. We use a single
  // thread to place this in the appropriate output position
  if (tidx == 0) {
    int64_t index = TH_INDEX_BASE + match.val;

    uint32_t outputOffset =
        cuda::detail::IndexToOffset<scalar_t, uint32_t, -1>::get(
            blockId, values);
    values.data[outputOffset] = mode;
    indices.data[outputOffset] = index;
  }
}

template <typename scalar_t>
void calculateMode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Tensor& sortBuffer,
    int64_t dimension,
    const std::vector<int64_t>& position) {
  AT_ASSERT(self.is_contiguous());

  // Because the input is contiguous, we want to get a reference to the
  // location of the buffer at the innermost dimension that we are going
  // to calculate the mode for --> we do this by manually doing the stride
  // calculations to get an offset
  auto data = self.data<scalar_t>();
  // FIXME: assert that the tensor is contiguous?
  for (int i = 0; i < position.size(); ++i) {
    data += position[i] * self.stride(i);
  }

  int64_t nElement = self.size(-1);
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());

  // Wrap input data, sortBuffer, in Thrust device vectors
  auto vecPtr = thrust::device_pointer_cast(data);
  thrust::device_vector<scalar_t> iter(vecPtr, vecPtr + nElement);
  auto sbPtr = thrust::device_pointer_cast(sortBuffer.data<int64_t>());
  thrust::device_vector<int64_t> seq(sbPtr, sbPtr + nElement);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Fill sortBuffer with [0, 1, 2, ... nElement - 1]
  thrust::sequence(
      thrust::cuda::par(allocator).on(stream), seq.begin(), seq.end());

  // Sort the input data. The original indices of the data are stored in seq
  thrust::sort_by_key(
      thrust::cuda::par(allocator).on(stream),
      iter.begin(),
      iter.end(),
      seq.begin());

  // Count # of unique elements via an inner product between adjacent elements.
  // Add 1 if two neighboring element are not equal.
  int unique = 1 +
      thrust::inner_product(
                   thrust::cuda::par(allocator).on(stream),
                   iter.begin(),
                   iter.end() - 1,
                   iter.begin() + 1,
                   0,
                   thrust::plus<int>(),
                   thrust::not_equal_to<scalar_t>());

  // Count frequency of each element
  thrust::device_vector<scalar_t> keys(unique);
  thrust::device_vector<int> counts(unique);
  thrust::reduce_by_key(
      thrust::cuda::par(allocator).on(stream),
      iter.begin(),
      iter.end(),
      thrust::constant_iterator<int>(1),
      keys.begin(),
      counts.begin());

  // Find index of maximum count
  auto it = thrust::max_element(
      thrust::cuda::par(allocator).on(stream), counts.begin(), counts.end());
  scalar_t mode = keys[it - counts.begin()];

  // Find first index within which it occurs
  auto positionIter = thrust::find(
      thrust::cuda::par(allocator).on(stream), iter.begin(), iter.end(), mode);

  AT_ASSERT(positionIter != iter.end());
  int64_t index = seq[positionIter - iter.begin()];

  // Place mode, index in output
  int64_t valuesOffset = values.storage_offset();
  int64_t indicesOffset = indices.storage_offset();

  for (int i = 0; i < position.size(); ++i) {
    int64_t pos = position[i];
    valuesOffset += values.stride(i) * pos;
    indicesOffset += indices.stride(i) * pos;
  }

  // FIXME: we inherited this from THC as THCStorage_set.
  // A sensible alternative with less syncing could be to create those on CPU,
  // use an accessor here and only move the finished tensors to GPU  afterwards.
  values.as_strided({1}, {1}, valuesOffset).fill_(mode);
  indices.as_strided({1}, {1}, indicesOffset).fill_(index);
}

// this probably could be a loop, not a recursive algorithm
template <typename scalar_t>
void dimApplyMode(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    Tensor& sortBuffer,
    int64_t dimension,
    std::vector<int64_t>& position,
    int64_t curDim) {
  int64_t ndim = self.dim();

  // Because we have transposed the Tensor, the data for the dimension we are
  // mode'ing along is always in the innermost dimension
  if (curDim == ndim - 1) {
    calculateMode<scalar_t>(
        values, indices, self, sortBuffer, dimension, position);
  } else {
    // Loop through the values and recurse
    for (int i = 0; i < self.size(curDim); ++i) {
      position[curDim] = i;
      dimApplyMode<scalar_t>(
          values, indices, self, sortBuffer, dimension, position, curDim + 1);
    }
  }
}

// Function that calls kernel --> note that we set the block dimensions here,
// and the amount of shared memory
template <typename scalar_t, int work_size>
void handle_mode(
    const Tensor& self_contiguous,
    cuda::detail::TensorInfo<scalar_t, uint32_t>& tiValues,
    cuda::detail::TensorInfo<int64_t, uint32_t>& tiIndices,
    int64_t slices,
    int64_t sliceSize) {
  // The number of blocks is the number of slices that we need to calculate
  // the mode for. Each block is responsible for computing a single mode
  dim3 grid;
  if (!getGridFromTiles(slices, grid)) {
    AT_ERROR("Slice to take mode of is too large");
  }
  dim3 blockSize(work_size / 2);
  int memsize =
      (sizeof(scalar_t) * work_size) + (2 * work_size * sizeof(uint32_t));
  auto stream = at::cuda::getCurrentCUDAStream();

  computeMode<scalar_t, work_size><<<grid, blockSize, memsize, stream>>>(
      self_contiguous.data<scalar_t>(), tiValues, tiIndices, sliceSize);
}

template <typename scalar_t>
void mode_cuda_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  int64_t ndim = self.dim();

  int64_t sliceSize = self.size(dim);
  int64_t slices = self.numel() / sliceSize;

  // If sliceSize is 1, copy input to values and set indices
  if (sliceSize == 1) {
    values.copy_(self);
    indices.zero_();
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze(dim);
    }
    return;
  }

  // Requirements for fused kernel implementation:
  //
  // 1. sliceSize <= 2 * max threads per block
  // 2. uses one block per slice, so number of slices must be less than the
  // maximum number of blocks for a kernel launch
  // 3. Can use 32-bit index math for indexing (mainly just for implementation
  // conciseness, could be changed)
  if (sliceSize <= MAX_BLOCK_SIZE && slices <= MAX_GRID_SIZE &&
      cuda::detail::canUse32BitIndexMath(self)) {
    // Beginning our optimized implementation. First thing we want to do is to
    // transpose the input Tensor along the sort dimension, and then make it
    // contiguous
    auto self_contiguous = self.transpose(dim, -1).clone();

    // We also need to view the values and indices Tensors as transposed in
    // order to properly determine the offset into the underlying storage in
    // which to place the mode and index for a particular set of dimension
    // values
    auto valuesTransposed = values.transpose(dim, -1);
    auto indicesTransposed = indices.transpose(dim, -1);

    // Set-up TensorInfo structs for passing to kernel
    auto tiValues =
        cuda::detail::getTensorInfo<scalar_t, uint32_t>(valuesTransposed);
    auto tiIndices =
        cuda::detail::getTensorInfo<int64_t, uint32_t>(indicesTransposed);

    // The blocksize is two elements per thread, rounded up to the nearest power
    // of 2
    int64_t ceilPowerOf2 = nextHighestPowerOf2(sliceSize);

    // Tradeoff between compilation time and the number of specializations.
    // Ideally we would have one HANDLE_MODE for each power of 2
    switch (ceilPowerOf2) {
      case 2048:
        handle_mode<scalar_t, 2048>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 1024:
      case 512:
      case 256:
        handle_mode<scalar_t, 1024>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 128:
      case 64:
        handle_mode<scalar_t, 128>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 32:
      case 16:
      case 8:
      case 4:
      case 2:
        handle_mode<scalar_t, 32>(
            self_contiguous, tiValues, tiIndices, slices, sliceSize);
        break;
      case 1:
      default:
        assert(false);
    }
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    // Beginning our naive implementation: We don't want to mutate the input
    // Tensor, but we need to be able to sort the inputs along the dimension in
    // order to calculate the mode. Additionally, its ideal if the data along
    // the dimension is contiguous. So we transpose the dimension with the
    // innermost dimension and make a new contiguous version that we can use.
    auto self_contiguous = self.transpose(dim, -1).clone();

    // We also need to view the values and indices Tensors as transposed in
    // order to properly determine the offset into the underlying storage in
    // which to place the mode and index for a particular set of dimension
    // values
    auto valuesTransposed = values.transpose(dim, -1);
    auto indicesTransposed = indices.transpose(dim, -1);

    // Position is a Storage that will store the dimension values we are
    // processing
    std::vector<int64_t> position(ndim);

    // Sort Buffer is a Storage that will be used in the internal sort required
    // to calculate the mode efficiently
    auto sortBuffer = at::empty({sliceSize}, indices.options());

    // Call mode
    dimApplyMode<scalar_t>(
        valuesTransposed,
        indicesTransposed,
        self_contiguous,
        sortBuffer,
        dim,
        position,
        0);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

// ****************************** kthvalue *************************

template <typename scalar_t, typename index_t, int Dim>
__global__ void gatherKthValue(
    cuda::detail::TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,

    index_t numInputSlices,
    index_t inputWithinSliceStride,

    cuda::detail::TensorInfo<scalar_t, index_t> kthValue,
    cuda::detail::TensorInfo<int64_t, index_t> indices) {
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
  index_t kthValueSliceStartIndex =
      cuda::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
  index_t indicesSliceStartIndex =
      cuda::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

  scalar_t* inputSliceStart = &input.data[sliceStartIndex];
  scalar_t* kthValueSliceStart = &kthValue.data[kthValueSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  scalar_t kValue = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t,
      false>(
      inputSliceStart,
      k,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &kValue);

  // Find the index of the k-th highest element
  index_t kValueIndex = 0;
  bool foundKValue = false;

  for (index_t i = threadIdx.x; i < inputSliceSize; i += blockDim.x) {
    bool inRange = (i < inputSliceSize);
    scalar_t v = inRange ? doLdg(&inputSliceStart[i * inputWithinSliceStride])
                         : static_cast<scalar_t>(0);
    bool isKValue = inRange && (THCNumerics<scalar_t>::eq(v, kValue));

    if (isKValue) {
      kValueIndex = i;
      foundKValue = true;
      break;
    }
  }

  if (foundKValue) {
    kthValueSliceStart[0] = kValue;
    indicesSliceStart[0] = kValueIndex;
  }
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      cuda::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      cuda::detail::TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      cuda::detail::TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    dim3 grid;
    if (!getGridFromTiles(num_slices, grid)) {
      AT_ERROR("slices are too many");
    }

    dim3 block(
        std::min(THCRoundUp(slice_size, (int64_t)WARP_SIZE), (int64_t)1024));
    auto stream = at::cuda::getCurrentCUDAStream();
    gatherKthValue<scalar_t, index_t, all_dims><<<grid, block, 0, stream>>>(
        self_info,
        slice_size,
        k,
        num_slices,
        /* The actual dimension that the k-selection is running in */
        /* may have changed from collapseDims() */
        self_info.strides[collapse_self_dim],
        values_info,
        indices_info);
  }
};

template <typename scalar_t>
void kthvalue_cuda_template(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t slicesize = self.size(dim);
  AT_CHECK(k >= 1 && k <= slicesize, "selected number k out of range");

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return;
  }

  AT_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(values) &&
      cuda::detail::canUse32BitIndexMath(indices)) {
    run_launcher<scalar_t, uint32_t>(
        values, indices, self, dim, KthValueLauncher(k));
  } else {
    run_launcher<scalar_t, uint64_t>(
        values, indices, self, dim, KthValueLauncher(k));
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::tuple<Tensor&, Tensor&> kthvalue_out_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "kthvalue", [&] {
    kthvalue_cuda_template<scalar_t>(values, indices, self, k, dim, keepdim);
  });
  return std::forward_as_tuple(values, indices);
}

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

std::tuple<Tensor&, Tensor&> mode_out_cuda(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "mode", [&] {
    mode_cuda_template<scalar_t>(values, indices, self, dim, keepdim);
  });
  return std::forward_as_tuple(values, indices);
}

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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "sort", [&] {
      sortKeyValueInplace<scalar_t>(values, indices, dim, descending);
    });

  } else {
    // Otherwise, fall back upon Thrust, which handles all other cases
    // (potentially slowly, with extra copies/memory allocations)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "sort", [&] {
      sortViaThrust<scalar_t>(values, indices, self, dim, descending);
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return std::forward_as_tuple(values, indices);
}

} // namespace native
} // namespace at
