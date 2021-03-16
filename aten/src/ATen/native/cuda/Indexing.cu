#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCTensorSort.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <THC/THCAtomics.cuh>

#include <cub/cub.cuh>

#include <c10/macros/Macros.h>

namespace {

template <typename scalar_t, int SZ>
__global__ void indexing_backward_kernel(
  int64_t* sorted_indices, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before, int64_t outer_dim) {
//numel is total number of flattened indices, not expanded to dimensions that are not indexed.
//stride is the cumulative size of the not-indexed last dimensions
//stride_before is the stride of the dimension immediately preceding first indexed dimension
//if indexing starts from the 0th dimension, stride_before does not matter because blockIdx.z will be 0 in this case
//outer_dim is number of elements in the first unindexed dimensions
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same destination index as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values processed by each thread (grain size)
  for (int64_t z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int64_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int64_t start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        const int64_t weight_row = ((int64_t) sorted_indices[idx]) * stride + z * stride_before;
        const int64_t grad_row = ((int64_t) indices[idx]) * stride + z * numel * stride;
        const accscalar_t scale = (accscalar_t)1.0;

        accscalar_t gradient[SZ];
        accscalar_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
              gradient[ii] = static_cast<accscalar_t>(grad_output[grad_row + feature_dim]);
              weight[ii] = static_cast<accscalar_t>(grad_weight[weight_row + feature_dim]);
            }
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            weight[ii] += gradient[ii] * scale;
          }

          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int64_t feature_dim = start_feature + ii * C10_WARP_SIZE;
            if (feature_dim < stride) {
                grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
            }
          }
          start_feature += gridDim.y * blockDim.x * SZ;
        }

        idx++;
      } while (idx < numel && sorted_indices[idx] == sorted_indices[idx - 1]);
    }
  }
}


}


namespace at { namespace native {

static Tensor wrapIndexOnce(const Tensor & index, int64_t dim, int64_t dim_size, bool check_range=true) {
//we don't need to check range in backward - if there were out of bounds indices forward should already have errored out
  if (index.numel() != 0 && check_range) {
    auto max_idx = index.max().item<int64_t>();
    auto min_idx = index.min().item<int64_t>();
    if (max_idx >= dim_size) {
      TORCH_CHECK_INDEX(false, "index ", max_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
    if (min_idx < -dim_size) {
      TORCH_CHECK_INDEX(false, "index ", min_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t>
computeLinearIndex(const Tensor & src, TensorList indices, bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  for (auto i = decltype(src.dim()){0}; i < src.dim(); i++) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i]).to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i>0) {
           strideBefore = src.stride(i-1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      emptyAfter++;
      nElemAfter *= src.size(i);
    } else {
      emptyBefore++;
      nElemBefore *= src.size(i);
    }
  }

  return std::make_tuple(std::move(linearIndex), nElemBefore, strideBefore, nElemAfter);
}


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>> makeLinearIndex(Tensor self, const c10::List<c10::optional<at::Tensor>>& orig, bool check_range) {
  checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) = transposeToFrontAndInvPerm(self, indices);
  }
  int64_t nElemBefore, strideBefore, nElemAfter;
  Tensor linearIndex;
  std::tie(linearIndex, nElemBefore, strideBefore, nElemAfter) = computeLinearIndex(self, indices, check_range);
  return std::make_tuple(linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm);
}


namespace {
void index_put_accum_kernel(Tensor & self, const c10::List<c10::optional<Tensor>>& indices, const Tensor & value, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(false, "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  auto value_ = value.contiguous();
  Tensor linearIndex, expandedValue, src;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) = makeLinearIndex(self, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();
  if (num_indices > 0 && sliceSize > 0) {
      const bool permuted = !src.is_contiguous();
      auto src_ = permuted ? src.contiguous() : src;
      linearIndex = linearIndex.reshape(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      using device_ptr = thrust::device_ptr<int64_t>;
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      linearIndex.divide_(sliceSize, "trunc");
      {
      sorted_indices.copy_(linearIndex);
      auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
      auto policy = thrust::cuda::par(allocator).on(stream);

      // Fill sortedOrigIndices with sequential indices
      const auto count_iter = thrust::counting_iterator<int64_t>(0);
      auto orig_data = device_ptr(orig_indices.data_ptr<int64_t>());
      thrust::copy(policy, count_iter, count_iter + num_indices, orig_data);

      // Sort the inputs into sorted with the corresponding indices; we
      // don't need a stable or multidimensional sort, so just use Thrust
      // directly
      // Sort; a stable sort is not required
      // NB - not passing comparator causes thrust to use radix sort, and it hurts perf A LOT, at least for medium (few K) sized indices
      auto sorted_data = device_ptr(sorted_indices.data_ptr<int64_t>());
      thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data, ThrustLTOp<int64_t>());
      }
      TORCH_INTERNAL_ASSERT(linearIndex.numel()*sliceSize*nElemBefore == value.numel(), "number of flattened indices did not match number of elements in the value tensor", linearIndex.numel()*sliceSize*nElemBefore, value.numel());
      const int UNROLL = 4;
      const int indices_per_block = 4;
      dim3 grid(THCCeilDiv(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], THCCeilDiv(sliceSize, (int64_t) (C10_WARP_SIZE*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(C10_WARP_SIZE, indices_per_block);

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      value_.scalar_type(), "indexing_backward", [&] {
        indexing_backward_kernel<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
          sorted_indices.data_ptr<int64_t>(),
          orig_indices.data_ptr<int64_t>(),
          value_.data_ptr<scalar_t>(),
          src_.data_ptr<scalar_t>(),
          num_indices,
          sliceSize,
          strideBefore,
          nElemBefore);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

      if (permuted)
          self.copy_(src_.permute(inversePerm));
  }
}

REGISTER_CUDA_DISPATCH(index_put_accum_stub, &index_put_accum_kernel);
} //anonymous


// Check tensor dimensions for index operations, and return the slice size.
static ptrdiff_t getSliceSize(const Tensor & dst,
                              int dim,
                              const Tensor & index,
                              const Tensor & src)
{
  int dstDims = dst.dim();
  int srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  ptrdiff_t dstSliceSize = 1;
  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }

  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(index.numel() == src.size(dim),
             "length of src.size[dim] is not equal to length of indices");

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims) mismatch = true;

  for (int d = 0; d < srcDims; d++) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d)) mismatch = true;
    }
  }

  TORCH_CHECK(dstSliceSize == srcSliceSize,
             "Source/destination tensor have different slice sizes (%ld vs %ld)",
             dstSliceSize, srcSliceSize);

  if (mismatch) {
    TORCH_WARN_ONCE(
        "Warning: source/destination slices have same size but different "
        "shape for an index operation.  This behavior is deprecated.\n");
  }

  return dstSliceSize;
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexAddLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                   cuda::detail::TensorInfo<T, IndexType> src,
                                   cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   int64_t dstAddDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      gpuAtomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexAddSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexAddLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                   cuda::detail::TensorInfo<T, IndexType> src,
                                   cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType totalSize,
                                   IndexType innerSize,
                                   int64_t dstAddDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
        indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(srcIndex, indices)];
    CUDA_KERNEL_ASSERT(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    gpuAtomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
  }
}

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.
template <typename scalar_t>
bool indexShouldBeMajor(cuda::detail::TensorInfo<scalar_t, unsigned int> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (int i = 0; i < info.dims; ++i) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

Tensor& index_add_cuda_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("index_add_cuda_");
  dim = maybe_wrap_dim(dim, self.dim());

  TensorArg self_arg{self, "self", 1}, index_arg{index, "index", 3}, source_arg{source, "source", 4};
  checkAllSameGPU("index_add", {self_arg, index_arg, source_arg});

  TORCH_CHECK_INDEX(index.dim() <= 1, "index_add_(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_add_(): Expected dtype int32/int64 for index");
  TORCH_CHECK(self.scalar_type() == source.scalar_type(),
              "index_add_(): self and source must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < source.dim(),
              "index_add_(): Indexing dim ", dim, " is out of bounds of tensor");
  TORCH_CHECK(index.numel() == (source.dim() == 0 ? 1 : source.size(dim)),
              "index_add_(): Number of indices should be equal to self.size(dim)");

  at::assert_no_internal_overlap(self);
  at::assert_no_overlap(self, index);
  at::assert_no_overlap(self, source);

  // Scalars are treated as 1-d tensor
  Tensor self_ = (self.dim() == 0) ? self.view(1) : self;
  Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(self.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);
  TORCH_CHECK(source.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);

  at::assert_no_internal_overlap(self);
  at::assert_no_partial_overlap(self, index);
  at::assert_no_partial_overlap(self, source);

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);
  ptrdiff_t sourceTotalSize = source.numel();
  int64_t selfAddDimSize = self_.size(dim);
  ptrdiff_t numIndex = index.numel();

  if (sliceSize == 0) {
    return self;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool indContig = index.is_contiguous();

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)  \
  indexAddSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM> \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                \
      selfInfo, sourceInfo, indexInfo,                                               \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize);                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexAddLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                       \
                     SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>           \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo,                                      \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize);                                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(sourceTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(sourceTotalSize, (ptrdiff_t)128));

  if (cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(source) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, unsigned int> selfInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(self_);
      int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);
      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        auto sourceInfo =
          cuda::detail::getTensorInfo<scalar_t, unsigned int>(source_);
        int sourceAddDim = sourceInfo.collapseDims(dim);
        sourceInfo.reduceDim(sourceAddDim);

        auto indexInfo =
        cuda::detail::getTensorInfo<index_t, unsigned int>(index);
        indexInfo.collapseDims();

        // A reasonable choice for when to have each thread iterate over
        // index to choose
        if (numIndex <= 16) {
          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
          } else {
            SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
          }
        } else {
          bool indexIsMajor = indexShouldBeMajor(selfInfo, selfAddDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
            } else {
              LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
            }
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "index_add", [&] {
      cuda::detail::TensorInfo<scalar_t, uint64_t> selfInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(self_);
      int selfAddDim = selfInfo.collapseDims(dim);
      selfInfo.reduceDim(selfAddDim);

      cuda::detail::TensorInfo<scalar_t, uint64_t> sourceInfo =
        cuda::detail::getTensorInfo<scalar_t, uint64_t>(source_);
      int sourceAddDim = sourceInfo.collapseDims(dim);
      sourceInfo.reduceDim(sourceAddDim);

      AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_cuda_", [&] () {
        cuda::detail::TensorInfo<index_t, uint64_t> indexInfo =
          cuda::detail::getTensorInfo<index_t, uint64_t>(index);
        indexInfo.collapseDims();

        LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
      });
    });
  }

  return self;
#undef SMALL_INDEX
#undef LARGE_INDEX
}

namespace {
// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<T, IndexType> src,
                                      cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndicesType, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexSelectLargeIndex(cuda::detail::TensorInfo<T, IndexType> dst,
                                      cuda::detail::TensorInfo<T, IndexType> src,
                                      cuda::detail::TensorInfo<IndicesType, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex, elementInSlice;
    if (IndexIsMajor) {
      dstIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      dstIndex = linearIndex % innerSize;
    }

    IndexType srcIndex =
      indices.data[cuda::detail::IndexToOffset<IndicesType, IndexType, IdxDim>::get(dstIndex, indices)];
    CUDA_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

    IndexType dstOffset =
      cuda::detail::IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      cuda::detail::IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

namespace {

// When using a 0-dim scalar tensor, we need the legacy (THC) semantics of
// TensorInfo: Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
cuda::detail::TensorInfo<T, IndexType>
tensorInfoLegacyIfScalar(cuda::detail::TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

}

template<typename scalar_t>
void index_select_out_cuda_impl(Tensor& out, const Tensor& self, long dim,
                                const Tensor& index) {
  ptrdiff_t numIndices = index.numel();

  int selfDims = self.dim() == 0 ? 1 : self.dim();

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(index.dim() <= 1,
             "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }
  at::native::resize_(out, newSize, {});

  ptrdiff_t outTotalSize = out.numel();
  if (outTotalSize == 0) {
    return;
  }

  bool indContig = index.is_contiguous();

  // The `self` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int64_t selfSelectDimSize = self.dim() == 0 ? 1 : self.size(dim);
  ptrdiff_t sliceSize = outTotalSize / numIndices;

  int mpc = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         \
  indexSelectSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>     \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      outInfo, selfInfo, indicesInfo,                                                   \
      outSelectDim, selfSelectDim, static_cast<TYPE>(sliceSize),                        \
      selfSelectDimSize);                                                               \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LARGE_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE,                           \
                    DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR)                   \
  indexSelectLargeIndex<TENSOR_TYPE, INDICES_TYPE, TYPE,                       \
                        DST_DIM, SRC_DIM, IDX_DIM, IDX_IS_MAJOR>               \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                          \
      outInfo, selfInfo, indicesInfo,                                          \
      outSelectDim, selfSelectDim, static_cast<TYPE>(outTotalSize),            \
      static_cast<TYPE>((IDX_IS_MAJOR) ? sliceSize : numIndices),              \
      selfSelectDimSize);                                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(outTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(outTotalSize, (ptrdiff_t)128));
  if (cuda::detail::canUse32BitIndexMath(out) &&
      cuda::detail::canUse32BitIndexMath(self) &&
      cuda::detail::canUse32BitIndexMath(index)) {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, unsigned int>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto  selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, unsigned int>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);

    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<index_t, unsigned int>(index));
      indicesInfo.collapseDims();

      // A reasonable choice for when to have each thread iterate over
      // indices to choose
      if (numIndices <= 16) {
        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
          SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
        } else {
          SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
        }
      } else {
        bool indexIsMajor = indexShouldBeMajor(outInfo, outSelectDim);

        if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
          LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
        } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
          if (indexIsMajor) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
          }
        } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
          if (indexIsMajor) {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
          } else {
            LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
          }
        } else {
          LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
        }
      }
    });
  } else {
    auto outInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, uint64_t>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto selfInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<scalar_t, uint64_t>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);
    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_select_out_cuda_impl", [&] () {
      auto indicesInfo = tensorInfoLegacyIfScalar(cuda::detail::getTensorInfo<index_t, uint64_t>(index));
      indicesInfo.collapseDims();

      LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
    });
  }
#undef SMALL_INDEX
#undef LARGE_INDEX
}
} // anonymous namespace

Tensor& index_select_out_cuda(Tensor& out, const Tensor& self, int64_t dim,
                              const Tensor& index) {
  static constexpr string_view DIM_WARNING =
    "Tensor too large or too many (> 25) dimensions";

  TORCH_CHECK(at::cuda::check_device({out, self, index}),
              "Input, output and indices must be on the current device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= MAX_TENSORINFO_DIMS, DIM_WARNING);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
      out.scalar_type(), "index_select_cuda",
      [&] { index_select_out_cuda_impl<scalar_t>(out, self, dim, index); });

  return out;
}

Tensor index_select_cuda(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor out = at::empty({0}, self.options());
  index_select_out_cuda(out, self, dim, index);
  return out;
}

template<typename T>
struct NonZeroOp
{
    __host__ __device__ __forceinline__ bool operator()(const T& a) const {
      return (a!=T(0));
    }
};

template<typename scalar_t>
void nonzero_cuda_out_impl(const Tensor& self, Tensor& out){
  Tensor self_ = self.contiguous();
  int N = self_.numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
// compute number of nonzero elements
  size_t temp_storage_bytes=0;
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto num_nonzeros = allocator.allocate(sizeof(int));
  cub::TransformInputIterator<bool, NonZeroOp<scalar_t>, scalar_t*> itr(self_.data_ptr<scalar_t>(), NonZeroOp<scalar_t>());
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, (int*)num_nonzeros.get(), N, stream);
  auto temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceReduce::Sum(temp_storage.get(), temp_storage_bytes, itr, (int*)num_nonzeros.get(), N, stream);
  int num_nonzeros_h;
  C10_CUDA_CHECK(cudaMemcpyAsync(&num_nonzeros_h, num_nonzeros.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  //need to synchronize to make sure data is available on the host
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  //expected output size is num_nonzeros x ndim
  //we are producing output with size {num_nonzeros, ndim} and strides {num_nonzeros, 1} (that is, transposed ndim x num_nonzeros output)
  //we are able to directly use passed output with this size and strides, and we can also (per contract)
  //resize passed output with incorrect sizes anyway we want.
  //However, out with correct sizes and incorrect strides will have to be copied to from the intermediate we've produced.
  bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros_h && out.sizes()[1] == self.dim() && !out.t().is_contiguous();
  at::Tensor out_temp = need_to_copy ?
    at::native::empty_cuda({self.dim(), num_nonzeros_h}, optTypeMetaToScalarType(out.options().dtype_opt()),
                           out.options().layout_opt(), out.options().device_opt(), out.options().pinned_memory_opt()) :
    out.resize_({self.dim(), num_nonzeros_h});
  //Scalars are expected to produce output of size (1,0), so we can't write to it
  if (self.dim() > 0) {
    cub::CountingInputIterator<int64_t> counting_itr(0);
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr,
      out_temp.data_ptr<int64_t>(), (int*)num_nonzeros.get(), N, stream);
    temp_storage = allocator.allocate(temp_storage_bytes);
    cub::DeviceSelect::Flagged(temp_storage.get(), temp_storage_bytes, counting_itr, itr,
      out_temp.data_ptr<int64_t>(), (int*)num_nonzeros.get(), N, stream);
    if (num_nonzeros_h > 0 && self.dim() > 1){
        int64_t div = 1;
        auto thrust_allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
        for (int dim = self.dim()-1; dim >= 0; dim--){
            int64_t dim_size = self.sizes()[dim];
            thrust::transform(
              thrust::cuda::par(thrust_allocator).on(stream),
              thrust::device_ptr<int64_t>(out_temp.data_ptr<int64_t>()),
              thrust::device_ptr<int64_t>(out_temp.data_ptr<int64_t>()) + num_nonzeros_h,
              thrust::device_ptr<int64_t>(out_temp.data_ptr<int64_t>()) + num_nonzeros_h * dim,
              [=] C10_HOST_DEVICE (const int64_t val) {return (val/div) % dim_size;}
            );
            div *= dim_size;
        }
    }
  }
  if (need_to_copy) {
    out.copy_(out_temp.t());
  } else {
    //transpose out so it is correct size
    Tensor out_ = out_temp.t();
    out.set_(out_);
  }
}

Tensor& nonzero_out_cuda(Tensor& out, const Tensor& self){
  TORCH_CHECK(self.numel() < std::numeric_limits<int>::max(), "nonzero is not supported for tensors with more than INT_MAX elements, \
  file a support request");
  TORCH_CHECK(out.dtype() == at::kLong, "Expected object of scalar type ", at::kLong, " as out, but got ", out.dtype());
  TORCH_CHECK(self.device() == out.device(), "expected self and out to be on the same device, but got out on ",
  out.device(), " and self on ", self.device());
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
    self.scalar_type(), "nonzero_cuda",
    [&] {nonzero_cuda_out_impl<scalar_t>(self, out);});
  return out;
}

Tensor nonzero_cuda(const Tensor& self){
  Tensor out = at::native::empty_cuda({0}, kLong, self.options().layout_opt(), self.options().device_opt(), self.options().pinned_memory_opt());
  return nonzero_out_cuda(out, self);
}

namespace {

template <typename mask_t>
void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBool, kHalf, kBFloat16, iter.common_dtype(), "masked_fill_", [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel(
            iter, [value_] GPU_LAMBDA(scalar_t self, mask_t mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

} // anonymous namespace

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Scalar& value) {
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kByte || mask.scalar_type() == kBool,
    "expected mask dtype to be Bool but got ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::YES) {
    TORCH_WARN(
      "Use of masked_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  Tensor b_mask;
  std::tie(b_mask) = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(b_mask)
      .build();

  if (b_mask.dtype() == at::ScalarType::Byte) {
    TORCH_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    masked_fill_kernel<uint8_t>(iter, value);
  } else {
    masked_fill_kernel<bool>(iter, value);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  return masked_fill__cuda(self, mask, value.item());
}

} // native
} // at
