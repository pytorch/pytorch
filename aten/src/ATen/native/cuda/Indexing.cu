#include <ATen/native/Indexing.h>
#include <ATen/native/IndexingUtils.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AccumulateType.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCTensorSort.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

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
  for (int z = blockIdx.z; z < outer_dim; z += gridDim.z){
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx < numel
        && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])){
      do {
        int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        const int weight_row = ((int) sorted_indices[idx]) * stride + z * stride_before;
        const int grad_row = ((int) indices[idx]) * stride + z * numel * stride;
        const accscalar_t scale = (accscalar_t)1.0;

        accscalar_t gradient[SZ];
        accscalar_t weight[SZ];

        while (start_feature < stride) {
          #pragma unroll
          for (int ii = 0; ii < SZ; ii++) {
            int feature_dim = start_feature + ii * C10_WARP_SIZE;
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
            int feature_dim = start_feature + ii * C10_WARP_SIZE;
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
      AT_INDEX_ERROR("index ", max_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
    if (min_idx < -dim_size) {
      AT_INDEX_ERROR("index ", min_idx, " is out of bounds for dimension ", dim, " with size ", dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor & tensor) {
  // computes the stride as if tensor were contigous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(sizes.rbegin(), sizes.rend() - 1, stride.rbegin() + 1, std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t> 
computeLinearIndex(const Tensor & src, TensorList indices, bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& backend = src.type().backend();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  for (auto i = decltype(src.dim()){0}; i < src.dim(); i++) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's backend
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i]).toBackend(backend);
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


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>> makeLinearIndex(Tensor self, TensorList orig, bool check_range) {
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
void index_put_accum_kernel(Tensor & self, TensorList indices, const Tensor & value, bool unsafe) {
  if (indices.size() > (size_t)self.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
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
      linearIndex = linearIndex.view(-1);
      auto sorted_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      auto orig_indices = at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      using device_ptr = thrust::device_ptr<int64_t>;
      const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
      linearIndex.div_(sliceSize);
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
      TORCH_CHECK(self.numel() < std::numeric_limits<int>::max(), "index_put_ with accumulation is not supported on large tensors, number of source elements =", self.numel(), "file a support request on github");
      TORCH_CHECK(value.numel() < std::numeric_limits<int>::max(), "index_put_ with accumulation is not supported on large tensors, number of source elements =", value.numel(), "file a support request on github");
      const int UNROLL = 4;
      const int indices_per_block = 4;
      dim3 grid(THCCeilDiv(num_indices, (int64_t) indices_per_block),
           std::min<int>(at::cuda::getCurrentDeviceProperties()->maxGridSize[1], THCCeilDiv(sliceSize, (int64_t) (C10_WARP_SIZE*UNROLL))),
           std::min(std::max<int>(1,nElemBefore), at::cuda::getCurrentDeviceProperties()->maxGridSize[2]));
      dim3 block(C10_WARP_SIZE, indices_per_block);
  
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_.scalar_type(), "embedding_backward", [&] {
      indexing_backward_kernel<scalar_t, UNROLL><<<grid, block, 0, stream>>>(
        sorted_indices.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>(),
        value_.data_ptr<scalar_t>(),
        src_.data_ptr<scalar_t>(),
        num_indices,
        sliceSize,
        strideBefore,
        nElemBefore);
      });
      THCudaCheck(cudaGetLastError());
      if (permuted)
          self.copy_(src_.permute(inversePerm));
  }
}

REGISTER_CUDA_DISPATCH(index_put_accum_stub, &index_put_accum_kernel);
} //anonymous
} //at
} //native


