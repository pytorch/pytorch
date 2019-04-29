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


namespace {

#ifdef __HIP_PLATFORM_HCC__
static const int WARP_SIZE = 64;
#else
static const int WARP_SIZE = 32;
#endif




template <typename scalar_t>
__global__ void indexing_backward_kernel(
  int64_t* input, int64_t* indices, scalar_t* grad_output, scalar_t* grad_weight,
  int64_t numel, int64_t stride, int64_t stride_before) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel
      && (idx == 0 || input[idx] != input[idx - 1])){
    do {
      const int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weight_row = ((int) input[idx]) * stride + blockIdx.z * stride_before;
      const int grad_row = ((int) indices[idx]) * stride + blockIdx.z * numel * stride;
      const accscalar_t scale = (accscalar_t)1.0;

      accscalar_t gradient[SZ];
      accscalar_t weight[SZ];

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int feature_dim = start_feature + ii * WARP_SIZE;
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
        int feature_dim = start_feature + ii * WARP_SIZE;
        if (feature_dim < stride) {
            grad_weight[weight_row + feature_dim] = static_cast<scalar_t>(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}


}    


namespace at { namespace native {

static std::tuple<Tensor, int64_t, int64_t, int64_t> 
computeLinearIndex(const Tensor & src, TensorList indices) {
  auto strides = computeLinearStride(src);

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t emptyBefore = 0, emptyAfter = 0, nElemBefore = 1, nElemAfter = 1, strideBefore =0;
  for (int64_t i = 0; i < src.dim(); i++) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's backend
      // This allows us to support ie indexing a cuda tensor with a cpu tensor
      Tensor index = (wrapIndexOnce(indices[i], i, src.size(i)) * strides[i]).to(kLong);
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


static std::tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>> makeLinearIndex(Tensor self, TensorList orig) {
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
  std::tie(linearIndex, nElemBefore, strideBefore, nElemAfter) = computeLinearIndex(self, indices);
  return std::make_tuple(linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm);
}


    
Tensor & index_put_cuda_(Tensor & self, TensorList indices, const Tensor & value, bool accumulate) {
  if (indices.size() > (size_t)self.dim()) {
    AT_INDEX_ERROR("too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  }
  if (accumulate) {
    auto value_ = value.contiguous();
    Tensor linearIndex, expandedValue, src;
    int64_t nElemBefore, strideBefore, sliceSize;
    std::vector<int64_t> inversePerm;
    std::tie(linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) = makeLinearIndex(self, indices);
    bool permuted = !src.is_contiguous();
    auto src_ = permuted ? src.contiguous() : src;
    linearIndex = linearIndex.view(-1);
    auto sorted_indices = at::empty_like(linearIndex);
    auto orig_indices = at::empty_like(linearIndex);
    using device_ptr = thrust::device_ptr<int64_t>;
    int64_t num_indices = linearIndex.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Sort the inputs into sorted with the corresponding indices; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
    linearIndex.div_(sliceSize);
    {
    sorted_indices.copy_(linearIndex);
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    // Fill sortedOrigIndices with sequential indices
    auto count_iter = thrust::counting_iterator<int64_t>(0);
    auto orig_data = device_ptr(orig_indices.data<int64_t>());
    thrust::copy(policy, count_iter, count_iter + num_indices, orig_data);

    // Sort; a stable sort is not required
    auto sorted_data = device_ptr(sorted_indices.data<int64_t>());
    thrust::sort_by_key(policy, sorted_data, sorted_data + num_indices, orig_data, ThrustLTOp<int64_t>());
    }
    AT_ASSERT(linearIndex.numel()*sliceSize*nElemBefore == value.numel());
    dim3 grid(THCCeilDiv(num_indices, (int64_t) 4), THCCeilDiv(sliceSize, (int64_t) 128), std::max<int>(1,nElemBefore));
    dim3 block(32, 4);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_.scalar_type(), "embedding_backward", [&] {
    indexing_backward_kernel<<<grid, block, 0, stream>>>(
      sorted_indices.data<int64_t>(),
      orig_indices.data<int64_t>(),
      value_.data<scalar_t>(),
      src_.data<scalar_t>(),
      num_indices,
      sliceSize,
      strideBefore);
  });
  THCudaCheck(cudaGetLastError());
  if (permuted) {
     return self.copy_(src_.permute(inversePerm));
  } else 
     return self; 
  } else {
    return self;
  }
/*  auto info = make_info(self, indices);
  auto iter = make_index_put_iterator(info, value);
  index_put_stub(iter->device_type(), *iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;*/
}
}
}


