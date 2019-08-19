#include <ATen/ATen.h>

namespace {

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename index_t, typename scalar_t, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      index_t linearId, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* indexOffset,
      const TensorInfo<scalar_t, index_t>& t1, index_t* t1Offset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      index_t curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      index_t linearId, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* indexOffset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      index_t curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename index_t, typename scalar_t>
struct IndexToScatterGatherOffsets<index_t, scalar_t, -1> {
  static __device__ void compute(
      index_t linearId, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* indexOffset,
      const TensorInfo<scalar_t, index_t>& t1, index_t* t1Offset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      index_t curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      index_t linearId, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* indexOffset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      index_t curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

template <typename index_t, typename scalar_t, int Dims>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void gather_kernel(
    TensorInfo<scalar_t, index_t> tensor,
    TensorInfo<scalar_t, index_t> src,
    TensorInfo<int64_t, index_t> index,
    const int dim,
    const index_t numel_) {
  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < numel;
       linearId += gridDim.x * blockDim.x) {
    index_t tensorOffset = 0;
    index_t srcOffset = 0;
    index_t indexOffset = 0;

    IndexToScatterGatherOffsets<index_t, scalar_t, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset,
                                                          src, &srcOffset);

    int64_t indexValue = index.data[indexOffset];
    assert(indexValue >= 0 && indexValue < src.sizes[dim]);
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

}  // namespace


#define RUN(TYPE, DIMS, scalar_t)                                           \
  gather_kernel<TYPE, scalar_t, DIMS><<<grid, block>>>(result_info, self_info, index_info, dim, static_cast<TYPE>(numel));

#undef RUN

namespace at { namespace native {

Tensor & gather_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == num_dims, "Index tensor must have same dimensions as input tensor");
  TORCH_CHECK(dim >= 0 && dim < num_dims, "Index dimension is out of bounds");
  TORCH_CHECK(std::max<int64_t>(result.dim(), 1) == num_dims, "Input tensor must have same dimensions as output tensor");
  for(int64_t i = 0; i < num_dims; i++) {
    if(i != dim) {
      AT_CHECK(index.size(i) == self.size(i), "Size does not match at dimension ", i, " get ", self.size(i), " vs ", index.size(i));
    }
  }
  result.resize_as_(index);

  int64_t numel = index.numel();
  int64_t block = 512;
  int64_t grid = std::min<int64_t>((size + block - 1) / block, 2048L);

  if (numel > 0) {
    if (THCTensor_canUse32BitIndexMath(state, tensor) &&
        THCTensor_canUse32BitIndexMath(state, src) &&
        THCTensor_canUse32BitIndexMath(state, index)) {
      auto result_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(result);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
      auto index_info = cuda::detail::getTensorInfo<int64_t, unsigned int>(index);

      // Specialize for a small number of dimensions.
      switch (index_info.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          break;
      }
    } else {
      auto result_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(result);
      auto self_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(self);
      auto index_info = cuda::detail::getTensorInfo<int64_t, uint64_t>(index);
      RUN(uint64_t, -1, scalar_t);
      THCudaCheck(cudaGetLastError());
    }
  }

  return result;
}

}}  // namespace at::native