#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/native/ScatterGather.h"

namespace {

using at::cuda::detail::TensorInfo;

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename index_t, typename scalar_t, int dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      index_t linear_id, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* index_offset,
      const TensorInfo<scalar_t, index_t>& t1, index_t* t1_offset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2_offset) {
    for (int d = dims - 1; d >= 0; d--) {
      index_t cur_dim_index = linear_id % index.sizes[d];
      *index_offset += cur_dim_index * index.strides[d];
      *t1_offset += cur_dim_index * t1.strides[d];
      if (d != dim) {
        *t2_offset += cur_dim_index * t2.strides[d];
      }
      linear_id /= index.sizes[d];
    }
  }

  static __device__ void compute(
      index_t linear_id, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* index_offset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2_offset) {
    for (int d = dims - 1; d >= 0; d--) {
      index_t cur_dim_index = linear_id % index.sizes[d];
      *index_offset += cur_dim_index * index.strides[d];
      if (d != dim) {
        *t2_offset += cur_dim_index * t2.strides[d];
      }
      linear_id /= index.sizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename index_t, typename scalar_t>
struct IndexToScatterGatherOffsets<index_t, scalar_t, -1> {
  static __device__ void compute(
      index_t linear_id, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* index_offset,
      const TensorInfo<scalar_t, index_t>& t1, index_t* t1_offset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2_offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      index_t cur_dim_index = linear_id % index.sizes[d];
      *index_offset += cur_dim_index * index.strides[d];
      *t1_offset += cur_dim_index * t1.strides[d];
      if (d != dim) {
        *t2_offset += cur_dim_index * t2.strides[d];
      }
      linear_id /= index.sizes[d];
    }
  }

  static __device__ void compute(
      index_t linear_id, const int dim,
      const TensorInfo<int64_t, index_t>& index, index_t* index_offset,
      const TensorInfo<scalar_t, index_t>& t2, index_t* t2_offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      index_t cur_dim_index = linear_id % index.sizes[d];
      *index_offset += cur_dim_index * index.strides[d];
      if (d != dim) {
        *t2_offset += cur_dim_index * t2.strides[d];
      }
      linear_id /= index.sizes[d];
    }
  }
};

template <typename index_t, typename scalar_t, int dims>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void gather_kernel(
    TensorInfo<scalar_t, index_t> tensor,
    TensorInfo<scalar_t, index_t> src,
    TensorInfo<int64_t, index_t> index,
    const int dim,
    const index_t numel) {
  for (index_t linear_id = blockIdx.x * blockDim.x + threadIdx.x;
       linear_id < numel;
       linear_id += gridDim.x * blockDim.x) {
    index_t tensor_offset = 0;
    index_t src_offset = 0;
    index_t index_offset = 0;

    IndexToScatterGatherOffsets<index_t, scalar_t, dims>::compute(linear_id, dim,
                                                          index, &index_offset,
                                                          tensor, &tensor_offset,
                                                          src, &src_offset);

    int64_t index_value = index.data[index_offset];
    assert(index_value >= 0 && index_value < src.sizes[dim]);
    src_offset += index_value * src.strides[dim];

    tensor.data[tensor_offset] = src.data[src_offset];
  }
}

}  // namespace

namespace at { namespace native {

Tensor & gather_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  TORCH_CHECK(std::max<int64_t>(index.dim(), 1) == num_dims, "Index tensor must have same dimensions as input tensor");
  dim = c10::maybe_wrap_dim(dim, self.dim());

  std::vector<int64_t> self_sizes = self.sizes().vec();
  std::vector<int64_t> index_sizes = index.sizes().vec();
  ensure_nonempty(self_sizes);
  ensure_nonempty(index_sizes);

  for(int64_t i = 0; i < num_dims; i++) {
    if(i != dim) {
      TORCH_CHECK(index_sizes[i] == self_sizes[i], "Size does not match at dimension ", i, " get ", self_sizes[i], " vs ", index_sizes[i]);
    }
  }
  result.resize_as_(index);

  int64_t numel = index.numel();
  int64_t block = 512;
  int64_t grid = std::min<int64_t>((numel + block - 1) / block, 2048L);

  if (numel > 0) {
    AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Bool, ScalarType::Half, self.scalar_type(), "gather_out_cuda", [&](){
      if (cuda::detail::canUse32BitIndexMath(result) &&
          cuda::detail::canUse32BitIndexMath(self) &&
          cuda::detail::canUse32BitIndexMath(index)) {
        auto result_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(result);
        auto self_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(self);
        auto index_info = cuda::detail::getTensorInfo<int64_t, unsigned int>(index);

        // Specialize for a small number of dimensions.
        switch (index_info.dims) {
          case 1:
            gather_kernel<unsigned int, scalar_t, 1><<<grid, block>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
          case 2:
            gather_kernel<unsigned int, scalar_t, 2><<<grid, block>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
          case 3:
            gather_kernel<unsigned int, scalar_t, 3><<<grid, block>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
          default:
            gather_kernel<unsigned int, scalar_t, -1><<<grid, block>>>(
              result_info, self_info, index_info, dim, static_cast<unsigned int>(numel));
            break;
        }
      } else {
        auto result_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(result);
        auto self_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(self);
        auto index_info = cuda::detail::getTensorInfo<int64_t, uint64_t>(index);
        gather_kernel<uint64_t, scalar_t, -1><<<grid, block>>>(
          result_info, self_info, index_info, dim, static_cast<uint64_t>(numel));
      }
    });
  }
  return result;
}

}}  // namespace at::native
