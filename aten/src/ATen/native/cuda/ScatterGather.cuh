#pragma once
#include <ATen/cuda/CUDAApplyUtils.cuh>

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