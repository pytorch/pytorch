#include "ATen/native/TensorTransformations.h"

#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"

namespace at {
namespace native {

template <typename scalar_t>
__global__
void flip_cuda_kernel(scalar_t* in_tensor, scalar_t* out_tensor, int64_t N, int64_t* flip_dims, int64_t flip_dims_size, int64_t* strides, int64_t* strides_contiguous, int64_t* shape, int64_t total_dims) {

  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }

  int64_t cur_indices = linear_index, rem = 0, dst_offset = 0;
  for (int64_t i = 0; i < total_dims; i++) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / strides_contiguous[i];
    rem = temp - cur_indices * strides_contiguous[i];
    // flip the indices if it is in flip_dims
    for (int64_t j = 0; j < flip_dims_size; j++) {
      if (i == flip_dims[j]) {
        cur_indices = shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * strides[i];
    cur_indices = rem;
  }
  out_tensor[linear_index] = in_tensor[dst_offset];
}

// Flip tensor given a list of dims
Tensor flip_cuda(const Tensor& self, IntList dims) {
  auto in_tensor = self;
  const int64_t flip_dims_size = dims.size(), total_dims = in_tensor.dim(), N = in_tensor.numel();
  check_errors(total_dims, flip_dims_size, dims);

  auto flip_dims = std::vector<int64_t>(dims);
  auto flip_dims_t = at::CPU(kLong).tensorFromBlob(flip_dims.data(), {static_cast<int64_t>(flip_dims.size())});

  auto shape = std::vector<int64_t>(in_tensor.sizes());
  auto shape_t = at::CPU(kLong).tensorFromBlob(shape.data(), {static_cast<int64_t>(shape.size())});

  auto strides = std::vector<int64_t>(in_tensor.strides());
  auto strides_t = at::CPU(kLong).tensorFromBlob(strides.data(), {static_cast<int64_t>(strides.size())});

  auto out_tensor = at::zeros_like(in_tensor);

  // stride_contiguous is the stride of non-contiguous tensor after called contiguous()
  // it is used to compute indices for each element in non-contiguous tensor
  Tensor stride_contiguous = at::zeros(CPU(kLong), {total_dims});
  int64_t* stride_contiguous_d = stride_contiguous.data<int64_t>();
  int64_t tmp = N;
  for (int64_t i = 0; i < total_dims; i++) {
    tmp = tmp / shape[i];
    stride_contiguous_d[i] = tmp;
  }

  int64_t block_size = 512;
  dim3 dim_block(block_size);
  dim3 dim_grid((N + block_size - 1) / block_size);

  AT_DISPATCH_ALL_TYPES_AND_HALF(in_tensor.type(), "flip_cuda", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    flip_cuda_kernel<<<dim_grid, dim_block, 0, globalContext().getCurrentCUDAStream()>>>(
      in_tensor.data<cuda_scalar_t>(), out_tensor.data<cuda_scalar_t>(), N, flip_dims_t.toType(CUDA(kLong)).data<int64_t>(), flip_dims_size, strides_t.toType(CUDA(kLong)).data<int64_t>(), stride_contiguous.toType(CUDA(kLong)).data<int64_t>(), shape_t.toType(CUDA(kLong)).data<int64_t>(), total_dims);
  });

  return out_tensor;
}

}} // namespace at::native
