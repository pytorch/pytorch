#include "ATen/NativeFunctions.h"
#include "ATen/ATen.h"
#include <algorithm>
#include <sstream>

#include "ATen/cuda/AccumulateType.cuh"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"


namespace at {
namespace native {

// Map the index of an element in tensor from 1D to nD
__device__ __forceinline__
void linear_index_to_indices(int64_t linear_index, int64_t* each_dim_len, int64_t total_dims, int64_t* indices) {
  int64_t res = linear_index;
  for (int64_t i = 0; i < total_dims; i++) {
    int64_t indices_i = linear_index * total_dims + i;
    indices[indices_i] = res / each_dim_len[i];
    res = res % each_dim_len[i];
  }
}

/*
Map the index of an element in tensor from nD to 1D. A tensor is originally in nD shape,
and 1D is the unfolded version of it (a vector).

Example: given a 3D tensor
[
  [ [1,2], [3,4] ],
  [ [5,6], [7,8] ]
]

Here element 3 has nD index = (0,1,0), and this corresponds to oneD index = 2
*/
__device__ __forceinline__
int64_t indices_to_linear_index(int64_t* indices, int64_t total_dims, int64_t* each_dim_len, int64_t src_linear_index) {
  int64_t dest_linear_index = 0;
  for (int64_t i = 0; i < total_dims; i++) {
    int64_t indices_i = src_linear_index * total_dims + i;
    dest_linear_index += indices[indices_i] * each_dim_len[i];
  }
  return dest_linear_index;
}

template <typename scalar_t>
__global__
void flip_cuda_kernel(scalar_t* in_t, scalar_t* out_t, int64_t N, int64_t* dims, int64_t* indices,
  int64_t flip_dims_size, int64_t* each_dim_len, int64_t* shape, int64_t total_dims) {

  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_index >= N) {
    return;
  }

  linear_index_to_indices(linear_index, each_dim_len, total_dims, indices);

  // flip nD index along each desired dimension
  for (int64_t i = 0 ; i < flip_dims_size; i++) {
    int64_t dim = dims[i];
    int64_t indices_dim = linear_index * total_dims + dim;
    indices[indices_dim] = shape[dim] - 1 - indices[indices_dim];
  }
  int64_t dest_linear_index = indices_to_linear_index(indices, total_dims, each_dim_len, linear_index);
  out_t[linear_index] = in_t[dest_linear_index];
}

Tensor flip_cuda(const Tensor& self, IntList dims) {

  // TODO: allow non-contiguous tensors
  self.contiguous();

  int64_t flip_dims_size = dims.size(), total_dims = self.dim(), N = self.numel();

  // check if number of axis in dim is valid
  if (flip_dims_size == 0) {
    std::stringstream ss;
    ss << "expected input tensor dims not empty, "
       << "but got tensor dims size=" << flip_dims_size;
    throw std::runtime_error(ss.str());
  }

  // check duplicates in dims
  auto flip_dims_v = std::vector<int64_t>(dims);
  flip_dims_v.erase(std::unique(flip_dims_v.begin(), flip_dims_v.end()), flip_dims_v.end());
  if ((int64_t)flip_dims_v.size() < flip_dims_size) {
    std::stringstream ss;
    ss << "dims has duplicates, "
       << "original flip dims size=" << flip_dims_size << ", "
       << "but unique flip dims size= " << flip_dims_v.size();
    throw std::runtime_error(ss.str());
  }

  // check len of dims
  if (flip_dims_size > total_dims) {
    std::stringstream ss;
    ss << "expected flip dims size <= tensor total dims, "
       << "but got flip dims size=" << flip_dims_size << " and "
       << "tensor total dim=" << total_dims;
    throw std::runtime_error(ss.str());
  }

  // check if dims axis within range
  int64_t min_d = total_dims, max_d = 0;
  for (auto d : dims) {
    min_d = std::min(min_d, d);
    max_d = std::max(max_d, d);
  }

  if (min_d < 0) {
    std::stringstream ss;
    ss << "expected flip dims axis >= 0, "
       << "but got min flip dims=" << min_d;
    throw std::runtime_error(ss.str());
  }

  if (max_d >= total_dims) {
    std::stringstream ss;
    ss << "expected flip dims axis < tensor total dims, "
       << "but got max flip dims=" << max_d << " and "
       << "tensor total dim=" << total_dims;
    throw std::runtime_error(ss.str());
  }

  Tensor flip_dims_t = at::zeros(CPU(kLong), {flip_dims_size});
  int64_t* flip_dims_t_d = flip_dims_t.data<int64_t>();
  for (int64_t i = 0; i < flip_dims_size; i++) {
    flip_dims_t_d[i] = dims[i];
  }

  auto shape = self.sizes();
  Tensor shape_t = at::zeros(CPU(kLong), {total_dims});
  int64_t* shape_t_d = shape_t.data<int64_t>();
  for (int64_t i = 0; i < total_dims; i++) {
    shape_t_d[i] = shape[i];
  }

  Tensor each_dim_len = at::zeros(CPU(kLong), {total_dims});
  int64_t* each_dim_len_d = each_dim_len.data<int64_t>();
  int64_t tmp = N;
  for (int64_t i = 0; i < total_dims; i++) {
    tmp = tmp / shape[i];
    each_dim_len_d[i] = tmp;
  }

  Tensor indices = at::zeros(CUDA(kLong), {N, total_dims});
  Tensor out_t = self.clone();

  int64_t block_size = 512;
  dim3 dim_block(block_size);
  dim3 dim_grid((N + block_size - 1) / block_size);

  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "flip_cuda", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    flip_cuda_kernel<<<dim_grid, dim_block, 0, globalContext().getCurrentCUDAStream()>>>(
      self.data<cuda_scalar_t>(), out_t.data<cuda_scalar_t>(), N, flip_dims_t.toType(CUDA(kLong)).data<int64_t>(), indices.data<int64_t>(), flip_dims_size, each_dim_len.toType(CUDA(kLong)).data<int64_t>(), shape_t.toType(CUDA(kLong)).data<int64_t>(), total_dims);
  });

  return out_t;
}

}} // namespace at::native
