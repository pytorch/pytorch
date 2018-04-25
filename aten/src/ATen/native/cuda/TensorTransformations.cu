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
void oneD_to_nD(int64_t oneD_index, int64_t* shape_size, int64_t shape_len, int64_t* nD_index) {
  int64_t res = oneD_index;
  for (int i = 0; i < shape_len; i++) {
    int64_t nD_i = oneD_index * shape_len + i;
    nD_index[nD_i] = res / shape_size[i];
    res = res % shape_size[i];
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
int64_t nD_to_oneD(int64_t* nD_index, int64_t shape_len, int64_t* shape_size, int64_t src_oneD_index) {
  int64_t dest_oneD_index = 0;
  for (int i = 0; i < shape_len; i++) {
    int64_t nD_i = src_oneD_index * shape_len + i;
    dest_oneD_index += nD_index[nD_i] * shape_size[i];
  }
  return dest_oneD_index;
}

template <typename T>
__global__
void flip_cuda_kernel(T* in_t, T* out_t, int64_t N, int64_t* dims, int64_t* nD_index,
  int64_t dims_len, int64_t* shape_size, int64_t* shape, int64_t shape_len) {

  int64_t oneD_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (oneD_index >= N) {
    return;
  }

  oneD_to_nD(oneD_index, shape_size, shape_len, nD_index);
  // flip nD index along each desired dimension
  for (int i = 0 ; i < dims_len; i++) {
    int64_t d = dims[i];
    int64_t nD_d = oneD_index * shape_len + d;
    nD_index[nD_d] = shape[d]-1-nD_index[nD_d];
  }
  int64_t dest_oneD_index = nD_to_oneD(nD_index, shape_len, shape_size, oneD_index);
  out_t[oneD_index] = in_t[dest_oneD_index];
}

Tensor flip_cuda(const Tensor& self, IntList dims) {

  int64_t dims_len = dims.size(), shape_len = self.dim(), N = self.numel();

  // check if number of axis in dim is valid
  if (dims_len == 0) {
    std::stringstream ss;
    ss << "expected dims not empty, "
       << "but got dims size=" << dims_len;
    throw std::runtime_error(ss.str());
  }

  // check duplicates in dims
  auto dims_v = std::vector<int64_t>(dims);
  dims_v.erase(std::unique(dims_v.begin(), dims_v.end()), dims_v.end());
  if (dims_v.size() < dims_len) {
    std::stringstream ss;
    ss << "dims has duplicates, "
       << "input dims size=" << dims_len << ", "
       << "but unique dims size= " << dims_v.size();
    throw std::runtime_error(ss.str());
  }

  // check len of dims
  if (dims_len > shape_len) {
    std::stringstream ss;
    ss << "expected dims to have size <= total tensor dims, "
       << "but got dims size=" << dims_len << " and "
       << "tensor dim=" << shape_len;
    throw std::runtime_error(ss.str());
  }

  // check if dims axis within range
  int64_t min_d = shape_len, max_d = 0;
  for (auto d : dims) {
    min_d = std::min(min_d, d);
    max_d = std::max(max_d, d);
  }

  if (min_d < 0) {
    std::stringstream ss;
    ss << "expected dims axis >= 0, "
       << "but got min dims=" << min_d;
    throw std::runtime_error(ss.str());
  }

  if (max_d >= shape_len) {
    std::stringstream ss;
    ss << "expected dims axis < total tensor dims, "
       << "but got max dims=" << max_d << " and "
       << "tensor dim=" << shape_len;
    throw std::runtime_error(ss.str());
  }

  Tensor dims_t = at::zeros(CPU(kLong), {dims_len});
  int64_t* dims_t_d = dims_t.data<int64_t>();
  for (int i = 0; i < dims_len; i++) {
    dims_t_d[i] = dims[i];
  }

  auto shape = self.sizes();
  Tensor shape_t = at::zeros(CPU(kLong), {shape_len});
  int64_t* shape_t_d = shape_t.data<int64_t>();
  for (int i = 0; i < shape_len; i++) {
    shape_t_d[i] = shape[i];
  }

  Tensor shape_size = at::zeros(CPU(kLong), {shape_len});
  int64_t* shape_size_d = shape_size.data<int64_t>();
  int64_t tmp = N;
  for (int i = 0; i < shape_len; i++) {
    tmp = tmp / shape[i];
    shape_size_d[i] = tmp;
  }

  Tensor nD_index = at::zeros(CUDA(kLong), {N, shape_len});

  Tensor out_t = self.clone();

  int64_t block_size = 512;
  dim3 dim_block(block_size);
  dim3 dim_grid((N + block_size - 1) / block_size);

  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), "flip_cuda", [&] {
    using cuda_scalar_t = cuda::type<scalar_t>;
    flip_cuda_kernel<<<dim_grid, dim_block, 0, globalContext().getCurrentCUDAStream()>>>(
      self.data<cuda_scalar_t>(), out_t.data<cuda_scalar_t>(), N, dims_t.toType(CUDA(kLong)).data<int64_t>(), nD_index.data<int64_t>(), dims_len, shape_size.toType(CUDA(kLong)).data<int64_t>(), shape_t.toType(CUDA(kLong)).data<int64_t>(), shape_len);
  });

  return out_t;
}

}} // namespace at::native
