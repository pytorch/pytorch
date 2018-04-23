#include "ATen/NativeFunctions.h"
#include <algorithm>
#include <sstream>

namespace at {
namespace native {

Tensor& eye_out_cuda(Tensor& result, int64_t n, int64_t m) {
  if (n <= 0) {
    std::ostringstream oss;
    oss << "n must be greater than 0, got: " << n;
    std::runtime_error(oss.str());
  }
  if(m <= 0) {
    m = n;
  }

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

__device__ void oneD_to_nD(int64_t oneD_index, int64_t* shape_size, int64_t shape_len, int64_t* nD_index) {
  int64_t res = oneD_index;
  for (int i = 0; i < shape_len; i++) {
    int64_t nD_i = oneD_index * shape_len + i;
    nD_index[nD_i] = res / shape_size[i];
    res = res % shape_size[i];
  }
}

__device__ int64_t nD_to_oneD(int64_t* nD_index, int64_t shape_len, int64_t* shape_size, int64_t src_oneD_index) {
  int64_t dest_oneD_index = 0;
  for (int i = 0; i < shape_len; i++) {
    int64_t nD_i = src_oneD_index * shape_len + i;
    dest_oneD_index += nD_index[nD_i] * shape_size[i];
  }
  return dest_oneD_index;
}

__global__ void flip_cuda_kernel(double* in_t, double* out_t, int64_t N, int64_t* dims, int64_t* nD_index,
  int64_t dims_len, int64_t* shape_size, int64_t* shape, int64_t shape_len) {

  int64_t oneD_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (oneD_index >= N) {
    return;
  }

  oneD_to_nD(oneD_index, shape_size, shape_len, nD_index);
  for (int i = 0 ; i < dims_len; i++) {
    int64_t d = dims[i];
    int64_t nD_d = oneD_index * shape_len + d;
    nD_index[nD_d] = shape[d]-1-nD_index[nD_d];
  }
  int64_t dest_oneD_index = nD_to_oneD(nD_index, shape_len, shape_size, oneD_index);
  out_t[oneD_index] = in_t[dest_oneD_index];
}

Tensor flip_cuda(const Tensor& self, IntList dims) {
  // check if number of axis in dim is valid
  if (dims.size() == 0) {
    std::stringstream ss;
    ss << "CUDA: expected dims not empty, "
       << "but got dims size=" << dims.size();
    throw std::runtime_error(ss.str());
  }

  // remove duplicates in dims
  auto dims_v = std::vector<int64_t>(dims);
  dims_v.erase(std::unique(dims_v.begin(), dims_v.end()), dims_v.end());
  dims = IntList(dims_v);

  int64_t dims_len = dims.size(), shape_len = self.dim(), N = self.numel();

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

  int64_t* d_dims_t;
  cudaMalloc(&d_dims_t, dims_len * sizeof(int64_t));
  cudaMemcpy(d_dims_t, dims_t.data<int64_t>(), dims_len * sizeof(int64_t), cudaMemcpyHostToDevice);

  Tensor shape = at::zeros(CPU(kLong), {shape_len});
  int64_t* shape_d = shape.data<int64_t>();
  for (int i = 0; i < shape_len; i++) {
    shape_d[i] = self.size(i);
  }

  int64_t* d_shape;
  cudaMalloc(&d_shape, shape_len * sizeof(int64_t));
  cudaMemcpy(d_shape, shape.data<int64_t>(), shape_len * sizeof(int64_t), cudaMemcpyHostToDevice);

  Tensor shape_size = at::zeros(CPU(kLong), {shape_len});
  int64_t* shape_size_d = shape_size.data<int64_t>();
  int64_t tmp = N;
  for (int i = 0; i < shape_len; i++) {
    tmp = tmp / shape_d[i];
    shape_size_d[i] = tmp;
  }

  int64_t* d_shape_size;
  cudaMalloc(&d_shape_size, shape_len * sizeof(int64_t));
  cudaMemcpy(d_shape_size, shape_size.data<int64_t>(), shape_len * sizeof(int64_t), cudaMemcpyHostToDevice);

  Tensor nD_index = at::zeros(CPU(kLong), {N, shape_len});
  int64_t* d_nD_index;
  cudaMalloc(&d_nD_index, N * shape_len * sizeof(int64_t));
  cudaMemcpy(d_nD_index, nD_index.data<int64_t>(), N * shape_len * sizeof(int64_t), cudaMemcpyHostToDevice);

  double* d_in_t;
  cudaMalloc(&d_in_t, N * sizeof(double));
  cudaMemcpy(d_in_t, self.data<double>(), N * sizeof(double), cudaMemcpyHostToDevice);

  Tensor out_t = self.clone();
  double* d_out_t;
  cudaMalloc(&d_out_t, N * sizeof(double));
  cudaMemcpy(d_out_t, out_t.data<double>(), N * sizeof(double), cudaMemcpyHostToDevice);

  int64_t block_size = 512;
  dim3 dim_block(block_size);
  dim3 dim_grid((N + block_size - 1) / block_size);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "flip_cuda", [&] {
    flip_cuda_kernel<<<dim_grid, dim_block, 0, globalContext().getCurrentCUDAStream()>>>(
      d_in_t, d_out_t, N, d_dims_t, d_nD_index, dims_len, d_shape_size, d_shape, shape_len);
  });

  cudaMemcpy(out_t.data<double>(), d_out_t, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_in_t);
  cudaFree(d_out_t);
  cudaFree(d_dims_t);
  cudaFree(d_nD_index);
  cudaFree(d_shape_size);
  cudaFree(d_shape);

  return out_t;
}

Tensor flip_backward_cuda(const Tensor& grad, IntList dims) {
  return flip_cuda(grad, dims);
}

}} // namespace at::native
