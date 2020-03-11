#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/BucketizationUtils.h>
#include <THC/THC.h>

namespace at {
namespace native {

// Implement a TF like searchsorted and a bucketize function running on cuda
// See details in ATen/nativate/Bucketization.cpp

namespace {

template<typename input_t>
__device__ int64_t lower_bound(const input_t *data_ss, int64_t start, int64_t end, input_t val) {
  while (start < end) {
    int64_t mid = start + (end - start) / 2;
    if (data_ss[mid] < val) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
__device__ int64_t upper_bound(const input_t *data_ss, int64_t start, int64_t end, input_t val) {
  while (start < end) {
    int64_t mid = start + (end - start) / 2;
    if (data_ss[mid] <= val) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
__device__ void searchsorted_cuda_calc_boundary(
  output_t *data_out,
  const input_t *data_in,
  const input_t *data_bd,
  int64_t idim_in,
  int64_t idim_bd,
  int64_t numel_in,
  bool right,
  bool is_1d_boundaries) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numel_in) {
    return;
  }

  // If boundaries tensor is 1d, we always search the entire boundary tensor
  int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
  int64_t end_bd = start_bd + idim_bd;

  int64_t pos = !right ?
    lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) - start_bd :
    upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) - start_bd;

  // type conversion might happen here
  data_out[tid] = pos;
}

template<typename input_t, typename output_t>
__global__ void searchsorted_cuda_kernel(
  output_t *data_out,
  const input_t *data_in,
  const input_t *data_bd,
  int64_t idim_in,
  int64_t idim_bd,
  int64_t numel_in,
  bool right,
  bool is_1d_boundaries) {

  searchsorted_cuda_calc_boundary<input_t, output_t>(
    data_out, data_in, data_bd, idim_in, idim_bd, numel_in, right, is_1d_boundaries);
}

template<typename input_t, typename output_t>
__global__ void searchsorted_cuda_shared_boundaries_kernel(
  output_t *data_out,
  const input_t *data_in,
  const input_t *data_bd,
  int64_t idim_in,
  int64_t idim_bd,
  int64_t numel_in,
  int64_t numel_bd,
  int64_t radius, // how many items to copy into shared memory per thread 
  bool right,
  bool is_1d_boundaries) {

  extern __shared__ unsigned char temp[];
  input_t *data_bd_shared = reinterpret_cast<input_t *>(temp);
 
  int64_t start = threadIdx.x * radius;
  int64_t end = start + radius > numel_bd ? numel_bd : start + radius;
  if (start < end) {
    #pragma unroll
    for (int64_t i = start; i < end; ++i) {
      data_bd_shared[i] = data_bd[i];
    }
  }
  __syncthreads();
 
  searchsorted_cuda_calc_boundary<input_t, output_t>(
    data_out, data_in, data_bd_shared, idim_in, idim_bd, numel_in, right, is_1d_boundaries);
}

template<typename input_t, typename output_t>
void searchsorted_cuda_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right) {
  // inner most dim size of input and boundaries
  int64_t idim_in = input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.data_ptr<input_t>();
  const input_t *data_bd = boundaries.data_ptr<input_t>();

  output_t *data_out = result.data_ptr<output_t>();
  int64_t numel_in = input.numel();

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  dim3 block = dim3(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock);
  dim3 grid  = dim3(cuda::ATenCeilDiv<int64_t>(numel_in, block.x));

  int64_t numel_bd = boundaries.numel();
  // this is a simple optimize strategy mainly aim to optimize 1D boundaries case, open to discuss.
  // older cuda hardware (compute capacity 1.x) uses 256 bytes shared memory to pass kernel function parameters, exclude it. 
  // to get better performance, make sure at least 4 different blocks runnig parallel within 1 SM.
  int64_t usable_shared_mem_per_block = (at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock - 256) / 4; 
  int64_t boundaries_tensor_size_byte = numel_bd * sizeof(input_t);

  if (boundaries_tensor_size_byte <= usable_shared_mem_per_block) {
    int64_t radius = numel_bd / block.x + 1;
    searchsorted_cuda_shared_boundaries_kernel<<<grid, block, boundaries_tensor_size_byte, stream>>>(
      data_out, data_in, data_bd, idim_in, idim_bd, numel_in, numel_bd, radius, right, boundaries.dim() == 1);
  }
  else {
    searchsorted_cuda_kernel<<<grid, block, 0, stream>>>(
      data_out, data_in, data_bd, idim_in, idim_bd, numel_in, right, boundaries.dim() == 1);
  }
  THCudaCheck(cudaGetLastError());
}

}

Tensor& searchsorted_out_cuda(Tensor& result, const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  searchsorted_pre_check(sorted_sequence, self, out_int32);
  if (result.numel() == 0) {
    result.resize_(self.sizes());
  }
  if (self.numel() == 0) {
    return result;
  }

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "searchsorted_out_cuda", [&] {
    if (out_int32) {
      searchsorted_generic_template(result, self, sorted_sequence, right, searchsorted_cuda_contiguous<scalar_t, int>);
    }
    else {
      searchsorted_generic_template(result, self, sorted_sequence, right, searchsorted_cuda_contiguous<scalar_t, int64_t>);
    }
  });
  return result;
}

Tensor searchsorted_cuda(const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  searchsorted_out_cuda(result, sorted_sequence, self, out_int32, right);
  return result;
}

Tensor& bucketize_out_cuda(Tensor& result, const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  searchsorted_out_cuda(result, boundaries, self, out_int32, right);
  return result;
}

Tensor bucketize_cuda(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  bucketize_out_cuda(result, self, boundaries, out_int32, right);
  return result;
}

}} // namespace at::native

