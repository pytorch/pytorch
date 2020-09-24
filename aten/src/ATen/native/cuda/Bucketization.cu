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
    int64_t mid = start + ((end - start) >> 1);
    if (!(data_ss[mid] >= val)) {
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
    int64_t mid = start + ((end - start) >> 1);
    if (!(data_ss[mid] > val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
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

  for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numel_in; tid += blockDim.x * gridDim.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right ?
      lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) - start_bd :
      upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) - start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

template<typename input_t, typename output_t>
void searchsorted_cuda_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.data_ptr<input_t>();
  const input_t *data_bd = boundaries.data_ptr<input_t>();
  output_t *data_out = result.data_ptr<output_t>();

  int64_t maxThread = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t maxGrid = 1024;
  dim3 block = dim3(std::min(maxThread, numel_in));
  dim3 grid  = dim3(std::min(maxGrid, cuda::ATenCeilDiv<int64_t>(numel_in, block.x)));
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  searchsorted_cuda_kernel<<<grid, block, 0, stream>>>(
    data_out, data_in, data_bd, idim_in, idim_bd, numel_in, right, boundaries.dim() == 1);
  THCudaCheck(cudaGetLastError());
}

void dispatch(Tensor& result, const Tensor& input, const Tensor& boundaries, bool out_int32, bool right) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "searchsorted_out_cuda", [&] {
      searchsorted_cuda_contiguous<scalar_t, int64_t>(result, input, boundaries, right);
    });
  }
  else {
    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "searchsorted_out_cuda", [&] {
      searchsorted_cuda_contiguous<scalar_t, int>(result, input, boundaries, right);
    });
  }
}

}

Tensor& searchsorted_out_cuda(Tensor& result, const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  searchsorted_pre_check(sorted_sequence, self, result, out_int32);
  if (result.numel() == 0) {
    result.resize_(self.sizes());
  }
  if (self.numel() == 0) {
    return result;
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype()) {
    dispatch(result, self, sorted_sequence, out_int32, right);
    return result;
  }

  Tensor trimmed_input;
  Tensor trimmed_boundaries;
  searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, self, sorted_sequence);
  const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
  const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
  dispatch(result, final_input, final_boundaries, out_int32, right);
  return result;
}

Tensor searchsorted_cuda(const Tensor& sorted_sequence, const Tensor& self, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  searchsorted_out_cuda(result, sorted_sequence, self, out_int32, right);
  return result;
}

Tensor searchsorted_cuda(const Tensor& sorted_sequence, Scalar self, bool out_int32, bool right) {
  return searchsorted_cuda(sorted_sequence, searchsorted_scalar_tensor(self, sorted_sequence.device()), out_int32, right);
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

Tensor bucketize_cuda(Scalar self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_cuda(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
}

}} // namespace at::native
