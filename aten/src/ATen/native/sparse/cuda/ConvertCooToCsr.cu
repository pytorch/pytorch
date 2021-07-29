#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {

namespace {

template <typename input_t, typename output_t>
__global__ void convert_indices_from_coo_to_csr_cuda_kernel(output_t* data_out, const input_t* data_in, const int64_t size, const int64_t numel) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    for (int64_t i = 0; i <= data_in[0]; i++)
      data_out[i] = static_cast<output_t>(0);
  } else if (tid < numel) {
    for (int64_t i = data_in[tid - 1]; i < data_in[tid]; i++)
      data_out[i + 1] = static_cast<output_t>(tid);
  } else if (tid == numel) {
    for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
      data_out[i] = static_cast<output_t>(numel);
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cuda(const Tensor& result, const Tensor& input, const int64_t size) {
  int64_t numel = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel == 0) {
    result.zero_();
    return;
  }

  // Run (numel + 1) threads...
  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (numel + THREADS) / THREADS;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  convert_indices_from_coo_to_csr_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(data_out, data_in, size, numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch(const Tensor& result, const Tensor& input, const int64_t size, const bool out_int32) {
  if (!out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cuda", [&] {
      convert_indices_from_coo_to_csr_cuda<scalar_t, int64_t>(result, input, size);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cuda", [&] {
      convert_indices_from_coo_to_csr_cuda<scalar_t, int>(result, input, size);
    });
  }
}

} // namespace

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cuda) (
  const Tensor& self, const int64_t size, const bool out_int32, const Tensor& result
) {
  dispatch(result, self, size, out_int32);
}

} // namespace native
} // namespace at
