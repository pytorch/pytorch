#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {

namespace {

template <typename input_t, typename output_t>
__global__ void convert_coo_to_csr_cuda_kernel(
    output_t* data_out,
    const input_t* data_in,
    int64_t size,
    int64_t numel_in) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid == 0) {
    for (int64_t i = 0; i <= data_in[0]; i++)
      data_out[i] = (output_t)0;
  } else if (tid < numel) {
    for (int64_t i = data_in[tid - 1]; i < data_in[tid]; i++)
      data_out[i + 1] = (output_t)tid;
  } else if (tid == numel) {
    for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
      data_out[i] = numel;
  }
}

template <typename input_t, typename output_t>
void convert_coo_to_csr_cuda(
    Tensor& result,
    const Tensor& input,
    const Scalar& size) {
  int64_t numel_in = input.numel();
  const input_t* data_in = input.data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel_in == 0) {
    result.zero_();
    return;
  }

  int64_t THREADS = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (numel + THREADS) / THREADS;

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  convert_coo_to_csr_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      data_out, data_in, size, numel_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch(
    Tensor& result,
    const Tensor& input,
    const Scalar& size,
    bool out_int32) {
  if (!out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_coo_to_csr_cuda", [&] {
          convert_coo_to_csr_cuda<scalar_t, int64_t>(result, input, size);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_coo_to_csr_cuda", [&] {
          convert_coo_to_csr_cuda<scalar_t, int>(result, input, size);
        });
  }
}

} // namespace

Tensor& _convert_coo_to_csr_out_cuda(
    const Tensor& self,
    const Scalar& size,
    bool out_int32,
    Tensor& result) {
  dispatch(result, self, size, out_int32);
  return result;
}

Tensor _convert_coo_to_csr_cuda(
    const Tensor& self,
    const Scalar& size,
    bool out_int32) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({size + 1}, options, MemoryFormat::Contiguous);
  at::native::_convert_coo_to_csr_out_cuda(self, size, out_int32, result);
  return result;
}

} // namespace native
} // namespace at
