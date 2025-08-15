#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace at::native {

__global__ void weight_int8pack_mm_kernel(const float* x, const int8_t* w, const float* scale, float* out, int B, int K, int N) {
  // one thread per output element: [B, N]
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= B || n >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += x[b * K + k] * static_cast<float>(w[n * K + k]);
  }

  out[b * N + n] = acc * scale[n];
}

void launch_weight_int8pack_mm_cuda_kernel(const Tensor& x, const Tensor& w_int8, const Tensor& scale, Tensor& out) {
  const int B = x.size(0);
  const int K = x.size(1);
  const int N = w_int8.size(0);

  const dim3 block(16, 16);
  const dim3 grid((N + block.x - 1) / block.x, (B + block.y - 1) / block.y);

  auto stream = at::cuda::getCurrentCUDAStream();

  weight_int8pack_mm_kernel<<<grid, block, 0, stream>>>(
      x.data_ptr<float>(),
      w_int8.data_ptr<int8_t>(),
      scale.data_ptr<float>(),
      out.data_ptr<float>(),
      B, K, N);
}


// Main GPU entry point
at::Tensor _weight_int8pack_mm_cuda(const at::Tensor& x, const at::Tensor& w_int8, const at::Tensor& scale) {
  // --- Check inputs ---
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w_int8.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(w_int8.dim() == 2, "w must be 2D");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D");

  TORCH_CHECK(x.size(1) == w_int8.size(1), "K dimension mismatch: x.size(1) != w.size(1)");
  TORCH_CHECK(w_int8.size(0) == scale.size(0), "Output dim mismatch: w.size(0) != scale.size(0)");

  // --- Determine shapes ---
  auto B = x.size(0);  // batch size
  auto N = w_int8.size(0);  // output dim

  // Ensure inputs are in the correct types for the kernel
  auto x_f32 = x.to(at::kFloat);
  auto w_int8_contiguous = w_int8.contiguous();
  auto scale_f32 = scale.to(at::kFloat);

  // --- Allocate output ---
  auto out = at::empty({B, N}, x.options().dtype(at::kFloat));

  // --- Launch kernel ---
  launch_weight_int8pack_mm_cuda_kernel(x_f32, w_int8_contiguous, scale_f32, out);

  return out;
}

} // namespace at::native
