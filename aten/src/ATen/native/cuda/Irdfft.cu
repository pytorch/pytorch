// Irdfft.cu - Inverse Real-to-Real FFT CUDA kernel implementation

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>
#include <string_view>
#include <string>
#include <optional>
#include <cmath>

// Define MATH_PI if not defined
#ifndef MATH_PI
#define MATH_PI M_PI
#endif

namespace at { namespace native {

  // Use static to avoid ODR violations when linking multiple .cu files
  static __device__ uint32_t reverse_bits_irdfft(uint32_t x)
  {
      x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
      x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
      x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
      x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
      return (x >> 16) | (x << 16);
  }

  // 添加：设备级三角函数包装，确保 float 使用 cosf/sinf，double 使用 cos/sin
  static __device__ __forceinline__ float device_cos(float x) { return ::cosf(x); }
  static __device__ __forceinline__ float device_sin(float x) { return ::sinf(x); }
  static __device__ __forceinline__ double device_cos(double x) { return ::cos(x); }
  static __device__ __forceinline__ double device_sin(double x) { return ::sin(x); }

  template<typename real_t>
  __global__ void ifft_inplace_kernel(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   // 第一维
    int r = blockIdx.y;  // 第二维
    int c = blockIdx.x;  // 第三维
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // int log2N = (int)log2f((real_t)N);

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // Step 1: IFFT butterfly stages (in reverse order)
    for (int s = log2N; s >= 1; --s)
    {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid; tid_global < num_work_items; tid_global += stride)
        {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0)
            {
                real_t x1 = (x[block_offset+ k + j] + x[block_offset+ k + j + L / 2]) * 0.5f;
                real_t x2 = (x[block_offset+ k + j] - x[block_offset+ k + j + L / 2]) * 0.5f;
                x[block_offset+ k + j] = x1;
                x[block_offset+ k + j + L / 2] = x2;
            }
            else if (j == L / 4)
            {
                real_t xr = x[block_offset+ k + j];
                real_t xi = x[block_offset+ k + j + L / 2];
                real_t real = (xr + xr) * 0.5;
                real_t imag = (xi - (-xi)) * 0.5;

                real_t theta = 2.0 * MATH_PI * j / L;
                real_t c = device_cos(theta);
                real_t s = device_sin(theta);

                real_t sub_real = (xr - xr) * 0.5;
                real_t sub_imag = (xi + xi) * 0.5;

                x[block_offset+ k + j] = real;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
            }
            else
            {
                real_t ar = x[block_offset+ k + j];
                real_t ai = x[block_offset+ k + L - j];
                real_t br = x[block_offset+ k + L / 2 - j];
                real_t bi = -x[block_offset+ k + L / 2 + j];

                real_t add_real = (ar + br) * 0.5;
                real_t add_imag = (ai + bi) * 0.5;
                real_t sub_real = (ar - br) * 0.5;
                real_t sub_imag = (ai - bi) * 0.5;

                real_t theta = 2.0 * MATH_PI * j / L;
                real_t c = device_cos(theta);
                real_t s = device_sin(theta);

                x[block_offset+ k + j] = add_real;
                x[block_offset+ k + L / 2 - j] = add_imag;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
                x[block_offset+ k + L - j] = c * sub_imag + s * sub_real;
            }
        }
        __syncthreads();
    }

    // Step 2: Bit-reversal permutation
    for (uint32_t i = tid; i < N; i += stride)
    {
        uint32_t j = reverse_bits_irdfft(i) >> (32 - log2N);
        if (j > i)
        {
            real_t tmp = x[block_offset+ i];
            x[block_offset+ i] = x[block_offset+ j];
            x[block_offset+ j] = tmp;
        }
    }
}

 template<typename real_t>
  __global__ void ifft_inplace_kernel_bf16(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   // 第一维
    int r = blockIdx.y;  // 第二维
    int c = blockIdx.x;  // 第三维
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // int log2N = (int)log2f((real_t)N);

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // Step 1: IFFT butterfly stages (in reverse order)
    for (int s = log2N; s >= 1; --s)
    {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid; tid_global < num_work_items; tid_global += stride)
        {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0)
            {
                float a_kj = __bfloat162float(x[block_offset+k+j]);
                float a_kj_L2 = __bfloat162float(x[block_offset+k+j+L/2]);

                float x1 = (a_kj+a_kj_L2) * 0.5f;
                float x2 = (a_kj-a_kj_L2) * 0.5f;
                x[block_offset+k+j] = __float2bfloat16(x1);
                x[block_offset+k+j+L/2] = __float2bfloat16(x2);
            }
            else if (j == L / 4)
            {
                float xr = __bfloat162float(x[block_offset+k+j]);
                float xi = __bfloat162float(x[block_offset+k+j+L/2]);
                float real = (xr+xr) * 0.5;
                float imag = (xi-(-xi)) * 0.5;

                float theta = 2.0 * MATH_PI * j/L;
                float c = device_cos(theta);
                float s = device_sin(theta);

                float sub_real = (xr-xr) * 0.5;
                float sub_imag = (xi+xi) * 0.5;

                x[block_offset+k+j] = __float2bfloat16(real);
                x[block_offset+k+j+L/2] = __float2bfloat16(c * sub_real-s * sub_imag);
            }
            else
            {
                float ar = __bfloat162float(x[block_offset+k+j]);
                float ai = __bfloat162float(x[block_offset+k+L-j]);
                float br = __bfloat162float(x[block_offset+k+L/2-j]);
                float bi = __bfloat162float(-x[block_offset+k+L/2+j]);

                float add_real = (ar+br) * 0.5;
                float add_imag = (ai+bi) * 0.5;
                float sub_real = (ar-br) * 0.5;
                float sub_imag = (ai-bi) * 0.5;

                float theta = 2.0 * MATH_PI * j / L;
                float c = device_cos(theta);
                float s = device_sin(theta);

                x[block_offset+k+j] = __float2bfloat16(add_real);
                x[block_offset+k+L/2-j] = __float2bfloat16(add_imag);
                x[block_offset+k+j+L/2] = __float2bfloat16(c * sub_real-s * sub_imag);
                x[block_offset+k+L-j] = __float2bfloat16(c * sub_imag+s * sub_real);
            }
        }
        __syncthreads();
    }

    // Step 2: Bit-reversal permutation
    for (uint32_t i = tid; i < N; i += stride)
    {
        uint32_t j = reverse_bits_irdfft(i) >> (32 - log2N);
        if (j > i)
        {
            real_t tmp = x[block_offset+ i];
            x[block_offset+ i] = x[block_offset+ j];
            x[block_offset+ j] = tmp;
        }
    }
}


void irdfft_cuda_inplace(Tensor& self, std::optional<c10::SymInt> n_opt, int64_t dim, std::optional<std::string> norm) {
    TORCH_CHECK(self.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(self.is_floating_point(), "Input must be float or double");
    TORCH_CHECK(self.dim() == 4, "Input must be 4D tensor");

    int64_t batch = self.size(0);
    int64_t r = self.size(1);
    int64_t c = self.size(2);
    int64_t k = self.size(3);
    int num_steps = static_cast<int>(std::log2(static_cast<double>(k)));

    dim3 block_wx1(1024);
    dim3 grid_wx1(c, r, batch);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (self.dtype() == at::kFloat) {
        float* d_data = self.data_ptr<float>();
        ifft_inplace_kernel<float><<<grid_wx1, block_wx1, 0, stream>>>(d_data, batch, r, c, k, num_steps);
    } else if (self.dtype() == at::kDouble) {
        double* d_data = self.data_ptr<double>();
        ifft_inplace_kernel<double><<<grid_wx1, block_wx1, 0, stream>>>(d_data, batch, r, c, k, num_steps);
    } else if (self.dtype() == at::kBFloat16) {
        __nv_bfloat16* d_data = reinterpret_cast<__nv_bfloat16*>(self.data_ptr<at::BFloat16>());
        ifft_inplace_kernel_bf16<__nv_bfloat16><<<grid_wx1, block_wx1, 0, stream>>>(d_data, batch, r, c, k, num_steps);
    }

    // optional: 同步，确保 kernel 执行完成
    // cudaDeviceSynchronize();
}


}} // namespace at::native
