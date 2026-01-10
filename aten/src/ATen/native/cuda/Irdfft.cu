// Copyright (c) 2025 PyTorch Contributors.
// All rights reserved.

/*
 * NOTE ON SEMANTICS
 * -----------------
 * This operator implements an in-place inverse real-domain FFT (irdFFT),
 * which is fundamentally different from torch.fft.irfft.
 *
 * - irFFT:
 *     Input : complex-valued tensor (Hermitian-packed spectrum)
 *     Output: real-valued tensor
 *     Semantics: inverse frequency-domain transform
 *
 * - irdFFT (this operator):
 *     Input : real-valued tensor (output of rdFFT)
 *     Output: real-valued tensor (same shape as input)
 *     Semantics: inverse real-domain transform used as an internal primitive
 *                for structured linear layers / parameter-efficient
 *                training (NOT a public spectral representation).
 *
 * In particular:
 *   - No Hermitian-packed complex spectrum is consumed
 *   - No complex input or output is involved
 *   - This operator is NOT a drop-in replacement for irfft
 *
 * The kernel operates entirely in the real domain and is only guaranteed
 * to be the inverse of the corresponding rdFFT operator.
 *
 * The input is NOT interpretable as a frequency-domain representation,
 * and passing arbitrary real tensors to this operator results in
 * undefined semantics.
 *
 * This operator is intended as a low-level primitive for memory-efficient
 * training and structured linear transformations, NOT as a public inverse
 * FFT API.
 */

#include "Irdfft.h"
#include "RdfftUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/SymInt.h>
#include <c10/util/Optional.h>
#include <cmath>
#include <string>
#include <string_view>
#include <optional>

namespace at {
namespace native {

template<typename real_t>
__global__ void ifft_inplace_kernel(
real_t *x, int _, int rows, int cols,
int N, int log2N) {
    int b = blockIdx.z;
    int r = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    for (int s = log2N; s >= 1; --s) {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid;
            tid_global < num_work_items;
            tid_global += stride) {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0) {
                real_t x1 = (x[block_offset+ k + j]
                    + x[block_offset+ k + j + L / 2]) * 0.5f;
                real_t x2 = (x[block_offset+ k + j]
                    - x[block_offset+ k + j + L / 2]) * 0.5f;
                x[block_offset+ k + j] = x1;
                x[block_offset+ k + j + L / 2] = x2;
            } else if (j == L / 4) {
                real_t xr = x[block_offset+ k + j];
                real_t xi = x[block_offset+ k + j + L / 2];
                real_t real = (xr + xr) * 0.5;
                real_t imag = (xi - (-xi)) * 0.5;

                real_t theta = 2.0 * M_PI * j / L;
                real_t c = rdfft_utils::device_cos(theta);
                real_t s = rdfft_utils::device_sin(theta);

                real_t sub_real = (xr - xr) * 0.5;
                real_t sub_imag = (xi + xi) * 0.5;

                x[block_offset+ k + j] = real;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
            } else {
                real_t ar = x[block_offset+ k + j];
                real_t ai = x[block_offset+ k + L - j];
                real_t br = x[block_offset+ k + L / 2 - j];
                real_t bi = -x[block_offset+ k + L / 2 + j];

                real_t add_real = (ar + br) * 0.5;
                real_t add_imag = (ai + bi) * 0.5;
                real_t sub_real = (ar - br) * 0.5;
                real_t sub_imag = (ai - bi) * 0.5;

                real_t theta = 2.0 * M_PI * j / L;
                real_t c = rdfft_utils::device_cos(theta);
                real_t s = rdfft_utils::device_sin(theta);

                x[block_offset+ k + j] = add_real;
                x[block_offset+ k + L / 2 - j] = add_imag;
                x[block_offset+ k + j + L / 2] = c * sub_real - s * sub_imag;
                x[block_offset+ k + L - j] = c * sub_imag + s * sub_real;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < N; i += stride) {
        uint32_t j = rdfft_utils::reverse_bits_32(i) >> (32 - log2N);
        if (j > i) {
            real_t tmp = x[block_offset+ i];
            x[block_offset+ i] = x[block_offset+ j];
            x[block_offset+ j] = tmp;
        }
    }
}

__global__ void ifft_inplace_kernel_bf16(
__nv_bfloat16 *x, int _, int rows, int cols,
int N, int log2N) {
    int b = blockIdx.z;
    int r = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    for (int s = log2N; s >= 1; --s) {
        int L = 1 << s;
        int num_groups = N / L;
        int num_work_items = num_groups * (L / 4 + 1);

        for (int tid_global = tid;
            tid_global < num_work_items;
            tid_global += stride) {
            int group_id = tid_global / (L / 4 + 1);
            int j = tid_global % (L / 4 + 1);
            int k = group_id * L;

            if (j == 0) {
                float a_kj = __bfloat162float(x[block_offset+k+j]);
                float a_kj_L2 = __bfloat162float(x[block_offset+k+j+L/2]);

                float x1 = (a_kj+a_kj_L2) * 0.5f;
                float x2 = (a_kj-a_kj_L2) * 0.5f;
                x[block_offset+k+j] = __float2bfloat16(x1);
                x[block_offset+k+j+L/2] = __float2bfloat16(x2);
            } else if (j == L / 4) {
                float xr = __bfloat162float(x[block_offset+k+j]);
                float xi = __bfloat162float(x[block_offset+k+j+L/2]);
                float real = (xr+xr) * 0.5;
                float imag = (xi-(-xi)) * 0.5;

                float theta = 2.0 * M_PI * j/L;
                float c = rdfft_utils::device_cos(theta);
                float s = rdfft_utils::device_sin(theta);

                float sub_real = (xr-xr) * 0.5;
                float sub_imag = (xi+xi) * 0.5;

                x[block_offset+k+j] = __float2bfloat16(real);
                x[block_offset+k+j+L/2] = __float2bfloat16(
                    c * sub_real-s * sub_imag);
            } else {
                float ar = __bfloat162float(x[block_offset+k+j]);
                float ai = __bfloat162float(x[block_offset+k+L-j]);
                float br = __bfloat162float(x[block_offset+k+L/2-j]);
                float bi = __bfloat162float(-x[block_offset+k+L/2+j]);

                float add_real = (ar+br) * 0.5;
                float add_imag = (ai+bi) * 0.5;
                float sub_real = (ar-br) * 0.5;
                float sub_imag = (ai-bi) * 0.5;

                float theta = 2.0 * M_PI * j / L;
                float c = rdfft_utils::device_cos(theta);
                float s = rdfft_utils::device_sin(theta);

                x[block_offset+k+j] = __float2bfloat16(add_real);
                x[block_offset+k+L/2-j] = __float2bfloat16(add_imag);
                x[block_offset+k+j+L/2] = __float2bfloat16(
                    c * sub_real-s * sub_imag);
                x[block_offset+k+L-j] = __float2bfloat16(
                    c * sub_imag+s * sub_real);
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < N; i += stride) {
        uint32_t j = rdfft_utils::reverse_bits_32(i) >> (32 - log2N);
        if (j > i) {
            __nv_bfloat16 tmp = x[block_offset+ i];
            x[block_offset+ i] = x[block_offset+ j];
            x[block_offset+ j] = tmp;
        }
    }
}


void irdfft_cuda_inplace(Tensor& self,
    std::optional<c10::SymInt> n_opt,
    int64_t dim, std::optional<std::string> norm) {
    TORCH_CHECK(self.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(self.is_floating_point(), "Input must be float or double");
    TORCH_CHECK(self.dim() == 4, "Input must be 4D tensor");

    int64_t batch = self.size(0);
    int64_t r = self.size(1);
    int64_t c = self.size(2);
    int64_t k = self.size(3);
    
    TORCH_CHECK((k & (k - 1)) == 0,
              "Last dimension must be power of 2");

    int num_steps = static_cast<int>(std::log2(static_cast<double>(k)));

    dim3 block_wx1(1024);
    dim3 grid_wx1(c, r, batch);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (self.dtype() == at::kFloat) {
        float* d_data = self.data_ptr<float>();
        ifft_inplace_kernel<float><<<
            grid_wx1, block_wx1, 0, stream>>>(
                d_data, batch, r, c, k, num_steps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (self.dtype() == at::kDouble) {
        double* d_data = self.data_ptr<double>();
        ifft_inplace_kernel<double><<<
            grid_wx1, block_wx1, 0, stream>>>(
                d_data, batch, r, c, k, num_steps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (self.dtype() == at::kBFloat16) {
        __nv_bfloat16* d_data = reinterpret_cast<__nv_bfloat16*>(
            self.data_ptr<at::BFloat16>());
        ifft_inplace_kernel_bf16<<<
            grid_wx1, block_wx1, 0, stream>>>(
                d_data, batch, r, c, k, num_steps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(false, "rdfft_: unsupported dtype");
    }

}

}  // namespace native
}  // namespace at
