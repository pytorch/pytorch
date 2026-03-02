// Copyright (c) 2025 PyTorch Contributors.
// All rights reserved.

/*
 * NOTE ON SEMANTICS
 * -----------------
 * This operator implements an in-place real-domain FFT (rdFFT),
 * which is fundamentally different from torch.fft.rfft.
 *
 * - rFFT:
 *     Input : real-valued tensor
 *     Output: complex-valued tensor (Hermitian-packed, reduced size)
 *     Semantics: frequency-domain representation
 *
 * - rdFFT (this operator):
 *     Input : real-valued tensor
 *     Output: real-valued tensor (same shape as input)
 *     Semantics: real-domain transform used as an internal primitive
 *                for structured linear layers and memory-efficient
 *                training (NOT a public spectral representation).
 *
 * In particular:
 *   - No complex output is produced
 *   - No Hermitian packing is produced
 *   - The output is NOT interpretable as a frequency-domain spectrum
 *   - This operator is NOT a drop-in replacement for rfft
 *
 * This operator is only guaranteed to be invertible by the corresponding
 * in-place inverse operator (irdFFT). Using torch.fft.irfft on the output
 * of this operator is semantically invalid.
 *
 * The kernel operates entirely in the real domain and is intended as a
 * low-level internal primitive. Passing arbitrary real tensors or treating
 * the output as a spectral representation results in undefined semantics.
 */


#include "Rdfft.h"
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
__global__ void fft_inplace_kernel(
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

  for (uint32_t i = tid; i < N; i += stride) {
    uint32_t j = rdfft_utils::reverse_bits_32(i) >> (32 - log2N);
    if (j > i) {
      real_t tmp = x[block_offset+i];
      x[block_offset+i] = x[block_offset+j];
      x[block_offset+j] = tmp;
    }
  }
  __syncthreads();

  for (int s = 1; s <= log2N; ++s) {
    int L = 1 << s;
    int num_groups = N / L;
    int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
    int total_work_items = num_groups * num_j;

    for (int idx = tid; idx < total_work_items; idx += stride) {
      int group_id = idx / num_j;
      int j = idx % num_j;
      int k = group_id * L;

      if (j == 0) {
        real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
        real_t t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
        x[block_offset+k+j] = t1;
        x[block_offset+k+j+L/2] = t2;
      } else {
        real_t angle1 = -2 * M_PI * j / L;
        real_t angle2 = -1 * M_PI * (L-2*j) / L;
        real_t t1 = (j == L/4)
          ? x[block_offset+k+j]+x[block_offset+k+j+L/2]*rdfft_utils::device_cos(angle1)
          : x[block_offset+k+j]+x[block_offset+k+j+L/2]*rdfft_utils::device_cos(angle1)
            -x[block_offset+k+L-j]*rdfft_utils::device_sin(angle1);
        real_t t2 = (j == L/4)
          ? 0
          : x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*rdfft_utils::device_sin(angle1)
            +x[block_offset+k+L-j]*rdfft_utils::device_cos(angle1);
        real_t t3 = (j == L/4)
          ? 0
          : x[block_offset+k+j]+x[block_offset+k+j+L/2]*rdfft_utils::device_cos(angle2)
            +x[block_offset+k+L-j]*rdfft_utils::device_sin(angle2);
        real_t t4 = (j == L/4)
          ? x[block_offset+k+j+L/2]*rdfft_utils::device_sin(angle1)
          : -x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*rdfft_utils::device_sin(angle2)
            -x[block_offset+k+L-j]*rdfft_utils::device_cos(angle2);

        x[block_offset+k+j] = t1;
        x[block_offset+k+L/2-j] = (t3!= 0) ? t3 : x[block_offset+k+L/2-j];
        x[block_offset+k+j+L/2] = t4;
        x[block_offset+k+L-j] = (t2!= 0) ? t2 : x[block_offset+k+L-j];
      }
    }
    __syncthreads();
  }
}


__global__ void fft_inplace_kernel_bf16(__nv_bfloat16 *x,
int _, int rows, int cols, int N, int log2N) {
  int b = blockIdx.z;
  int r = blockIdx.y;
  int c = blockIdx.x;

  int tid = threadIdx.x;
  int stride = blockDim.x;

  if (b >= _ || r >= rows || c >= cols)
    return;
  int block_offset = b * rows * cols * N + r * cols * N + c * N;

  for (uint32_t i = tid; i < N; i += stride) {
    uint32_t j = rdfft_utils::reverse_bits_32(i) >> (32 - log2N);
    if (j > i) {
      __nv_bfloat16 tmp = x[block_offset+i];
      x[block_offset+i] = x[block_offset+j];
      x[block_offset+j] = tmp;
    }
  }
  __syncthreads();

  for (int s = 1; s <= log2N; ++s) {
    int L = 1 << s;
    int num_groups = N / L;
    int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
    int total_work_items = num_groups * num_j;

    for (int idx = tid; idx < total_work_items; idx += stride) {
      int group_id = idx / num_j;
      int j = idx % num_j;
      int k = group_id * L;

      if (j == 0) {
        __nv_bfloat16 t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
        __nv_bfloat16 t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
        x[block_offset+k+j] = t1;
        x[block_offset+k+j+L/2] = t2;
      } else {
        float angle1 = -2 * M_PI * j / L;
        float angle2 = -1 * M_PI * (L-2*j) / L;

        float a_kj = __bfloat162float(x[block_offset+k+j]);
        float a_kj_L2 = __bfloat162float(x[block_offset+k+j+L/2]);
        float a_kL_j = __bfloat162float(x[block_offset+k+L-j]);
        float a_kL2_j = __bfloat162float(x[block_offset+k+L/2-j]);

        float t1 = (j == L/4)
          ? a_kj+a_kj_L2*rdfft_utils::device_cos(angle1)
          : a_kj+a_kj_L2*rdfft_utils::device_cos(angle1)
            -a_kL_j*rdfft_utils::device_sin(angle1);
        float t2 = (j == L/4)
          ? 0
          : a_kL2_j+a_kj_L2*rdfft_utils::device_sin(angle1)
            +a_kL_j*rdfft_utils::device_cos(angle1);
        float t3 = (j == L/4)
          ? 0
          : a_kj+a_kj_L2*rdfft_utils::device_cos(angle2)
            +a_kL_j*rdfft_utils::device_sin(angle2);
        float t4 = (j == L/4)
          ? a_kj_L2*rdfft_utils::device_sin(angle1)
          : -a_kL2_j+a_kj_L2*rdfft_utils::device_sin(angle2)
            -a_kL_j*rdfft_utils::device_cos(angle2);

        x[block_offset+k+j] =  __float2bfloat16(t1);
        x[block_offset+k+L/2-j] =  (t3!= 0)
          ? __float2bfloat16(t3) : x[block_offset+k+L/2-j];
        x[block_offset+k+j+L/2] =  __float2bfloat16(t4);
        x[block_offset+k+L-j] = (t2!= 0)
          ?  __float2bfloat16(t2) : x[block_offset+k+L-j];
      }
    }
    __syncthreads();
  }
}

void rdfft_cuda_inplace(Tensor& self,
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
      fft_inplace_kernel<float><<<
        grid_wx1, block_wx1, 0, stream>>>(
          d_data, batch, r, c, k, num_steps);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (self.dtype() == at::kDouble) {
      double* d_data = self.data_ptr<double>();
      fft_inplace_kernel<double><<<
        grid_wx1, block_wx1, 0, stream>>>(
          d_data, batch, r, c, k, num_steps);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (self.dtype() == at::kBFloat16) {
      __nv_bfloat16* d_data = reinterpret_cast<__nv_bfloat16*>(
        self.data_ptr<at::BFloat16>());
      fft_inplace_kernel_bf16<<<
        grid_wx1, block_wx1, 0, stream>>>(
          d_data, batch, r, c, k, num_steps);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      TORCH_CHECK(false, "rdfft_: unsupported dtype");
    }

}

}  // namespace native
}  // namespace at
