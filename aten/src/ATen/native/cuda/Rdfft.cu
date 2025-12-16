// Rdfft.cu - Real-to-Real FFT CUDA kernel implementation

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
#include <cuda_bf16.h>

namespace at { namespace native {

  // Use static to avoid ODR violations when linking multiple .cu files
  static __device__ uint32_t reverse_bits_rdfft(uint32_t x)
  {
      x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
      x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
      x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
      x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
      return (x >> 16) | (x << 16);
  }

  // 添加：设备级三角函数包装，确保 float 使用 cosf/sinf，double 使用 cos/sin
  static __device__ __forceinline__ float device_cos(float x)  { return ::cosf(x); }
  static __device__ __forceinline__ float device_sin(float x)  { return ::sinf(x); }
  static __device__ __forceinline__ double device_cos(double x){ return ::cos(x);  }
  static __device__ __forceinline__ double device_sin(double x){ return ::sin(x);  }

  template<typename real_t>
  __global__ void fft_inplace_kernel(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   // 第一维
    int r = blockIdx.y;  // 第二维
    int c = blockIdx.x;  // 第三维
    // int i = threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // int tid = threadIdx.x;
    // int stride = blockDim.x;
    // int log2N = (int)log2f((real_t)N);
    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // --- Step 1: Bit-reversal permutation ---
    for (uint32_t i = tid; i < N; i += stride)
    {
      uint32_t j = reverse_bits_rdfft(i) >> (32 - log2N);
      if (j > i)
      {
        real_t tmp = x[block_offset+i];
        x[block_offset+i] = x[block_offset+j];
        x[block_offset+j] = tmp;
      }
    }
    __syncthreads();

    // --- Step 2: FFT computation ---
    for (int s = 1; s <= log2N; ++s)
    {
      int L = 1 << s;
      int num_groups = N / L;
      int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
      int total_work_items = num_groups * num_j;

      for (int idx = tid; idx < total_work_items; idx += stride)
      {
        int group_id = idx / num_j;
        int j = idx % num_j;
        int k = group_id * L;

      // if (L > 4)
      // if(j <= L/4){
        if (j == 0){
          real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
          real_t t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
          x[block_offset+k+j] = t1;
          x[block_offset+k+j+L/2] = t2;
        // } else if (j == L/4){
        //   real_t angle1 = -2 * M_PI * j / L;
        //   real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2]*cosf(angle1);
        //   real_t t2 = x[block_offset+k+j+L/2]*sinf(angle1);
        //   x[block_offset+k+j] = t1;
        //   x[block_offset+k+j+L/2] = t2;
        } else{
          real_t angle1 = -2 * M_PI * j / L;
          real_t angle2 = -1 * M_PI * (L-2*j) / L;
          real_t t1 = (j == L/4) ? x[block_offset+k+j]+x[block_offset+k+j+L/2]*device_cos(angle1)
                                 : x[block_offset+k+j]+x[block_offset+k+j+L/2]*device_cos(angle1)-x[block_offset+k+L-j]*device_sin(angle1);
          real_t t2 = (j == L/4) ? 0
                                 : x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*device_sin(angle1)+x[block_offset+k+L-j]*device_cos(angle1);
          real_t t3 = (j == L/4) ? 0
                                 : x[block_offset+k+j]+x[block_offset+k+j+L/2]*device_cos(angle2)+x[block_offset+k+L-j]*device_sin(angle2);
          real_t t4 = (j == L/4) ? x[block_offset+k+j+L/2]*device_sin(angle1)
                                 : -x[block_offset+k+L/2-j]+x[block_offset+k+j+L/2]*device_sin(angle2)-x[block_offset+k+L-j]*device_cos(angle2);

          x[block_offset+k+j] = t1;
          x[block_offset+k+L/2-j] = (t3!= 0) ? t3 : x[block_offset+k+L/2-j]; 
          x[block_offset+k+j+L/2] = t4; 
          x[block_offset+k+L-j] = (t2!= 0) ? t2 : x[block_offset+k+L-j];  
        }
     }
      __syncthreads(); // sync between stages
   }
    
   }


  template<typename real_t>
  __global__ void fft_inplace_kernel_bf16(real_t *x, int _, int rows, int cols, int N, int log2N) 
  {
    int b = blockIdx.z;   // 第一维
    int r = blockIdx.y;  // 第二维
    int c = blockIdx.x;  // 第三维
    // int i = threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // int tid = threadIdx.x;
    // int stride = blockDim.x;
    // int log2N = (int)log2f((real_t)N);
    if (b >= _ || r >= rows || c >= cols)
        return;
    int block_offset = b * rows * cols * N + r * cols * N + c * N;

    // --- Step 1: Bit-reversal permutation ---
    for (uint32_t i = tid; i < N; i += stride)
    {
      uint32_t j = reverse_bits_rdfft(i) >> (32 - log2N);
      if (j > i)
      {
        real_t tmp = x[block_offset+i];
        x[block_offset+i] = x[block_offset+j];
        x[block_offset+j] = tmp;
      }
    }
    __syncthreads();

    // --- Step 2: FFT computation ---
    for (int s = 1; s <= log2N; ++s)
    {
      int L = 1 << s;
      int num_groups = N / L;
      int num_j = (L > 4) ? (L / 4 + 1) : (L / 2);
      int total_work_items = num_groups * num_j;

      for (int idx = tid; idx < total_work_items; idx += stride)
      {
        int group_id = idx / num_j;
        int j = idx % num_j;
        int k = group_id * L;

      // if (L > 4)
      // if(j <= L/4){
        if (j == 0){
          real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2];
          real_t t2 = x[block_offset+k+j]-x[block_offset+k+j+L/2];
          x[block_offset+k+j] = t1;
          x[block_offset+k+j+L/2] = t2;
        // } else if (j == L/4){
        //   real_t angle1 = -2 * M_PI * j / L;
        //   real_t t1 = x[block_offset+k+j]+x[block_offset+k+j+L/2]*cosf(angle1);
        //   real_t t2 = x[block_offset+k+j+L/2]*sinf(angle1);
        //   x[block_offset+k+j] = t1;
        //   x[block_offset+k+j+L/2] = t2;
        } else{
          float angle1 = -2 * M_PI * j / L;
          float angle2 = -1 * M_PI * (L-2*j) / L;

          float a_kj = __bfloat162float(x[block_offset+k+j]);
          float a_kj_L2 = __bfloat162float(x[block_offset+k+j+L/2]);
          float a_kL_j = __bfloat162float(x[block_offset+k+L-j]);
          float a_kL2_j = __bfloat162float(x[block_offset+k+L/2-j]);

          float t1 = (j == L/4) ? a_kj+a_kj_L2*device_cos(angle1)
                                 : a_kj+a_kj_L2*device_cos(angle1)-a_kL_j*device_sin(angle1);
          float t2 = (j == L/4) ? 0
                                 : a_kL2_j+a_kj_L2*device_sin(angle1)+a_kL_j*device_cos(angle1);
          float t3 = (j == L/4) ? 0
                                 : a_kj+a_kj_L2*device_cos(angle2)+a_kL_j*device_sin(angle2);
          float t4 = (j == L/4) ? a_kj_L2*device_sin(angle1)
                                 : -a_kL2_j+a_kj_L2*device_sin(angle2)-a_kL_j*device_cos(angle2);

          x[block_offset+k+j] =  __float2bfloat16(t1);
          x[block_offset+k+L/2-j] =  (t3!= 0) ? __float2bfloat16(t3) : x[block_offset+k+L/2-j]; 
          x[block_offset+k+j+L/2] =  __float2bfloat16(t4); 
          x[block_offset+k+L-j] = (t2!= 0) ?  __float2bfloat16(t2) : x[block_offset+k+L-j];  
          
        }
     }
      __syncthreads(); // sync between stages
   }
    
   }

void rdfft_cuda_inplace(Tensor& self, std::optional<c10::SymInt> n_opt, int64_t dim, std::optional<std::string> norm) {
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
        fft_inplace_kernel<float><<<grid_wx1, block_wx1, 0, stream>>>(d_data, batch, r, c, k, num_steps);
    } else if (self.dtype() == at::kDouble) {
        double* d_data = self.data_ptr<double>();
        fft_inplace_kernel<double><<<grid_wx1, block_wx1, 0, stream>>>(d_data, batch, r, c, k, num_steps);
    } else if (self.dtype() == at::kBFloat16) {
        __nv_bfloat16* d_data = reinterpret_cast<__nv_bfloat16*>(self.data_ptr<at::BFloat16>());
        fft_inplace_kernel_bf16<__nv_bfloat16><<<grid_wx1, block_wx1, 0, stream>>>(d_data, batch, r, c, k, num_steps);
    }

    // optional: 同步，确保 kernel 执行完成
    // cudaDeviceSynchronize();
}

}} // namespace at::native
