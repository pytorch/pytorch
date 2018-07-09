#include "THC/THC.h"
#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/cudnn/Exceptions.h" // for ATEN_CUDA_CHECK

#include <cublas_v2.h>

namespace at {
namespace native {

template<class scalar_t>
int cublasGetri(cublasHandle_t handle, int n, scalar_t **a, int lda, int *pivots, scalar_t **b, int m, int *info, int batch_size) {
  AT_ERROR("cuBLAS getri is only implemented for double and float");
}

template<class scalar_t>
int cublasGetrf(cublasHandle_t handle, int n, scalar_t **a, int lda, int *pivots, int *info, int batch_size) {
  AT_ERROR("cuBLAS getrf is only implemented for double and float");
}

template<> int cublasGetri<float>(
cublasHandle_t handle, int n, float **a, int lda, int *pivots, float **b, int m, int *info, int batch_size) {
    return cublasSgetriBatched(handle, n, (const float**)a, lda, pivots, b, m, info, batch_size);
}

template<> int cublasGetri<double>(
cublasHandle_t handle, int n, double **a, int lda, int *pivots, double **b, int m, int *info, int batch_size) {
    return cublasDgetriBatched(handle, n, (const double**)a, lda, pivots, b, m, info, batch_size);
}

template<> int cublasGetrf<float>(
cublasHandle_t handle, int n, float **a, int lda, int *pivots, int *info, int batch_size) {
    return cublasSgetrfBatched(handle, n, a, lda, pivots, info, batch_size);
}

template<> int cublasGetrf<double>(
cublasHandle_t handle, int n, double **a, int lda, int *pivots, int *info, int batch_size) {
    return cublasDgetrfBatched(handle, n, a, lda, pivots, info, batch_size);
}

// Float and double have different sizes, so this needs to be templated
template<class scalar_t>
__global__ void createInverseBuffers(scalar_t **buffer_a, scalar_t *data_a, scalar_t **buffer_b, scalar_t *data_b, int n, int batch_size)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size)
    {
      buffer_a[idx] = data_a + n*n*idx;
      buffer_b[idx] = data_b + n*n*idx;
    }
}

// Tensor inverse
Tensor inverse_cuda(const Tensor& self) {
 
    if(self.dim() < 2)
    {
        AT_ERROR("Inverse: expected a tensor with 2 or more dimensions, got ", self.dim());
    }

    cublasHandle_t handle = globalContext().getCurrentCublasHandle();
    THCState *thcState = globalContext().getTHCState();
    Tensor input_contiguous = self.contiguous();

    int n = self.size(-1);
    int m = self.size(-2);
    int lda = n;
    if(n != m)
    {
        AT_ERROR("Inverse: expected square matrices");
    }
    Tensor input_reshaped = input_contiguous.view({-1, m, n});
    Tensor input_colmajor = cloneBatchedColumnMajor(input_reshaped);
    int batch_size = input_colmajor.size(0);
    Tensor result = at::empty_like(input_colmajor);

    AT_DISPATCH_FLOATING_TYPES(input_reshaped.type(), "inverse_cuda", [&] {
        auto input_data = input_colmajor.data<scalar_t>();
        auto output_data = result.data<scalar_t>();
        bool errorOccurred = false;
        int *info_cpu = new int[batch_size];

        scalar_t **input_gpu;
        scalar_t **output_gpu;
        scalar_t **input_ptrs = new scalar_t*[batch_size];
        scalar_t **output_ptrs = new scalar_t*[batch_size];
        AT_CUDA_CHECK(THCudaMalloc(thcState, (void**)&input_gpu, batch_size*sizeof(scalar_t*)));
        AT_CUDA_CHECK(THCudaMalloc(thcState, (void**)&output_gpu, batch_size*sizeof(scalar_t*)));
        int *pivots_gpu, *info_gpu;
        AT_CUDA_CHECK(THCudaMalloc(thcState, (void**)&pivots_gpu, n*batch_size*sizeof(int)));
        AT_CUDA_CHECK(THCudaMalloc(thcState, (void**)&info_gpu, batch_size*sizeof(int)));
        
        const int block = 512;
        const int grid = (batch_size + block - 1) / block;

        createInverseBuffers<<<grid, block, 0, globalContext().getCurrentCUDAStream()>>>(
            input_gpu, input_data, output_gpu, output_data, n, batch_size);

        // Lu
        cublasGetrf(handle, n, input_gpu, lda, pivots_gpu, info_gpu, batch_size);

        // Check error codes
        AT_CUDA_CHECK(cudaMemcpy(info_cpu, info_gpu, batch_size*sizeof(int), cudaMemcpyDeviceToHost));
        for(int i=0; i<batch_size; ++i) {
           errorOccurred |= info_cpu[i];
        }

        if(!errorOccurred) {
            // Invert
            cublasGetri(handle, n, input_gpu, lda, pivots_gpu, output_gpu, n, info_gpu, batch_size);
 
            // Check error codes
            AT_CUDA_CHECK(cudaMemcpy(info_cpu, info_gpu, batch_size*sizeof(int), cudaMemcpyDeviceToHost));
            for(int i=0; i<batch_size; ++i) {
               errorOccurred |= info_cpu[i];
            }
        }

        AT_CUDA_CHECK(THCudaFree(thcState, input_gpu));
        AT_CUDA_CHECK(THCudaFree(thcState, output_gpu));
        AT_CUDA_CHECK(THCudaFree(thcState, pivots_gpu));
        AT_CUDA_CHECK(THCudaFree(thcState, info_gpu));
        delete input_ptrs;
        delete output_ptrs;
        delete info_cpu;
        
        if(errorOccurred) { 
            AT_ERROR("Unable to invert some of the matrices.");
        }
    });

    return result.view(self.sizes()).transpose(-1, -2);
}

Tensor& inverse_out_cuda(Tensor &result, const Tensor& self) {
    result = inverse_cuda(self);
    return result;
}

}} // namespace at::native
