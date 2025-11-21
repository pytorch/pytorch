#include <ATen/cuda/CUDAContextLight.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/c/shim.h>

AOTITorchError torch_get_current_cuda_blas_handle(void** ret_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cublasHandle_t*)(ret_handle) = at::cuda::getCurrentCUDABlasHandle();
  });
}
