#include <iostream>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cublas_v2.h>

torch::Tensor noop_cublas_function(torch::Tensor x) {
  cublasHandle_t handle;
  TORCH_CUDABLAS_CHECK(cublasCreate(&handle));
  TORCH_CUDABLAS_CHECK(cublasDestroy(handle));
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("noop_cublas_function", &noop_cublas_function, "a cublas function");
}
