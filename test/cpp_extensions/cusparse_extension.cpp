#include <iostream>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cublas_v2.h>
#include <cusparse.h>

torch::Tensor noop_cusparse_function(torch::Tensor x) {
  cusparseHandle_t sparse_handle;
  TORCH_CUDASPARSE_CHECK(cusparseCreate(&sparse_handle));
  TORCH_CUDASPARSE_CHECK(cusparseDestroy(sparse_handle));
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("noop_cusparse_function", &noop_cusparse_function, "a cublas function");
}
