#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cusolverDn.h>


torch::Tensor noop_cusolver_function(torch::Tensor x) {
  cusolverDnHandle_t handle;
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(&handle));
  TORCH_CUSOLVER_CHECK(cusolverDnDestroy(handle));
  return x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("noop_cusolver_function", &noop_cusolver_function, "a cusolver function");
}
