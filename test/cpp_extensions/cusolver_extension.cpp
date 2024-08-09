#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cusolverDn.h>
#include <cusolverSp.h>


torch::Tensor noop_cusolver_function(torch::Tensor x) {
  cusolverDnHandle_t handle;
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(&handle));
  TORCH_CUSOLVER_CHECK(cusolverDnDestroy(handle));
  cusolverSpHandle_t sphandle;
  TORCH_CUSOLVER_CHECK(cusolverSpCreate(&sphandle));
  TORCH_CUSOLVER_CHECK(cusolverSpDestroy(sphandle));
  return x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("noop_cusolver_function", &noop_cusolver_function, "a cusolver function");
}
