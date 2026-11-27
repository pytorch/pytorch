#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/accelerator.h>

void* my_get_curr_cuda_blas_handle() {
  return torch::stable::accelerator::getCurrentCUDABlasHandle();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_get_curr_cuda_blas_handle() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_get_curr_cuda_blas_handle", TORCH_BOX(&my_get_curr_cuda_blas_handle));
}
