#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/c/shim.h>

void* my_get_curr_cuda_blas_handle() {
  void* ret_handle;
  TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&ret_handle));
  return ret_handle;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_get_curr_cuda_blas_handle() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_get_curr_cuda_blas_handle", TORCH_BOX(&my_get_curr_cuda_blas_handle));
}
