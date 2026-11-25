#include <cublas_v2.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <tuple>

std::tuple<int64_t, int64_t> my_get_curr_cuda_blas_handle_stream(bool setup) {
  void* handle;
  TORCH_ERROR_CODE_CHECK(
      torch_get_current_cuda_blas_handle_v2(setup, &handle));
  cudaStream_t stream;
  STD_TORCH_CHECK(
      cublasGetStream(static_cast<cublasHandle_t>(handle), &stream) ==
          CUBLAS_STATUS_SUCCESS,
      "cublasGetStream failed");
  return {reinterpret_cast<int64_t>(handle), reinterpret_cast<int64_t>(stream)};
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_get_curr_cuda_blas_handle_stream(bool setup) -> (int, int)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_get_curr_cuda_blas_handle_stream",
      TORCH_BOX(&my_get_curr_cuda_blas_handle_stream));
}
