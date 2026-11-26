#include <cublas_v2.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <tuple>

std::tuple<void*, void*> my_get_curr_cuda_blas_handle_stream(bool setup) {
  void* handle = torch::stable::accelerator::getCurrentCUDABlasHandle(setup);
  cudaStream_t stream;
  STD_TORCH_CHECK(
      cublasGetStream(static_cast<cublasHandle_t>(handle), &stream) ==
          CUBLAS_STATUS_SUCCESS,
      "cublasGetStream failed");
  return {handle, stream};
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_get_curr_cuda_blas_handle_stream(bool setup) -> (int, int)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_get_curr_cuda_blas_handle_stream",
      TORCH_BOX(&my_get_curr_cuda_blas_handle_stream));
}
