
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <c10/cuda/CUDAStream.h>

#include <iostream>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTITorchError::Failure;                          \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTITorchError::Failure;                          \
  }                                                          \
  return AOTITorchError::Success;

AOTITorchError aoti_torch_set_current_cuda_stream(
    void* stream,
    int32_t device_index) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    c10::cuda::setCurrentCUDAStream(
        at::cuda::getStreamFromExternal(cuda_stream, device_index));
  });
}
