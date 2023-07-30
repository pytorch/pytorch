
#include <torch/csrc/aot_inductor/c/aten_shim.h>

#include <c10/cuda/CUDAStream.h>

void aot_inductor_set_current_cuda_stream(
    void* stream,
    AOTInductorDeviceIndex device_index) {
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  c10::cuda::setCurrentCUDAStream(
      at::cuda::getStreamFromExternal(cuda_stream, device_index));
}
