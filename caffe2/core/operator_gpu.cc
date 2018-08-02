#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <>
void Operator<CUDAContext>::SyncDevice() {
  auto* context = getContext();
  int device;
  cudaGetDevice(&device);

  cudaEvent_t ev;
  cudaSetDevice(context->cuda_gpu_id());
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaEventRecord(ev, context->cuda_stream());
  cudaEventSynchronize(ev);
  cudaEventDestroy(ev);
  cudaSetDevice(device);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    CAFFE_THROW("Encountered CUDA error Stop: ", cudaGetErrorString(error));
  }
}

} // namespace caffe2
