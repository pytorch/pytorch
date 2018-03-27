#include "caffe2/core/context_gpu.h"
#include "conv_op_shared.h"

namespace caffe2 {

template <>
void createSharedBuffer<CUDAContext>(Workspace* ws) {
  auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA_MUTEX__")
                       ->GetMutable<std::unique_ptr<std::mutex>>();
  mutexPtr->reset(new std::mutex());
  ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__");
}

template <>
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<CUDAContext>* buffer)> f) {
  auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA_MUTEX__");
  CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

  auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
  std::lock_guard<std::mutex> g(**mutexPtr);
  auto* buffer = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__")
                     ->GetMutable<TensorCUDA>();
  f(buffer);
}
}
