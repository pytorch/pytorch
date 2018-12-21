#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/conv_op_shared.h"

namespace caffe2 {

template <>
void createSharedBuffer<CUDAContext>(Workspace* ws) {
  auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA_MUTEX__")
                       ->GetMutable<std::unique_ptr<std::mutex>>();
  mutexPtr->reset(new std::mutex());
  ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__");
}

template <>
void runWithSharedBuffer<CUDAContext>(
    Workspace* ws,
    std::function<void(Tensor* buffer)> f) {
  auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA_MUTEX__");
  CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

  auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
  std::lock_guard<std::mutex> g(**mutexPtr);
  auto* buffer = BlobGetMutableTensor(
      ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__"), CUDA);
  f(buffer);
}
}
