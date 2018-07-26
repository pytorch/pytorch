#include "conv_op_shared.h"
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/workspace.h"

CAFFE2_DEFINE_bool(
    caffe2_force_shared_col_buffer,
    false,
    "Always use the shared col buffer");

namespace caffe2 {

template <>
void createSharedBuffer<CPUContext>(Workspace* ws) {
  auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU_MUTEX__")
                       ->GetMutable<std::unique_ptr<std::mutex>>();
  mutexPtr->reset(new std::mutex());
  ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__");
}

template <>
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<CPUContext>* buffer)> f) {
  auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU_MUTEX__");
  CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

  auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
  std::lock_guard<std::mutex> g(**mutexPtr);
  auto* buffer =
      ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__")->GetMutable<TensorCPU>();
  f(buffer);
}
}
