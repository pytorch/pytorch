#include "conv_op_shared.h"
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/workspace.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(
    caffe2_force_shared_col_buffer,
    false,
    "Always use the shared col buffer");

namespace caffe2 {

template <>
void createSharedBuffer<CPUContext>(Workspace* ws) {
  auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU_MUTEX__")
                       ->GetMutable<std::unique_ptr<std::mutex>>();
  // NOLINTNEXTLINE(modernize-make-unique)
  mutexPtr->reset(new std::mutex());
  ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__");
}

template <>
void runWithSharedBuffer<CPUContext>(
    Workspace* ws,
    std::function<void(Tensor* buffer)> f) {
  auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU_MUTEX__");
  CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

  auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
  std::lock_guard<std::mutex> g(**mutexPtr);
  auto* buffer = BlobGetMutableTensor(
      ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__"), CPU);
  f(buffer);
}
}
