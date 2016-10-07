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
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<CPUContext>* buffer)> f) {
  static std::mutex m;
  std::lock_guard<std::mutex> g(m);
  auto* buffer = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CPU__")
                     ->GetMutable<TensorCPU>();
  f(buffer);
}
}
