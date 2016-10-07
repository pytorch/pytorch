#include "caffe2/core/context_gpu.h"
#include "conv_op_shared.h"

namespace caffe2 {

template <>
void runWithSharedBuffer(
    Workspace* ws,
    std::function<void(Tensor<CUDAContext>* buffer)> f) {
  static std::mutex m;
  std::lock_guard<std::mutex> g(m);
  auto* buffer = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_CUDA__")
                     ->GetMutable<TensorCUDA>();
  f(buffer);
}
}
