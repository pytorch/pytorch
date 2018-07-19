#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/cudnn_wrappers.h"

#include "caffe2/core/init.h"

namespace caffe2 {

CuDNNWrapper::PerGPUCuDNNStates& CuDNNWrapper::cudnn_states() {
  // New it (never delete) to avoid calling the destructors on process
  // exit and racing against the CUDA shutdown sequence.
  static auto* p = new CuDNNWrapper::PerGPUCuDNNStates();
  CHECK_NOTNULL(p);
  return *p;
}

namespace {
bool PrintCuDNNInfo(int*, char***) {
  VLOG(1) << "Caffe2 is built with CuDNN version " << CUDNN_VERSION;
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(PrintCuDNNInfo, &PrintCuDNNInfo,
                              "Print CuDNN Info.");

}  // namespace
}  // namespace caffe2
