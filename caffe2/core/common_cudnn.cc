#include <sstream>

#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/init.h"

namespace caffe2 {
namespace {
bool PrintCuDNNInfo() {
  CAFFE_LOG_INFO << "Caffe2 is built with CuDNN version " << CUDNN_VERSION;
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(PrintCuDNNInfo, &PrintCuDNNInfo,
                              "Print CuDNN Info.");

}  // namespace
}  // namespace caffe2
