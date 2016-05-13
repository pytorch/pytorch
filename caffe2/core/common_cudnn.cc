#include "caffe2/core/common_cudnn.h"

#include "caffe2/core/init.h"

namespace caffe2 {

thread_local ThreadLocalCuDNNObjects CuDNNWrapper::tls_cudnn_objects_;

std::array<CuDNNWorkspace, CAFFE2_COMPILE_TIME_MAX_GPUS> CuDNNWrapper::scratch_;
std::array<size_t, CAFFE2_COMPILE_TIME_MAX_GPUS> CuDNNWrapper::nbytes_ = {0};
std::array<std::mutex, CAFFE2_COMPILE_TIME_MAX_GPUS> CuDNNWrapper::mutex_;

namespace {
bool PrintCuDNNInfo(int*, char***) {
  CAFFE_VLOG(1) << "Caffe2 is built with CuDNN version " << CUDNN_VERSION;
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(PrintCuDNNInfo, &PrintCuDNNInfo,
                              "Print CuDNN Info.");

}  // namespace
}  // namespace caffe2
