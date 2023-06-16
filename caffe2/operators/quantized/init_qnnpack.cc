#include <mutex>

#include <qnnpack.h>

#include "caffe2/core/logging.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

void initQNNPACK() {
  static std::once_flag once;
  static enum qnnp_status qnnpackStatus = qnnp_status_uninitialized;
  std::call_once(once, []() { qnnpackStatus = qnnp_initialize(); });
  CAFFE_ENFORCE(
      qnnpackStatus == qnnp_status_success, "failed to initialize QNNPACK");
}

} // namespace caffe2
