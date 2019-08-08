#ifdef USE_QNNPACK

#include "init_qnnpack.h"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <qnnpack.h>

namespace at {
namespace native {
void initQNNPACK() {
  static std::once_flag once;
  static enum qnnp_status qnnpackStatus = qnnp_status_uninitialized;
  std::call_once(once, []() { qnnpackStatus = qnnp_initialize(); });
  TORCH_CHECK(
      qnnpackStatus == qnnp_status_success, "failed to initialize QNNPACK");
}
} // namespace native
} // namespace at

#endif
