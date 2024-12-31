#ifdef USE_PYTORCH_QNNPACK

#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <c10/util/Exception.h>
#include <pytorch_qnnpack.h>

#include <c10/util/CallOnce.h>

namespace at::native {

void initQNNPACK() {
  static c10::once_flag once;
  static enum pytorch_qnnp_status qnnpackStatus =
      pytorch_qnnp_status_uninitialized;
  c10::call_once(once, []() { qnnpackStatus = pytorch_qnnp_initialize(); });
  TORCH_CHECK(
      qnnpackStatus == pytorch_qnnp_status_success,
      "failed to initialize QNNPACK");
}

} // namespace at::native

#endif
