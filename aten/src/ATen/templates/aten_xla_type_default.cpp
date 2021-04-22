// ${generated_comment}
#include <torch_xla/csrc/aten_xla_type_default.h>

#include <ATen/Context.h>
#include <torch/library.h>
#include <ATen/CPUGeneratorImpl.h>

#include <tensorflow/compiler/xla/xla_client/debug_macros.h>
#include <tensorflow/compiler/xla/xla_client/metrics.h>
#include <tensorflow/compiler/xla/xla_client/tf_logging.h>
#include <torch_xla/csrc/aten_xla_bridge.h>
#include <torch_xla/csrc/aten_xla_type.h>
#include <torch_xla/csrc/function_call_tracker.h>

namespace ${cpp_namespace} {

${dispatch_aten_fallback_definitions}



TORCH_LIBRARY_IMPL(aten, XLA, m) {
${dispatch_registrations}

}
TORCH_LIBRARY_IMPL(aten, AutogradXLA, m) {
${dispatch_autograd_registrations}

}

}  // namespace torch_xla
