// We register ops with a higher priority dispatch key (BackendSelect) than the usual backend-specific keys (e.g. CPU)
// which makes calls to the factory functions dispatch to here.
// We then 'manually' compute a lower-priority to re-dispatch to (e.g. CPU) to get to the eventually correct backend.

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorOptions.h>

namespace at {

namespace {

${backend_select_method_definitions}

static auto registry = torch::RegisterOperators()
  ${backend_select_function_registrations};


} // namespace
} // at
