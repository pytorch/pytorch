// This is used for factory methods dispatching. We register ops with a high priority dispatch key,
// which makes the dispatcher to dispatch here. We compute the dispatch key 'manually' and navigate to the
// correct backend.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/TensorOptions.h>

namespace at {
namespace native {

namespace {

${backend_select_method_definitions}

static auto registry = torch::RegisterOperators()
  ${backend_select_function_registrations};

} // namespace
} // native
} // at
