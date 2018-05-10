#include <ATen/Type.h>
#include <ATen/Context.h>
#include <ATen/detail/VariableHooksInterface.h>

${cuda_type_headers}

namespace at {

void register_cuda_types(Context * context) {
  ${cuda_type_registrations}
}

} // namespace at
