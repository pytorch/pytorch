#include <ATen/RegisterCPU.h>

// ${generated_comment}

#include <ATen/Type.h>
#include <ATen/Context.h>
#include <ATen/detail/VariableHooksInterface.h>

${cpu_type_headers}

namespace at {

void register_cpu_types(Context * context) {
  ${cpu_type_registrations}
}

} // namespace at
