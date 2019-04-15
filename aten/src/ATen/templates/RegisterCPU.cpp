#include <ATen/RegisterCPU.h>

// ${generated_comment}

#include <ATen/Type.h>
#include <ATen/Context.h>
#include <ATen/UndefinedType.h>
#include <ATen/core/VariableHooksInterface.h>

${cpu_type_headers}

namespace at {

void register_cpu_types(Context * context) {
  ${cpu_type_registrations}
  context->registerType(Backend::Undefined, new UndefinedType());
}

} // namespace at
