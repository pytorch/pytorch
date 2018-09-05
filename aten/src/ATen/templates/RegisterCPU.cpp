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
  context->type_registry[static_cast<int>(Backend::Undefined)]
                        [static_cast<int>(ScalarType::Undefined)].reset(new UndefinedType());
}

} // namespace at
