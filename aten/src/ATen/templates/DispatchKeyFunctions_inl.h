// ${generated_comment}

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// TODO: If necessary, consider adding <ATen/ops/{function}_key.h> headers
#ifdef TORCH_ASSERT_ONLY_METHOD_OPERATORS
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider if your change would be better placed in another file, or if a more \
  specific header might achieve the same goal.                                 \
  See NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

namespace at {
namespace ${dispatch_namespace} {

${dispatch_namespaced_declarations}

} // namespace ${dispatch_namespace}
} // namespace at
