#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/DispatchStub.h>

namespace at {

// NB: this is explicitly copied here (via codegen) rather than
// included via NativeFunctions.h to avoid recompiling this file when
// NativeFunctions.h changes
namespace meta {
$ {
  meta_declaration
}
} // namespace meta

namespace native {
${native_declaration} $ {
  native_definitions
}
} // namespace native
} // namespace at
