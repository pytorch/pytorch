#include <ATen/native/ufunc/${name}.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>

namespace at {

// NB: this is explicitly copied here (via codegen) rather than
// included via NativeFunctions.h to avoid recompiling this file when
// NativeFunctions.h changes
namespace meta {
${meta_declaration}
}

namespace native {
${native_definitions}
}} // namespace at::native
