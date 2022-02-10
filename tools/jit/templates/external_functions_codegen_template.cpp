// ${generated_comment}
#include <torch/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch {
namespace jit {
namespace tensorexpr {

#ifdef C10_MOBILE
extern "C" {
#endif

${external_functions}

#ifndef C10_MOBILE
${external_registrations}
#endif

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace tensorexpr
} // namespace jit
} // namespace torch
