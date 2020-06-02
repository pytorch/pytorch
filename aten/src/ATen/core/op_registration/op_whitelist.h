#pragma once

/**
 * This header implements functionality to build PyTorch with only a certain
 * set of operators (+ dependencies) included.
 * - Build with -DTORCH_OPERATOR_WHITELIST="aten::add;aten::sub" and only these
 *   two ops will be included in your build.  NB: the whitelist does NOT contain
 *   overload names.
 *
 * Internally, this is done by removing the operator registration calls
 * using compile time programming, and the linker will then prune all
 * operator functions that weren't registered.
 */

#include <c10/util/constexpr_string_functions.h>

namespace c10 {

// Returns true iff the given op name is on the whitelist
// and should be registered
constexpr bool op_whitelist_check(const char* op_name) {
#if !defined(TORCH_OPERATOR_WHITELIST)
  // If the TORCH_OPERATOR_WHITELIST parameter is not defined,
  // all ops are to be registered
  return true;
#else
  return c10::util::op_whitelist_contains(C10_STRINGIZE(TORCH_OPERATOR_WHITELIST), op_name, ';');
#endif
}

}
