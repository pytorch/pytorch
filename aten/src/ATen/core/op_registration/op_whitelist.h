#pragma once

// TODO: unify to C10_MOBILE. In theory this header could be used in OSS.
#ifdef XPLAT_MOBILE_BUILD
#include <ATen/selected_mobile_ops.h>
#endif

/**
 * This header implements functionality to build PyTorch with only a certain
 * set of operators (+ dependencies) included.
 *
 * - Build with -DTORCH_OPERATOR_WHITELIST="aten::add;aten::sub" and only these
 *   two ops will be included in your build.  The whitelist records operators
 *   only, no overloads; if you include aten::add, all overloads of aten::add
 *   will be included.
 *
 * Internally, this is done by removing the operator registration calls
 * using compile time programming, and the linker will then prune all
 * operator functions that weren't registered.
 * See Note [Selective build] for more details
 *
 * WARNING: The whitelist mechanism doesn't work for all ways you could go about
 * registering an operator.  If the dispatch key / operator name is not
 * sufficiently obvious at compile time, then the whitelisting mechanism
 * will fail (and the operator will be included in the binary anyway).
 */

#include <c10/util/string_view.h>
#include <c10/core/DispatchKey.h>
#include <c10/macros/Macros.h>

namespace c10 {

namespace impl {

// returns true iff whitelist contains item
// op_whitelist_contains("a;bc;d", "bc") == true
constexpr bool op_whitelist_contains(string_view whitelist, string_view item) {
    size_t next = -1;
    for (size_t cur = 0; cur <= whitelist.size(); cur = next) {
      next = whitelist.find(';', cur);
      if (next != string_view::npos) {
        if (whitelist.substr(cur, next - cur).compare(item) == 0) {
          return true;
        }
        next++;
      } else {
        if (whitelist.substr(cur).compare(item) == 0) {
          return true;
        }
        break;
      }
    }
    return false;
}

// Returns true iff the given op name is on the whitelist
// and should be registered
constexpr bool op_whitelist_check(string_view op_name) {
  assert(op_name.find("::") != string_view::npos);
#if !defined(TORCH_OPERATOR_WHITELIST)
  // If the TORCH_OPERATOR_WHITELIST parameter is not defined,
  // all ops are to be registered
  return true;
#else
  return op_whitelist_contains(
    C10_STRINGIZE(TORCH_OPERATOR_WHITELIST),
    // This function is majorly used for mobile selective build with
    // root operators, where the overload is included in the whitelist.
    op_name);
    // // Strip overload name (as whitelist doesn't contain overloads)
    // // Another function based on this may be added when there's usage
    // // on op names without overload.
    // OperatorNameView::parse(op_name).name);
#endif
}

// Returns true iff the given schema string is on the whitelist
// and should be registered
constexpr bool schema_whitelist_check(string_view schema) {
#if defined(TORCH_FORCE_SCHEMA_REGISTRATION)
  return true;
#else
  return op_whitelist_check(schema.substr(0, schema.find("(")));
#endif
}

// schema_whitelist_check() implicitly depends on a macro, TORCH_OPERATOR_WHITELIST.
// Add this API to pass arbitrary whitelist.
constexpr bool op_whitelist_contains_name_in_schema(string_view whitelist, string_view schema) {
  return op_whitelist_contains(whitelist, schema.substr(0, schema.find("(")));
}

// Returns true iff the given dispatch key is on the whitelist
// and should be registered.  When we turn this on, the list of valid
// mobile dispatch keys is hard coded (but you need to make sure
// that you have the correct set of dispatch keys for this).
constexpr bool dispatch_key_whitelist_check(DispatchKey k) {
#ifdef C10_MOBILE
  return true;
  // Disabled for now: to be enabled later!
  // return k == DispatchKey::CPU || k == DispatchKey::Vulkan || k == DispatchKey::QuantizedCPU || k == DispatchKey::BackendSelect || k == DispatchKey::CatchAll;
#else
  return true;
#endif
}

} // namespace impl
} // namespace c10
