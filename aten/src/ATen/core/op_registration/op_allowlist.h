#pragma once

// TODO: unify to C10_MOBILE. In theory this header could be used in OSS.
#ifdef TEMPLATE_SELECTIVE_BUILD
#include <ATen/selected_mobile_ops.h>
#endif

/**
 * This header implements functionality to build PyTorch with only a certain
 * set of operators (+ dependencies) included.
 *
 * - Build with -DTORCH_OPERATOR_WHITELIST="aten::add;aten::sub" and only these
 *   two ops will be included in your build.  The allowlist records operators
 *   only, no overloads; if you include aten::add, all overloads of aten::add
 *   will be included.
 *
 * Internally, this is done by removing the operator registration calls
 * using compile time programming, and the linker will then prune all
 * operator functions that weren't registered.
 * See Note [Selective build] for more details
 *
 * WARNING: The allowlist mechanism doesn't work for all ways you could go about
 * registering an operator.  If the dispatch key / operator name is not
 * sufficiently obvious at compile time, then the allowlisting mechanism
 * will fail (and the operator will be included in the binary anyway).
 */

#include <c10/core/DispatchKey.h>
#include <c10/macros/Macros.h>
#include <c10/util/string_view.h>

#if defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE)
#include <ATen/record_function.h>
#endif

namespace c10::impl {

constexpr bool allowlist_contains(
    string_view allowlist,
    string_view item); // Forward Declare

/**
 * In selective build mode returns true/false depending on whether a build
 * feature is available or not.
 *
 * In instrumenting mode (tracing mode), always returns true, and doesn't
 * trigger any side effects.
 */
constexpr bool is_build_feature_available(const char* name) {
#if !defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE)
  // Selective Build mode.
#if !defined(TORCH_BUILD_FEATURE_ALLOWLIST)
  (void)name;
  return true;
#else
  return allowlist_contains(C10_STRINGIZE(TORCH_BUILD_FEATURE_ALLOWLIST), name);
#endif

#else
  // Instrumenting mode.
  (void)name;
  return true;
#endif
}

[[noreturn]] void build_feature_required_feature_not_available(
    const char* feature);

/**
 * Use BUILD_FEATURE_REQUIRED macro in user-code.
 *
 * In selective build mode becomes a no-op if the build feature passed
 * in is available. If not available, throws an exception (c10::Error).
 * The compiler is able to perform dead code elimination for code
 * following this method if the build feature is not available.
 *
 * In instrumenting mode (tracing mode), registers (as a side effect)
 * the presence of this specific build feature being triggered.
 */
#if !defined(ENABLE_RECORD_KERNEL_FUNCTION_DTYPE) // selective build mode

#if defined(TORCH_BUILD_FEATURE_ALLOWLIST)
#define BUILD_FEATURE_REQUIRED(NAME)                                 \
  if (!c10::impl::is_build_feature_available(NAME)) {                \
    ::c10::impl::build_feature_required_feature_not_available(NAME); \
  }
#else // Everything trivially selected
#define BUILD_FEATURE_REQUIRED(NAME)

#endif

#else // trace mode
#define BUILD_FEATURE_REQUIRED(NAME) \
  RECORD_FUNCTION_WITH_SCOPE(        \
      at::RecordScope::BUILD_FEATURE, std::string(NAME), {});
#endif

// Use this macro, and not is_build_feature_available
#define BUILD_FEATURE_AVAILABLE(NAME) \
  ::c10::impl::is_build_feature_available(NAME)

// returns true iff allowlist contains item
// allowlist_contains("a;bc;d", "bc") == true
constexpr bool allowlist_contains(string_view allowlist, string_view item) {
  // Choose a really big value for next so that if something goes wrong
  // this code will blow up in a hopefully detectable way.
  size_t next = std::numeric_limits<size_t>::max();
  for (size_t cur = 0; cur <= allowlist.size(); cur = next) {
    next = allowlist.find(';', cur);
    if (next != string_view::npos) {
      if (allowlist.substr(cur, next - cur).compare(item) == 0) {
        return true;
      }
      next++;
    } else {
      if (allowlist.substr(cur).compare(item) == 0) {
        return true;
      }
      break;
    }
  }
  return false;
}

// Returns true iff the given op name is on the allowlist
// and should be registered
constexpr bool op_allowlist_check(string_view op_name [[maybe_unused]]) {
  assert(op_name.find("::") != string_view::npos);
  // Use assert() instead of throw() due to a gcc bug. See:
  // https://stackoverflow.com/questions/34280729/throw-in-constexpr-function
  // https://github.com/fmtlib/fmt/issues/682
  assert(op_name.find("(") == string_view::npos);
#if !defined(TORCH_OPERATOR_WHITELIST)
  // If the TORCH_OPERATOR_WHITELIST parameter is not defined,
  // all ops are to be registered
  return true;
#else
  return allowlist_contains(
      C10_STRINGIZE(TORCH_OPERATOR_WHITELIST),
      // This function is majorly used for mobile selective build with
      // root operators, where the overload is included in the allowlist.
      op_name);
  // // Strip overload name (as allowlist doesn't contain overloads)
  // // Another function based on this may be added when there's usage
  // // on op names without overload.
  // OperatorNameView::parse(op_name).name);
#endif
}

// Returns true iff the given schema string is on the allowlist
// and should be registered
constexpr bool schema_allowlist_check(string_view schema) {
#if defined(TORCH_FORCE_SCHEMA_REGISTRATION)
  return true;
#else
  return op_allowlist_check(schema.substr(0, schema.find("(")));
#endif
}

// Returns true iff the given custom class name is on the allowlist
// and should be registered
constexpr bool custom_class_allowlist_check(string_view custom_class_name) {
#if !defined(TORCH_CUSTOM_CLASS_ALLOWLIST)
  // If the TORCH_CUSTOM_CLASS_ALLOWLIST parameter is not defined,
  // all custom classes are to be registered
  (void)custom_class_name;
  return true;
#else
  return allowlist_contains(
      C10_STRINGIZE(TORCH_CUSTOM_CLASS_ALLOWLIST), custom_class_name);
#endif
}

// schema_allowlist_check() implicitly depends on a macro,
// TORCH_OPERATOR_WHITELIST. Add this API to pass arbitrary allowlist.
constexpr bool op_allowlist_contains_name_in_schema(
    string_view allowlist,
    string_view schema) {
  return allowlist_contains(allowlist, schema.substr(0, schema.find("(")));
}

// Returns true iff the given dispatch key is on the allowlist
// and should be registered.  When we turn this on, the list of valid
// mobile dispatch keys is hard coded (but you need to make sure
// that you have the correct set of dispatch keys for this).
constexpr bool dispatch_key_allowlist_check(DispatchKey /*k*/) {
#ifdef C10_MOBILE
  return true;
  // Disabled for now: to be enabled later!
  // return k == DispatchKey::CPU || k == DispatchKey::Vulkan || k ==
  // DispatchKey::QuantizedCPU || k == DispatchKey::BackendSelect || k ==
  // DispatchKey::CatchAll;
#else
  return true;
#endif
}

} // namespace c10::impl
