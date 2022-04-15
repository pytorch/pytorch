#pragma once

#include <ATen/record_function.h>
#include <c10/util/Synchronized.h>
#include <map>
#include <set>
#include <string>

namespace torch {
namespace jit {
namespace mobile {

/* The CustomClassTracer class handles the attachment and removal of a recording
 * callback that traces the invocation of code that handles loading custom
 * classes on mobile.
 *
 * You can get the set of used custom classes using
 * getLoadedClasses().
 *
 * Note: This class is not thread safe or re-entrant, and should not be used
 * across multiple threads of execution.
 *
 */
struct CustomClassTracer final {
  at::CallbackHandle handle_;
  /* These are the custom class names (constant
   * character string) which shows up in code.
   */
  typedef std::set<std::string> custom_classes_type;

  CustomClassTracer();
  static c10::Synchronized<custom_classes_type>& getLoadedClasses();

  ~CustomClassTracer() {
    at::removeCallback(handle_);
  }
};

} // namespace mobile
} // namespace jit
} // namespace torch
