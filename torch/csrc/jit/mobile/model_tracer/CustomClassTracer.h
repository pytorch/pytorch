#pragma once

#include <ATen/record_function.h>
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

  CustomClassTracer() {
    auto recorder_cb = [](const at::RecordFunction& fn)
        -> std::unique_ptr<at::ObserverContext> {
      std::string name = fn.name();
      getLoadedClasses().insert(name);
      return nullptr;
    };

    handle_ =
        at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                  .scopes({at::RecordScope::CUSTOM_CLASS}));
  }

  static custom_classes_type& getLoadedClasses();

  ~CustomClassTracer() {
    at::removeCallback(handle_);
  }
};

} // namespace mobile
} // namespace jit
} // namespace torch
