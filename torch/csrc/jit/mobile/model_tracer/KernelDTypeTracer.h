#pragma once

#include <ATen/record_function.h>
#include <c10/util/Synchronized.h>
#include <map>
#include <set>
#include <string>

namespace torch {
namespace jit {
namespace mobile {
/* The KernelDTypeTracer class handles the attachment and removal of a recording
 * callback that traces the invocation of code that handles specific dtypes in
 * kernel function implementations that are tagged with specific tags.
 *
 * You can get the set of kernel tags and the dtypes using
 * getCalledKernelTags().
 *
 * Note: This class is not thread safe or re-entrant, and should not be used
 * across multiple threads of execution.
 *
 */
struct KernelDTypeTracer final {
  at::CallbackHandle handle_;
  /* The key of the map below (std::string) is the kernel tag name (constant
   * character string) which shows up in code. The value part of type
   * std::set<std::string> is the collection of dtypes for which we need to
   * generate code for the said kernel tag.
   */
  typedef std::map<std::string, std::set<std::string>> kernel_tags_type;

  KernelDTypeTracer();
  static c10::Synchronized<kernel_tags_type>& getCalledKernelTags();

  ~KernelDTypeTracer() {
    at::removeCallback(handle_);
  }
};
} // namespace mobile
} // namespace jit
} // namespace torch
