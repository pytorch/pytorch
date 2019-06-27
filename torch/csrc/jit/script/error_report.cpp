#include <torch/csrc/jit/script/error_report.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/script/tree.h>

namespace torch {
namespace jit {
namespace script {

thread_local std::vector<SourceRange> call_stack;
thread_local std::vector<std::string> fn_stack;

void push_call(const SourceRange& range) {
  call_stack.push_back(range);
}

void pop_call() {
  call_stack.pop_back();
}

void push_function(const std::string& name) {
  fn_stack.push_back(name);
}

void pop_function() {
  fn_stack.pop_back();
}

C10_EXPORT const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  if (context) {
    msg << ":\n";
    context->highlight(msg);
  } else {
    msg << ".\n";
  }

  if (call_stack.size() > 0) {
    // Print the calling stack to show why this function was being compiled,
    // skip the first entry in the list since it's the current function
    auto range_it = call_stack.rbegin() + 1;
    auto names_it = fn_stack.rbegin();
    for (; range_it != call_stack.rend() && names_it != fn_stack.rend();
         ++range_it, ++names_it) {
      msg << "\n";
      msg << "'" << *names_it
          << "' is being compiled since it was called from '" << *(names_it + 1)
          << "'\n";
      range_it->highlight(msg);
    }
  }

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace script
} // namespace jit
} // namespace torch
