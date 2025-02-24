#include <torch/csrc/jit/frontend/error_report.h>

#include <torch/csrc/jit/frontend/tree.h>

namespace torch::jit {

// Avoid storing objects with destructor in thread_local for mobile build.
#ifndef C10_MOBILE
thread_local std::vector<Call> calls;
#endif // C10_MOBILE

ErrorReport::ErrorReport(const ErrorReport& e)
    : ss(e.ss.str()),
      context(e.context),
      the_message(e.the_message),
      error_stack(e.error_stack.begin(), e.error_stack.end()) {}

#ifndef C10_MOBILE
ErrorReport::ErrorReport(const SourceRange& r)
    : context(r), error_stack(calls.begin(), calls.end()) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {
  calls.back().caller_range = range;
}

ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {
  calls.push_back({name, range});
}

ErrorReport::CallStack::~CallStack() {
  calls.pop_back();
}
#else // defined C10_MOBILE
ErrorReport::ErrorReport(const SourceRange& r) : context(r) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {}

ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {}

ErrorReport::CallStack::~CallStack() {}
#endif // C10_MOBILE

static std::string get_stacked_errors(const std::vector<Call>& error_stack) {
  std::stringstream msg;
  if (!error_stack.empty()) {
    for (auto it = error_stack.rbegin(); it != error_stack.rend() - 1; ++it) {
      auto callee = it + 1;

      msg << "'" << it->fn_name
          << "' is being compiled since it was called from '" << callee->fn_name
          << "'\n";
      callee->caller_range.highlight(msg);
    }
  }
  return msg.str();
}

std::string ErrorReport::current_call_stack() {
#ifndef C10_MOBILE
  return get_stacked_errors(calls);
#else
  TORCH_CHECK(false, "Call stack not supported on mobile");
#endif // C10_MOBILE
}

const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  msg << ":\n";
  context.highlight(msg);

  msg << get_stacked_errors(error_stack);

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace torch::jit
