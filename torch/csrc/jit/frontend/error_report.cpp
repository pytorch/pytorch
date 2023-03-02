#include <torch/csrc/jit/frontend/error_report.h>

#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/utils/cpp_stacktraces.h>
#include <torch/csrc/utils/memory.h>

namespace torch::jit {

// Avoid storing objects with destructor in thread_local for mobile build.
#ifndef C10_MOBILE
thread_local std::vector<Call> calls;

namespace {
std::string unwrap_backtrace(const c10::optional<std::string>& backtrace) {
  if (backtrace.has_value()) {
    return backtrace.value();
  }
  return c10::get_backtrace(/*frames_to_skip=*/1);
}
} // namespace
#else // defined c10_MOBILE

namespace {
std::string unwrap_backtrace(const c10::optional<std::string>& backtrace) {
  if (backtrace.has_value()) {
    return backtrace.value();
  }
  return std::string("");
}
} // namespace

#endif // C10_MOBILE

ErrorReport::ErrorReport(const ErrorReport& e)
    : ss(e.ss.str()),
      context(e.context),
      the_message(e.the_message),
      error_stack(e.error_stack.begin(), e.error_stack.end()),
      backtrace_(e.backtrace_) {}

#ifndef C10_MOBILE
ErrorReport::ErrorReport(
    SourceRange r,
    const c10::optional<std::string>& backtrace)
    : context(std::move(r)),
      error_stack(calls.begin(), calls.end()),
      backtrace_(unwrap_backtrace(backtrace)) {}

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
ErrorReport::ErrorReport(
    SourceRange r,
    const c10::optional<std::string>& backtrace)
    : context(std::move(r)), backtrace_(unwrap_backtrace(backtrace)) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {}

ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {}

ErrorReport::CallStack::~CallStack() {}
#endif // C10_MOBILE

std::string get_stacked_errors(const std::vector<Call>& error_stack) {
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
  AT_ERROR("Call stack not supported on mobile");
#endif // C10_MOBILE
}

const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  msg << ":\n";
  context.highlight(msg);

  msg << get_stacked_errors(error_stack);

  if (get_cpp_stacktraces_enabled()) {
    msg << "\n" << backtrace_;
  }

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace torch::jit
