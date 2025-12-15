#include <torch/csrc/jit/frontend/error_report.h>

namespace torch::jit {

// Avoid storing objects with destructor in thread_local for mobile build.
#ifndef C10_MOBILE
// [NOTE: Thread-safe CallStack]
// `calls` maintains a stack of Python calls that resulted in the
// currently compiled TorchScript code. RAII ErrorReport::CallStack
// push and pop from the `calls` object during compilation to track
// these stacks so that they can be used to report compilation errors
//
// Q: Why can't this just be a thread_local vector<Call> (as it was previously)?
//
// A: Sometimes a CallStack RAII guard is created in Python in a given
//    thread (say, thread A). Then later, someone can call
//    sys._current_frames() from another thread (thread B), which causes
//    thread B to hold references to the CallStack guard. e.g.
//    1. CallStack RAII guard created by thread A
//    2. CallStack guard now has a reference from thread B
//    3. thread A releases guard, but thread B still holds a reference
//    4. thread B releases guard, refcount goes to 0, and we
//       call the destructor
//    under this situation, **we pop an element off the wrong `call`
//    object (from the wrong thread!)
//
//    To fix this:
//    * in CallStack, store a reference to which thread's `calls`
//      the CallStack corresponds to, so you can pop from the correct
//      `calls` object.
//    * make it a shared_ptr and add a mutex to make this thread safe
//      (since now multiple threads access a given thread_local calls object)
static thread_local std::shared_ptr<ErrorReport::Calls> calls =
    std::make_shared<ErrorReport::Calls>();
#endif // C10_MOBILE

ErrorReport::ErrorReport(const ErrorReport& e)
    : ss(e.ss.str()),
      context(e.context),
      the_message(e.the_message),
      error_stack(e.error_stack.begin(), e.error_stack.end()) {}

#ifndef C10_MOBILE
ErrorReport::ErrorReport(const SourceRange& r)
    : context(r), error_stack(calls->get_stack()) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {
  calls->update_pending_range(range);
}

ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {
  source_callstack_ = calls;
  source_callstack_->push_back({name, range});
}

ErrorReport::CallStack::~CallStack() {
  if (source_callstack_) {
    source_callstack_->pop_back();
  }
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
  return get_stacked_errors(calls->get_stack());
#else
  TORCH_CHECK(false, "Call stack not supported on mobile");
#endif // C10_MOBILE
}

const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << '\n' << ss.str();
  msg << ":\n";
  context.highlight(msg);

  msg << get_stacked_errors(error_stack);

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace torch::jit
