#include <c10/util/Exception.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Type.h>
#include <c10/util/Logging.h>

#include <iostream>
#include <sstream>
#include <numeric>
#include <string>

namespace c10 {

Error::Error(
    const std::string& new_msg,
    const std::string& backtrace,
    const void* caller)
    : msg_stack_{new_msg}, backtrace_(backtrace), caller_(caller) {
  refresh_msg();
}

// PyTorch-style error message
// Error::Error(SourceLocation source_location, const std::string& msg)
// NB: This is defined in Logging.cpp for access to GetFetchStackTrace

// Caffe2-style error message
Error::Error(
    const char* file,
    const uint32_t line,
    const char* condition,
    const std::string& msg,
    const std::string& backtrace,
    const void* caller)
    : Error(
          str("[enforce fail at ",
              detail::StripBasename(file),
              ":",
              line,
              "] ",
              condition,
              ". ",
              msg),
          backtrace,
          caller) {}

std::string Error::compute_msg(bool include_backtrace) const {
  if (msg_stack_.size() == 0) {
    if (include_backtrace) return backtrace_;
    return "";
  }

  std::ostringstream oss;

  if (msg_stack_.size() == 2) {
    // Fold error and context in one line
    oss << msg_stack_[0] << " (" << msg_stack_[1] << ")";
  } else {
    oss << msg_stack_[0];
    for (size_t i = 1; i < msg_stack_.size(); i++) {
      oss << "\n  " << msg_stack_[i];
    }
  }

  if (include_backtrace) {
    oss << "\n";
    oss << backtrace_;
  }

  return oss.str();
}

void Error::refresh_msg() {
  msg_ = compute_msg(/*include_backtrace*/ true);
  msg_without_backtrace_ = compute_msg(/*include_backtrace*/ false);
}

void Error::add_context(const std::string& new_msg) {
  msg_stack_.push_back(new_msg);
  // TODO: Calling add_context O(n) times has O(n^2) cost.  We can fix
  // this perf problem by populating the fields lazily... if this ever
  // actually is a problem.
  // NB: If you do fix this, make sure you do it in a thread safe way!
  // what() is almost certainly expected to be thread safe even when
  // accessed across multiple threads
  refresh_msg();
}

namespace Warning {

namespace {
  WarningHandler* getBaseHandler() {
    static WarningHandler base_warning_handler_ = WarningHandler();
    return &base_warning_handler_;
  };

  class ThreadWarningHandler {
    public:
      ThreadWarningHandler() = delete;

      static WarningHandler* get_handler() {
        if (!warning_handler_) {
          warning_handler_ = getBaseHandler();
        }
        return warning_handler_;
      }

      static void set_handler(WarningHandler* handler) {
        warning_handler_ = handler;
      }

    private:
      static thread_local WarningHandler* warning_handler_;
  };

  thread_local WarningHandler* ThreadWarningHandler::warning_handler_ = nullptr;

}

void warn(SourceLocation source_location, const std::string& msg) {
  ThreadWarningHandler::get_handler()->process(source_location, msg);
}

void set_warning_handler(WarningHandler* handler) noexcept(true) {
  ThreadWarningHandler::set_handler(handler);
}

WarningHandler* get_warning_handler() noexcept(true) {
  return ThreadWarningHandler::get_handler();
}

} // namespace Warning

void WarningHandler::process(
    const SourceLocation& source_location,
    const std::string& msg) {
  std::cerr << "Warning: " << msg << " (" << source_location << ")\n";
}


std::string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return demangle(typeid(e).name()) + ": " + e.what();
#else
  return std::string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

} // namespace c10
