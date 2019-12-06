#include "c10/util/Exception.h"
#include "c10/util/Backtrace.h"
#include "c10/util/Type.h"
#include "c10/util/Logging.h"

#include <iostream>
#include <numeric>
#include <string>

namespace c10 {

Error::Error(
    const std::string& new_msg,
    const std::string& backtrace,
    const void* caller)
    : msg_stack_{new_msg}, backtrace_(backtrace), caller_(caller) {
  msg_ = msg();
  msg_without_backtrace_ = msg_without_backtrace();
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
              msg,
              "\n"),
          backtrace,
          caller) {}

std::string Error::msg() const {
  return std::accumulate(
             msg_stack_.begin(), msg_stack_.end(), std::string("")) +
      backtrace_;
}

std::string Error::msg_without_backtrace() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string(""));
}

void Error::AppendMessage(const std::string& new_msg) {
  msg_stack_.push_back(new_msg);
  // Refresh the cache
  // TODO: Calling AppendMessage O(n) times has O(n^2) cost.  We can fix
  // this perf problem by populating the fields lazily... if this ever
  // actually is a problem.
  msg_ = msg();
  msg_without_backtrace_ = msg_without_backtrace();
}

namespace Warning {

namespace {
  WarningHandler* getHandler() {
    static WarningHandler base_warning_handler_ = WarningHandler();
    return &base_warning_handler_;
  };
  static thread_local WarningHandler* warning_handler_ = getHandler();

}

void warn(SourceLocation source_location, const std::string& msg) {
  warning_handler_->process(source_location, msg);
}

void set_warning_handler(WarningHandler* handler) noexcept(true) {
  warning_handler_ = handler;
}

WarningHandler* get_warning_handler() noexcept(true) {
  return warning_handler_;
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
