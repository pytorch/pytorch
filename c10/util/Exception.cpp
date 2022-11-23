#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Type.h>

#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

namespace c10 {

Error::Error(std::string msg, std::string backtrace, const void* caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
  refresh_what();
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

std::string Error::compute_what(bool include_backtrace) const {
  std::ostringstream oss;

  oss << msg_;

  if (context_.size() == 1) {
    // Fold error and context in one line
    oss << " (" << context_[0] << ")";
  } else {
    for (const auto& c : context_) {
      oss << "\n  " << c;
    }
  }

  if (include_backtrace) {
    oss << "\n" << backtrace_;
  }

  return oss.str();
}

void Error::refresh_what() {
  what_ = compute_what(/*include_backtrace*/ true);
  what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
}

void Error::add_context(std::string new_msg) {
  context_.push_back(std::move(new_msg));
  // TODO: Calling add_context O(n) times has O(n^2) cost.  We can fix
  // this perf problem by populating the fields lazily... if this ever
  // actually is a problem.
  // NB: If you do fix this, make sure you do it in a thread safe way!
  // what() is almost certainly expected to be thread safe even when
  // accessed across multiple threads
  refresh_what();
}

namespace detail {

void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg) {
  throw ::c10::Error({func, file, line}, msg);
}

void torchCheckFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  throw ::c10::Error({func, file, line}, msg);
}

void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const char* userMsg) {
  torchCheckFail(func, file, line, c10::str(condMsg, userMsg));
}

// This should never be called. It is provided in case of compilers
// that don't do any dead code stripping in debug builds.
void torchInternalAssertFail(
    const char* func,
    const char* file,
    uint32_t line,
    const char* condMsg,
    const std::string& userMsg) {
  torchCheckFail(func, file, line, c10::str(condMsg, userMsg));
}

} // namespace detail

namespace WarningUtils {

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

} // namespace

void set_warning_handler(WarningHandler* handler) noexcept(true) {
  ThreadWarningHandler::set_handler(handler);
}

WarningHandler* get_warning_handler() noexcept(true) {
  return ThreadWarningHandler::get_handler();
}

bool warn_always = false;

void set_warnAlways(bool setting) noexcept(true) {
  warn_always = setting;
}

bool get_warnAlways() noexcept(true) {
  return warn_always;
}

WarnAlways::WarnAlways(bool setting /*=true*/)
    : prev_setting(get_warnAlways()) {
  set_warnAlways(setting);
}

WarnAlways::~WarnAlways() {
  set_warnAlways(prev_setting);
}

} // namespace WarningUtils

void warn(const Warning& warning) {
  WarningUtils::ThreadWarningHandler::get_handler()->process(warning);
}

Warning::Warning(
    warning_variant_t type,
    const SourceLocation& source_location,
    const std::string& msg,
    const bool verbatim)
    : type_(type),
      source_location_(source_location),
      msg_(msg),
      verbatim_(verbatim) {}

Warning::Warning(
    warning_variant_t type,
    SourceLocation source_location,
    detail::CompileTimeEmptyString msg,
    const bool verbatim)
    : Warning(type, std::move(source_location), "", verbatim) {}

Warning::Warning(
    warning_variant_t type,
    SourceLocation source_location,
    const char* msg,
    const bool verbatim)
    : type_(type),
      source_location_(std::move(source_location)),
      msg_(std::string(msg)),
      verbatim_(verbatim) {}

Warning::warning_variant_t Warning::type() const {
  return type_;
}

const SourceLocation& Warning::source_location() const {
  return source_location_;
}

const std::string& Warning::msg() const {
  return msg_;
}

bool Warning::verbatim() const {
  return verbatim_;
}

void WarningHandler::process(const Warning& warning) {
  LOG_AT_FILE_LINE(
      WARNING, warning.source_location().file, warning.source_location().line)
      << "Warning: " << warning.msg() << " (function "
      << warning.source_location().function << ")";
}

std::string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return demangle(typeid(e).name()) + ": " + e.what();
#else
  return std::string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

} // namespace c10
