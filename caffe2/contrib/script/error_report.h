#pragma once

#include "caffe2/contrib/script/tree.h"

namespace caffe2 {
namespace script {

struct ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e)
      : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

  ErrorReport() : context(nullptr) {}
  explicit ErrorReport(const SourceRange& r)
      : context(std::make_shared<SourceRange>(r)) {}
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}
  virtual const char* what() const noexcept override {
    std::stringstream msg;
    msg << "\n" << ss.str();
    if (context != nullptr) {
      msg << ":\n";
      context->highlight(msg);
    } else {
      msg << ".\n";
    }
    the_message = msg.str();
    return the_message.c_str();
  }

 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);

  mutable std::stringstream ss;
  std::shared_ptr<SourceRange> context;
  mutable std::string the_message;
};

template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}

#define C2S_ASSERT(ctx, cond)                                              \
  if (!(cond)) {                                                           \
    throw ::caffe2::script::ErrorReport(ctx)                               \
        << __FILE__ << ":" << __LINE__ << ": assertion failed: " << #cond; \
  }
} // namespace script
} // namespace caffe2
