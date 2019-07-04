#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/jit/script/tree.h>

namespace torch {
namespace jit {
namespace script {


struct CAFFE2_API ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e)
      : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

  ErrorReport() : context(c10::nullopt) {}
  explicit ErrorReport(SourceRange r) : context(std::move(r)) {}
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}

  const char* what() const noexcept override;

  struct CallStack {
    // These functions are used to report why a function was being compiled (i.e.
    // what was the call stack of user functions at compilation time that led to
    // this error)
    static void update_pending_range(const SourceRange& range);
    static void push_function(const std::string& name);
    static void pop_function();
  };

 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);

  mutable std::stringstream ss;
  c10::optional<SourceRange> context;
  mutable std::string the_message;
};

template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}

} // namespace script
} // namespace jit
} // namespace torch
