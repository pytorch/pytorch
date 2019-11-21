#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/jit/script/tree.h>

namespace torch {
namespace jit {
namespace script {

struct Call {
  std::string fn_name;
  c10::optional<SourceRange> caller_range;
};

struct CAFFE2_API ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e);

  ErrorReport();
  explicit ErrorReport(SourceRange r);
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}

  const char* what() const noexcept override;

  struct CAFFE2_API CallStack {
    // These functions are used to report why a function was being compiled
    // (i.e. what was the call stack of user functions at compilation time that
    // led to this error)
    CallStack(const std::string& name);
    ~CallStack();

    // Change the range that is relevant for the current function (i.e. after
    // each successful expression compilation, change it to the next expression)
    static void update_pending_range(const SourceRange& range);
  };

 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);

  mutable std::stringstream ss;
  c10::optional<SourceRange> context;
  mutable std::string the_message;
  std::vector<Call> error_stack;
};

template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}

} // namespace script
} // namespace jit
} // namespace torch
