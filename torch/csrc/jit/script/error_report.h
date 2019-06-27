#pragma once

#include <torch/csrc/jit/script/tree.h>
#include <c10/util/Optional.h>

namespace torch {
namespace jit {
namespace script {

void push_call(const SourceRange& range);
void pop_call();
void push_function(const std::string& name);
void pop_function();

struct CAFFE2_API ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e)
      : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

  ErrorReport() : context(c10::nullopt) {}
  explicit ErrorReport(SourceRange r)
      : context(std::move(r)) {}
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}
  const char* what() const noexcept override;

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
