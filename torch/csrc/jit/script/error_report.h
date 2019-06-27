#pragma once

#include <torch/csrc/jit/script/tree.h>
#include <c10/util/Optional.h>

namespace torch {
namespace jit {
namespace script {


// List of source ranges representing the current stack of calls being compiled
inline std::vector<SourceRange>& get_stack() {
  static std::vector<SourceRange> call_stack;
  return call_stack;
}

inline std::vector<std::string>& get_fn_stack() {
  static std::vector<std::string> fns;
  return fns;
}

inline void push_call(const SourceRange& range) {
  get_stack().push_back(range);
}

inline void pop_call() {
  get_stack().pop_back();
}

inline void push_function(const std::string& name) {
  get_fn_stack().push_back(name);
}

inline void pop_function() {
  get_fn_stack().pop_back();
}

struct ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e)
      : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

  ErrorReport() : context(c10::nullopt) {}
  explicit ErrorReport(SourceRange r)
      : context(std::move(r)) {}
  explicit ErrorReport(const TreeRef& tree) : ErrorReport(tree->range()) {}
  explicit ErrorReport(const Token& tok) : ErrorReport(tok.range) {}
  const char* what() const noexcept override {
    std::stringstream msg;
    msg << "\n" << ss.str();
    if (context) {
      msg << ":\n";
      context->highlight(msg);
    } else {
      msg << ".\n";
    }

    if (get_stack().size() > 0) {
      // Print the calling stack to show why this function was being compiled,
      // skip the first entry in the list since it's the current function
      auto range_it = get_stack().rbegin() + 1;
      auto names_it = get_fn_stack().rbegin();
      for (;
           range_it != get_stack().rend() && names_it != get_fn_stack().rend();
           ++range_it, ++names_it) {
        msg << "\n";
        msg << "'" << *names_it
            << "' is being compiled since it was called from '"
            << *(names_it + 1) << "'\n";
        range_it->highlight(msg);
      }
    }

    the_message = msg.str();
    return the_message.c_str();
  }

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
