#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <memory>
#include <string>
#include <unordered_map>

// `TorchScript` offers a simple bisection utility function
// that can be configured through environment variable `PYTORCH_OPT_LIMIT`.
// The purpose is to limit how many optimization you can make per pass.
// This is useful for debugging performance pass related pass.

// Bisecting is enabled on a per file basis (hence per pass). For example, in
// `constant_propagation.cpp`, `PYTORCH_OPT_LIMIT` should be
// set to `constant_propagation.cpp=<opt_limt>` or, simply, to
// `constant_propagation.cpp=<opt_limit>` where <opt_limit> is the number of
// optimizations you want to make for the pass. (i.e.
// `PYTORCH_OPT_LIMIT="constant_propagation=<opt_limit>"`).

// Multiple files can be logged by separating each file name with a colon `:` as
// in the following example,
// `PYTORCH_OPT_LIMIT="constant_propagation=<opt_limit>:dead_code_elimination=<opt_limit>"`

// You can also log the certain pass has completed the number of optimizations
// by calling JIT_BISECT_LOG(<msg>) where <msg> is a custom message. Format is
// <msg>:[<pass_name>:<line_number>:current_counter=<current_counter]

namespace torch {
namespace jit {

struct Node;
struct Graph;

TORCH_API int64_t jit_bisect_level();

static std::unordered_map<std::string, int64_t> passes_to_current_counter;

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_bisect_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_bisect_prefix(
    const std::string& msg,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_bisect_enabled(const char* pass_name);

TORCH_API std::ostream& operator<<(std::ostream& out, int64_t level);

#define JIT_BISECT(...) (is_bisect_enabled(__FILE__));

#define JIT_BISECT_LOG(MSG, ...)                           \
  if (is_bisect_enabled(__FILE__)) {                       \
    std::cerr << jit_bisect_prefix(                        \
        MSG, __FILE__, __LINE__, ::c10::str(__VA_ARGS__)); \
  }

} // namespace jit
} // namespace torch
