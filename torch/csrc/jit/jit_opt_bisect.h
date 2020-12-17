#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <memory>
#include <string>

namespace torch {
namespace jit {

struct Node;
struct Graph;

TORCH_API int64_t jit_bisect_level();

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_bisect_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_bisect_prefix(
    int64_t level,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_bisect_enabled(
    const char* pass_name,
    int64_t* current_counter);

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    int64_t level);

#define JIT_BISECT(level, ...) (is_bisect_enabled(__FILE__, level));

} // namespace jit
} // namespace torch
