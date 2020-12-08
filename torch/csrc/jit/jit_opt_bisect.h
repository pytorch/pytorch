#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <memory>
#include <string>

namespace torch {
namespace jit {

struct Node;
struct Graph;

enum class JitBisectionLevels {
  PYTORCH_OPT_LIMIT = INT64_MAX,
};

std::string TORCH_API getHeader(const Node* node);

std::string TORCH_API bisect_function(const std::shared_ptr<Graph>& graph);

TORCH_API ::torch::jit::JitBisectionLevels jit_bisect_level();

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_log_prefix(
    ::torch::jit::JitBisectionLevels level,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_enabled(
    const char* cfname,
    ::torch::jit::JitBisectionLevels level);

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    ::torch::jit::JitBisectionLevels level);

#define JIT_OPT_BISECT(level, ...)                                  \
  if (is_enabled(__FILE__, level)) {                         \
    std::cerr << ::torch::jit::jit_log_prefix(               \
        level, __FILE__, __LINE__, ::c10::str(__VA_ARGS__)); \
  }



} // namespace jit
} // namespace torch
