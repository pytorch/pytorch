#pragma once
#include <memory>
#include <string>
#include <torch/csrc/WindowsTorchApiMacro.h>

// To enable logging please set(export) PYTORCH_JIT_LOG_LEVEL to
// the ordinal value of one of the following logging levels: 1 for GRAPH_DUMP,
// 2 for GRAPH_UPDATE, 3 for GRAPH_DEBUG.
// * Use GRAPH_DUMP for dumping graphs after optimization passes
// * Use GRAPH_UPDATE for reporting graph transformations (i.e. node deletion,
//   constant folding, CSE)
// * Use GRAPH_DEBUG to provide information useful for debugging
//   the internals of a particular optimization pass or analysis

namespace torch {
namespace jit {

struct Node;
struct Graph;

enum class JitLoggingLevels {
  GRAPH_DUMP = 0,
  GRAPH_UPDATE,
  GRAPH_DEBUG,
};

std::string debugValueOrDefault(const Node* n);

std::string TORCH_API log_function(const std::shared_ptr<Graph> &graph);

TORCH_API JitLoggingLevels jit_log_level();

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_log_prefix(
    JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_enabled(const char *cfname, JitLoggingLevels level);

TORCH_API std::ostream& operator<<(std::ostream& out, JitLoggingLevels level);

#define JIT_LOG(level, ...)                                                    \
  if (is_enabled(__FILE__, level)) {                                           \
    std::cerr << jit_log_prefix(level, __FILE__, __LINE__,                     \
                                ::c10::str(__VA_ARGS__));                      \
  }

// tries to reconstruct original python source
#define SOURCE_DUMP(MSG, G)                                                    \
  JIT_LOG(JitLoggingLevels::GRAPH_DUMP, MSG, "\n", log_function(G));
// use GRAPH_DUMP for dumping graphs after optimization passes
#define GRAPH_DUMP(MSG, G) \
  JIT_LOG(JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());
// use GRAPH_UPDATE for reporting graph transformations (i.e. node deletion,
// constant folding, CSE)
#define GRAPH_UPDATE(...) JIT_LOG(JitLoggingLevels::GRAPH_UPDATE, __VA_ARGS__);
// use GRAPH_DEBUG to provide information useful for debugging a particular opt
// pass
#define GRAPH_DEBUG(...) JIT_LOG(JitLoggingLevels::GRAPH_DEBUG, __VA_ARGS__);
} // namespace jit
} // namespace torch
