#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <memory>
#include <string>
#include <unordered_map>

// `TorchScript` offers a simple logging facility that can enabled by setting an
// environment variable `PYTORCH_JIT_LOG_LEVEL`.

// Logging is enabled on a per file basis. To enable logging in
// `dead_code_elimination.cpp`, `PYTORCH_JIT_LOG_LEVEL` should be
// set to `dead_code_elimination.cpp` or, simply, to `dead_code_elimination`
// (i.e. `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination`).

// Multiple files can be logged by separating each file name with a colon `:` as
// in the following example,
// `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination:guard_elimination`

// There are 3 logging levels available for your use ordered by the detail level
// from lowest to highest.

// * `GRAPH_DUMP` should be used for printing entire graphs after optimization
// passes
// * `GRAPH_UPDATE` should be used for reporting graph transformations (i.e.
// node deletion, constant folding, etc)
// * `GRAPH_DEBUG` should be used for providing information useful for debugging
//   the internals of a particular optimization pass or analysis

// The default logging level is `GRAPH_DUMP` meaning that only `GRAPH_DUMP`
// statements will be enabled when one specifies a file(s) in
// `PYTORCH_JIT_LOG_LEVEL`.

// `GRAPH_UPDATE` can be enabled by prefixing a file name with an `>` as in
// `>alias_analysis`.
// `GRAPH_DEBUG` can be enabled by prefixing a file name with an `>>` as in
// `>>alias_analysis`.
// `>>>` is also valid and **currently** is equivalent to `GRAPH_DEBUG` as there
// is no logging level that is higher than `GRAPH_DEBUG`.

namespace torch {
namespace jit {

struct Node;
struct Graph;

enum class JitLoggingLevels {
  GRAPH_DUMP = 0,
  GRAPH_UPDATE,
  GRAPH_DEBUG,
};

class JitLoggingConfig {
 public:
  static JitLoggingConfig& getInstance() {
    static JitLoggingConfig instance;
    return instance;
  }
  JitLoggingConfig(JitLoggingConfig const&) = delete;
  void operator=(JitLoggingConfig const&) = delete;

 private:
  std::string logging_levels;
  std::unordered_map<std::string, size_t> files_to_levels;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  JitLoggingConfig() {
    const char* jit_log_level = std::getenv("PYTORCH_JIT_LOG_LEVEL");
    logging_levels.assign(jit_log_level == nullptr ? "" : jit_log_level);
    parse();
  }
  void parse();

 public:
  std::string getLoggingLevels() {
    return this->logging_levels;
  }
  void setLoggingLevels(std::string levels) {
    this->logging_levels = levels;
    parse();
  }

  std::unordered_map<std::string, size_t> getFilesToLevels() {
    return this->files_to_levels;
  }
};

std::string TORCH_API get_jit_logging_levels();

void TORCH_API set_jit_logging_levels(std::string level);

std::string TORCH_API getHeader(const Node* node);

std::string TORCH_API log_function(const std::shared_ptr<Graph>& graph);

TORCH_API ::torch::jit::JitLoggingLevels jit_log_level();

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_log_prefix(
    ::torch::jit::JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_enabled(
    const char* cfname,
    ::torch::jit::JitLoggingLevels level);

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    ::torch::jit::JitLoggingLevels level);

#define JIT_LOG(level, ...)                                  \
  if (is_enabled(__FILE__, level)) {                         \
    std::cerr << ::torch::jit::jit_log_prefix(               \
        level, __FILE__, __LINE__, ::c10::str(__VA_ARGS__)); \
  }

// tries to reconstruct original python source
#define SOURCE_DUMP(MSG, G)                       \
  JIT_LOG(                                        \
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP, \
      MSG,                                        \
      "\n",                                       \
      ::torch::jit::log_function(G));
// use GRAPH_DUMP for dumping graphs after optimization passes
#define GRAPH_DUMP(MSG, G) \
  JIT_LOG(                 \
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());
// use GRAPH_UPDATE for reporting graph transformations (i.e. node deletion,
// constant folding, CSE)
#define GRAPH_UPDATE(...) \
  JIT_LOG(::torch::jit::JitLoggingLevels::GRAPH_UPDATE, __VA_ARGS__);
// use GRAPH_DEBUG to provide information useful for debugging a particular opt
// pass
#define GRAPH_DEBUG(...) \
  JIT_LOG(::torch::jit::JitLoggingLevels::GRAPH_DEBUG, __VA_ARGS__);

#define GRAPH_DUMP_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_DUMP))
#define GRAPH_UPDATE_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_UPDATE))
#define GRAPH_DEBUG_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_DEBUG))
} // namespace jit
} // namespace torch
