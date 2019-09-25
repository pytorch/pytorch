
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {

// gets a string representation of a node header
// (e.g. outputs, a node kind and outputs)
std::string getHeader(const Node *node) {
  std::stringstream ss;
  node->print(ss, 0, {}, false, false, false, false);
  return ss.str();
}

static std::unordered_map<std::string, size_t>
parseJITLogOption(const char *option) {

  std::stringstream in_ss;
  in_ss << "function:";
  if (option) {
    in_ss << option;
  }

  std::unordered_map<std::string, size_t> files_to_levels;
  std::string line;
  while (std::getline(in_ss, line, ':')) {
    if (line.size() == 0) {
      continue;
    }

    auto index_at = line.find_last_of('>');
    auto begin_index = index_at == std::string::npos ? 0 : index_at + 1;
    size_t logging_level = index_at == std::string::npos ? 1 : index_at + 2;
    auto end_index = line.find_last_of('.') == std::string::npos
                         ? line.size()
                         : line.find_last_of('.');
    auto filename = line.substr(begin_index, end_index - begin_index);
    files_to_levels.insert({filename, logging_level});
  }

  return files_to_levels;
}

bool is_enabled(const char *cfname, JitLoggingLevels level) {

  static const char* c_log_level = std::getenv("PYTORCH_JIT_LOG_LEVEL");
  static const std::unordered_map<std::string, size_t> files_to_levels =
      parseJITLogOption(c_log_level);
  std::string fname{cfname};
  fname = c10::detail::StripBasename(fname);
  auto end_index = fname.find_last_of('.') == std::string::npos
                       ? fname.size()
                       : fname.find_last_of('.');
  auto fname_no_ext = fname.substr(0, end_index);

  auto it = files_to_levels.find(fname_no_ext);
  if (it == files_to_levels.end()) {
    return false;
  }

  return level <= static_cast<JitLoggingLevels>(it->second);
}

// Unfortunately, in `GraphExecutor` where `log_function` is invoked
// we won't have access to an original function, so we have to construct
// a dummy function to give to PythonPrint
std::string log_function(const std::shared_ptr<torch::jit::Graph> &graph) {
  torch::jit::Function func("source_dump", graph, nullptr);
  std::vector<at::Tensor> tensors;
  std::vector<c10::NamedTypePtr> deps;
  PythonPrint pp(tensors, deps, false);
  pp.printFunction(func);
  return pp.str();
}

std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str) {
  std::stringstream in_ss(in_str);
  std::stringstream out_ss;
  std::string line;
  while (std::getline(in_ss, line)) {
    out_ss << prefix << line << std::endl;
  }

  return out_ss.str();
}

std::string jit_log_prefix(
    JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str) {
  std::stringstream prefix_ss;
  prefix_ss << "[";
  prefix_ss << level << " ";
  prefix_ss << c10::detail::StripBasename(std::string(fn)) << ":";
  prefix_ss << std::setfill('0') << std::setw(3) << l;
  prefix_ss << "] ";

  return jit_log_prefix(prefix_ss.str(), in_str);
}

std::ostream& operator<<(std::ostream& out, JitLoggingLevels level) {
  switch (level) {
    case JitLoggingLevels::GRAPH_DUMP:
      out << "DUMP";
      break;
    case JitLoggingLevels::GRAPH_UPDATE:
      out << "UPDATE";
      break;
    case JitLoggingLevels::GRAPH_DEBUG:
      out << "DEBUG";
      break;
    default:
      TORCH_INTERNAL_ASSERT("Invalid level");
  }

  return out;
}

} // namespace jit
} // namespace torch
