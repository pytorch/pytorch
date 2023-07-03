#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/serialization/python_print.h>

namespace torch {
namespace jit {

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
  std::ostream* out;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  JitLoggingConfig() {
    const char* jit_log_level = std::getenv("PYTORCH_JIT_LOG_LEVEL");
    logging_levels.assign(jit_log_level == nullptr ? "" : jit_log_level);
    out = &std::cerr;
    parse();
  }
  void parse();

 public:
  std::string getLoggingLevels() const {
    return this->logging_levels;
  }
  void setLoggingLevels(std::string levels) {
    this->logging_levels = std::move(levels);
    parse();
  }

  const std::unordered_map<std::string, size_t>& getFilesToLevels() const {
    return this->files_to_levels;
  }

  void setOutputStream(std::ostream& out_stream) {
    this->out = &out_stream;
  }

  std::ostream& getOutputStream() {
    return *(this->out);
  }
};

std::string get_jit_logging_levels() {
  return JitLoggingConfig::getInstance().getLoggingLevels();
}

void set_jit_logging_levels(std::string level) {
  JitLoggingConfig::getInstance().setLoggingLevels(std::move(level));
}

void set_jit_logging_output_stream(std::ostream& stream) {
  JitLoggingConfig::getInstance().setOutputStream(stream);
}

std::ostream& get_jit_logging_output_stream() {
  return JitLoggingConfig::getInstance().getOutputStream();
}

// gets a string representation of a node header
// (e.g. outputs, a node kind and outputs)
std::string getHeader(const Node* node) {
  std::stringstream ss;
  node->print(ss, 0, {}, false, false, false, false);
  return ss.str();
}

void JitLoggingConfig::parse() {
  std::stringstream in_ss;
  in_ss << "function:" << this->logging_levels;

  files_to_levels.clear();
  std::string line;
  while (std::getline(in_ss, line, ':')) {
    if (line.empty()) {
      continue;
    }

    auto index_at = line.find_last_of('>');
    auto begin_index = index_at == std::string::npos ? 0 : index_at + 1;
    size_t logging_level = index_at == std::string::npos ? 0 : index_at + 1;
    auto end_index = line.find_last_of('.') == std::string::npos
        ? line.size()
        : line.find_last_of('.');
    auto filename = line.substr(begin_index, end_index - begin_index);
    files_to_levels.insert({filename, logging_level});
  }
}

bool is_enabled(const char* cfname, JitLoggingLevels level) {
  const auto& files_to_levels =
      JitLoggingConfig::getInstance().getFilesToLevels();
  std::string fname{cfname};
  fname = c10::detail::StripBasename(fname);
  const auto end_index = fname.find_last_of('.') == std::string::npos
      ? fname.size()
      : fname.find_last_of('.');
  const auto fname_no_ext = fname.substr(0, end_index);

  const auto it = files_to_levels.find(fname_no_ext);
  if (it == files_to_levels.end()) {
    return false;
  }

  return level <= static_cast<JitLoggingLevels>(it->second);
}

// Unfortunately, in `GraphExecutor` where `log_function` is invoked
// we won't have access to an original function, so we have to construct
// a dummy function to give to PythonPrint
std::string log_function(const std::shared_ptr<torch::jit::Graph>& graph) {
  torch::jit::GraphFunction func("source_dump", graph, nullptr);
  std::vector<at::IValue> constants;
  PrintDepsTable deps;
  PythonPrint pp(constants, deps);
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
      TORCH_INTERNAL_ASSERT(false, "Invalid level");
  }

  return out;
}

} // namespace jit
} // namespace torch
