#include <torch/csrc/jit/jit_log.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir.h>
#include <c10/util/StringUtil.h>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace torch {
namespace jit {

JitLoggingLevels jit_log_level() {
  static const char* c_log_level = std::getenv("PYTORCH_JIT_LOG_LEVEL");
  static const JitLoggingLevels log_level = c_log_level
      ? static_cast<JitLoggingLevels>(std::atoi(c_log_level))
      : JitLoggingLevels::OFF;
  return log_level;
}

template <typename T>
void logArguments(std::ostream& ss, T t) {
  ss << t;
}

template <>
void logArguments<Value*>(std::ostream& ss, Value* v) {
  ss << v->debugName();
}

template <typename T, typename... Rest>
void logArguments(std::ostream& ss, T t, Rest... rest) {
  logArguments(ss, t);
  ss << ", ";
  logArguments(ss, rest...);
}

template <typename... Args>
std::string logNode(Node* node, Args... args) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < node->outputs().size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    node->outputs()[i]->debugName();
  }
  ss << ") = ";
  ss << node->kind().toQualString() << "(";
  logArguments(ss, args...);
  ss << ")";
  return ss.str();
}

std::string debugValueOrDefault(const Node* n) {
  return n->outputs().size() > 0 ? n->outputs().at(0)->debugName() : "n/a";
}

std::string jit_log_prefix(
    JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str) {
  std::stringstream in_ss(in_str);
  std::stringstream out_ss(in_str);
  std::string line;
  while (std::getline(in_ss, line, '\n')) {
    out_ss << "[";
    out_ss << level << " ";
    out_ss << c10::detail::StripBasename(std::string(fn)) << ":";
    out_ss << std::setfill('0') << std::setw(3) << l;
    out_ss << "] ";
    out_ss << line << std::endl;
  }

  return out_ss.str();
}

std::ostream& operator<<(std::ostream& out, JitLoggingLevels level) {
  switch (level) {
    case JitLoggingLevels::OFF:
      TORCH_INTERNAL_ASSERT("UNREACHABLE");
      break;
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
