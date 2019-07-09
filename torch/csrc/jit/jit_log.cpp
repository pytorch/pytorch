#include <torch/csrc/jit/jit_log.h>
#include <c10/util/Exception.h>
#include <cstdlib>
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

std::string jit_log_prefix(JitLoggingLevels level, const std::string& in_str) {
  std::stringstream in_ss(in_str);
  std::stringstream out_ss(in_str);
  std::string line;

  while (std::getline(in_ss, line, '\n')) {
    out_ss << level << " " << line << std::endl;
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
