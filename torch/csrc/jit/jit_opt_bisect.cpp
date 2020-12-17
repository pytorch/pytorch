#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_opt_bisect.h>
#include <torch/csrc/jit/serialization/python_print.h>

namespace torch {
namespace jit {

static int parseOptLimit(const std::string& opt_limit) {
  // TODO bad code
  try {
    int64_t n = std::stoi(opt_limit);
    return n;
  } catch (...) {
    return -1;
  }
}

static std::unordered_map<std::string, int64_t> parseJITBisectOption(
    const char* option) {
  std::stringstream in_ss;
  in_ss << "function:";
  if (option) {
    in_ss << option;
  }

  std::unordered_map<std::string, int64_t> passes_to_opt_limits;
  std::string line;
  while (std::getline(in_ss, line, ':')) {
    if (line.size() == 0) {
      continue;
    }
    auto index_at = line.find_last_of('=');
    auto pass_name = line.substr(0, index_at);
    pass_name = c10::detail::ExcludeFileExtension(pass_name);
    auto opt_limit = parseOptLimit(line.substr(index_at+1));
    passes_to_opt_limits.insert({pass_name, opt_limit});
  }

  return passes_to_opt_limits;
}

bool is_bisect_enabled(const char* pass_name, int64_t* current_counter) {
  static const char* opt_limit = std::getenv("PYTORCH_OPT_LIMIT");
  static const std::unordered_map<std::string, int64_t> passes_to_opt_limits =
      parseJITBisectOption(opt_limit);
  std::string pass{pass_name};
  pass = c10::detail::StripBasename(pass);
  pass = c10::detail::ExcludeFileExtension(pass);
  auto it = passes_to_opt_limits.find(pass);
  if (it == passes_to_opt_limits.end()) {
    return false;
  }

  auto opt_limit_for_file = it->second;
  if (*current_counter >= opt_limit_for_file) {
    return false;
  }

  (*current_counter)++;
  return true;
}

std::string jit_bisect_prefix(
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

std::string jit_bisect_prefix(
    int64_t level,
    const char* pn,
    int l,
    const std::string& in_str) {
  std::stringstream prefix_ss;
  prefix_ss << "[";
  prefix_ss << std::to_string(level) << " ";
  prefix_ss << c10::detail::StripBasename(std::string(pn)) << ":";
  prefix_ss << std::setfill('0') << std::setw(3) << l;
  prefix_ss << "] ";

  return jit_bisect_prefix(prefix_ss.str(), in_str);
}

std::ostream& operator<<(std::ostream& out, int64_t level) {
  out << "OPT_LIMIT:" << std::to_string(level);
  return out;
}

} // namespace jit
} // namespace torch
