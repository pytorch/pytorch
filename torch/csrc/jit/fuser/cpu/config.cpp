#include "torch/csrc/jit/fuser/cpu/config.h"

#include "torch/csrc/jit/code_template.h"

#include <cstdlib>
#include <string>

namespace torch { namespace jit { namespace fuser { namespace cpu {

CompilerConfig& getConfig() {
  static CompilerConfig config;
  return config;
}

static const std::string check_exists_string = "which '${program}' > /dev/null";
static bool programExists(const std::string& program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return 0 == system(cmd.c_str());
}

CompilerConfig::CompilerConfig() {
  const char* cxx_env = getenv("CXX");
  if (cxx_env != nullptr) {
    cxx = cxx_env;
  }

  if (!programExists(cxx)) {
    cxx = "";
  }
  
  const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
  debug = debug_env && atoi(debug_env) != 0;
}

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
