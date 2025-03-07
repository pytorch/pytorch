#include "torch/csrc/runtime/common/ConfigUtils.h"
#include <string.h>

namespace torch::runtime {
std::optional<std::string> maybeGetEnv(std::string_view envVar) {
  const char* env = getenv(envVar.data());
  if (env == nullptr || strlen(env) == 0) {
    return std::nullopt;
  }
  return std::string(env);
}

} // namespace torch::runtime
