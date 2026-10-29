#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "device/jit/jit_utils.hpp"

namespace nccl_ep::jit {

// Only jit_runtime.cc redirects env_value here; other settings retain upstream
// behavior. Header paths must belong to this EP copy, even if nccl4py is loaded.
std::string torch_nccl_ep_jit_env_value(const char* name) {
  const std::string_view key(name);
  if (key != "NCCL_EP_JIT_SOURCE_DIR" && key != "NCCL_EP_JIT_BUILD_INCLUDE_DIR") {
    return env_value(name);
  }
  static const auto home = [] {
    static const int module_anchor = 0;
    Dl_info info{};
    if (dladdr(&module_anchor, &info) == 0 || info.dli_fname == nullptr) {
      // @allow-raw-throw: This external static library does not depend on c10.
      throw std::runtime_error("Cannot locate PyTorch's NCCL EP JIT headers");
    }
    return std::filesystem::canonical(info.dli_fname).parent_path() / "share" /
        "nccl_ep";
  }();
  const auto include_dir = home / "include";
  return (key == "NCCL_EP_JIT_SOURCE_DIR" ? include_dir / "nccl_ep" : include_dir)
      .string();
}

} // namespace nccl_ep::jit
