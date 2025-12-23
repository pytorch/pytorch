#include <c10/util/env.h>

namespace c10d::symmetric_memory {

static int getenv_nblocks() {
  static int num_blocks = -1; // Uninitialized
  if (num_blocks == -1) {
    auto str = c10::utils::get_env("TORCH_SYMMMEM_NBLOCKS");
    if (str.has_value()) {
      num_blocks = std::stoi(str.value());
    } else {
      num_blocks = -2; // Not set
    }
  }
  return num_blocks;
}

} // namespace c10d::symmetric_memory