#include <cstdlib>
#include <cstring>

namespace c10d::symmetric_memory {

static int getenv_nblocks() {
  static int num_blocks = -1; // Uninitialized
  if (num_blocks == -1) {
    const char* str = getenv("TORCH_SYMMMEM_NBLOCKS");
    if (str && strlen(str) > 0) {
      num_blocks = atoi(str);
    } else {
      num_blocks = -2; // User did not set env
    }
  }
  return num_blocks;
}

} // namespace c10d::symmetric_memory