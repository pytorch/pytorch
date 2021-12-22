#include <c10/core/free_cpu.h>

namespace c10 {

void free_cpu(void* data) {
#ifdef _MSC_VER
  _aligned_free(data);
#else
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(data);
#endif
}

} // namespace c10
