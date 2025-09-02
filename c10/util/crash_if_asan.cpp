#include <c10/util/crash_if_asan.h>

namespace c10 {
int crash_if_asan(int arg) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  volatile char x[3];
  x[arg] = 0;
  return x[0];
}
} // namespace c10
