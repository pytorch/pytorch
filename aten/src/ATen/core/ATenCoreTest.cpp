#include <ATen/core/ATenCoreTest.h>

namespace at {

static int CoreTestGlobal = 0;
int CoreTest() {
  return CoreTestGlobal++;
}

} // namespace at
