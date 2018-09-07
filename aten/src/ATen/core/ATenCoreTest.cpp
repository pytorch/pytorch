#include <ATen/core/ATenCoreTest.h>
#include <ATen/core/Tensor.h>

namespace at {

static int CoreTestGlobal = 0;
int CoreTest() {
  Tensor x;
  return CoreTestGlobal++;
}

} // namespace at
