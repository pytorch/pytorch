#include "ATen/ATen.h"
#include "ATen/cudnn/Descriptors.h"
#include "ATen/cudnn/Handles.h"
#include "test_assert.h"

using namespace at;
using namespace at::native;

int main() {
#if CUDNN_VERSION < 7000
  auto handle = getCudnnHandle();
  DropoutDescriptor desc1, desc2;
  desc1.initialize_rng(at::CUDA(kByte), handle, 0.5, 42);
  desc2.set(handle, 0.5, desc1.state);

  ASSERT(desc1.desc()->dropout == desc2.desc()->dropout);
  ASSERT(desc1.desc()->nstates == desc2.desc()->nstates);
  ASSERT(desc1.desc()->states == desc2.desc()->states);
#endif
  std::cerr << "DONE" << std::endl;
  return 0;
}
