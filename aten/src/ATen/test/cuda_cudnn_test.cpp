#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>

using namespace at;
using namespace at::native;

TEST(CUDNNTest, CUDNNTestCUDA) {
  if (!at::cuda::is_available()) return;
  manual_seed(123);

#if CUDNN_VERSION < 7000
  auto handle = getCudnnHandle();
  DropoutDescriptor desc1, desc2;
  desc1.initialize_rng(handle, 0.5, 42, TensorOptions().device(DeviceType::CUDA).dtype(kByte));
  desc2.set(handle, 0.5, desc1.state);
  bool isEQ;
  isEQ = (desc1.desc()->dropout == desc2.desc()->dropout);
  ASSERT_TRUE(isEQ);
  isEQ = (desc1.desc()->nstates == desc2.desc()->nstates);
  ASSERT_TRUE(isEQ);
  isEQ = (desc1.desc()->states == desc2.desc()->states);
  ASSERT_TRUE(isEQ);
#endif
}
