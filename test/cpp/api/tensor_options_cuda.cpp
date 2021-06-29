#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/cuda.h>

// NB: This file is compiled even in CPU build (for some reason), so
// make sure you don't include any CUDA only headers.

using namespace at;

// TODO: This might be generally helpful aliases elsewhere.
at::Device CPUDevice() {
  return at::Device(at::kCPU);
}
at::Device CUDADevice(DeviceIndex index) {
  return at::Device(at::kCUDA, index);
}

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                      \
  ASSERT_EQ(options.device().type(), Device((device_), (index_)).type()); \
  ASSERT_TRUE(                                                                \
      options.device().index() == Device((device_), (index_)).index());       \
  ASSERT_EQ(typeMetaToScalarType(options.dtype()), (type_));                  \
  ASSERT_TRUE(options.layout() == (layout_))

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)                \
  ASSERT_EQ(tensor.device().type(), Device((device_), (index_)).type());   \
  ASSERT_EQ(tensor.device().index(), Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.scalar_type(), (type_));                                \
  ASSERT_TRUE(tensor.options().layout() == (layout_))

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorOptionsTest, ConstructsWellFromCUDATypes_CUDA) {
  auto options = CUDA(kFloat).options();
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kStrided);

  options = CUDA(kInt).options();
  REQUIRE_OPTIONS(kCUDA, -1, kInt, kStrided);

  options = getDeprecatedTypeProperties(Backend::SparseCUDA, kFloat).options();
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kSparse);

  options = getDeprecatedTypeProperties(Backend::SparseCUDA, kByte).options();
  REQUIRE_OPTIONS(kCUDA, -1, kByte, kSparse);

  // NOLINTNEXTLINE(bugprone-argument-comment,cppcoreguidelines-avoid-magic-numbers)
  options = CUDA(kFloat).options(/*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kStrided);

  options =
      // NOLINTNEXTLINE(bugprone-argument-comment,cppcoreguidelines-avoid-magic-numbers)
      getDeprecatedTypeProperties(Backend::SparseCUDA, kFloat).options(/*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kSparse);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TensorOptionsTest, ConstructsWellFromCUDATensors_MultiCUDA) {
  auto options = empty(5, device(kCUDA).dtype(kDouble)).options();
  REQUIRE_OPTIONS(kCUDA, 0, kDouble, kStrided);

  options = empty(5, getDeprecatedTypeProperties(Backend::SparseCUDA, kByte)).options();
  REQUIRE_OPTIONS(kCUDA, 0, kByte, kSparse);

  if (torch::cuda::device_count() > 1) {
    Tensor tensor;
    {
      DeviceGuard guard(CUDADevice(1));
      tensor = empty(5, device(kCUDA));
    }
    options = tensor.options();
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

    {
      DeviceGuard guard(CUDADevice(1));
      tensor = empty(5, device(kCUDA).layout(kSparse));
    }
    options = tensor.options();
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kSparse);
  }
}
