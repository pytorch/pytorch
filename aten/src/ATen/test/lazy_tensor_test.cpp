#include <gtest/gtest.h>

#include <ATen/ATen.h>

void LazyTensorTest(c10::DispatchKey dispatch_key, at::DeviceType device_type) {
  auto tensor_impl =
      c10::make_intrusive<c10::TensorImpl, c10::UndefinedTensorImpl>(
          dispatch_key,
          caffe2::TypeMeta::Make<float>(),
          at::Device(device_type, 0));
  at::Tensor t(std::move(tensor_impl));
  ASSERT_TRUE(t.device() == at::Device(device_type, 0));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(XlaTensorTest, TestNoStorage) {
  LazyTensorTest(at::DispatchKey::XLA, at::DeviceType::XLA);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LazyTensorTest, TestNoStorage) {
  LazyTensorTest(at::DispatchKey::Lazy, at::DeviceType::Lazy);
}
