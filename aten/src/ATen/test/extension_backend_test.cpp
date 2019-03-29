#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExtensionBackendRegistration.h>

using namespace at;

static int test_int;

Tensor empty_override(IntArrayRef size, const TensorOptions & options) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 1)), nullptr, false),
      MSNPUTensorId(),
      false);
  return Tensor(std::move(tensor_impl));
}

Tensor empty_like_override(const Tensor & self, const TensorOptions & options) {
  test_int = 2;
  return self;
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 3;
  return a;
}

TEST(BackendExtensionTest, TestRegisterOp) {
  EXPECT_ANY_THROW(empty({5, 5}, at::kMSNPU));
  register_extension_backend_op(
    Backend::MSNPU,
    "empty(IntArrayRef size, TensorOptions options) -> Tensor", &empty_override);
  Tensor a = empty({5, 5}, at::kMSNPU);
  ASSERT_EQ(a.device().type(), at::kMSNPU);
  ASSERT_EQ(a.device().index(), 1);
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  ASSERT_EQ(test_int, 1);

  EXPECT_ANY_THROW(empty_like(a, at::kMSNPU));
  register_extension_backend_op(
    Backend::MSNPU,
    "empty_like(Tensor self, TensorOptions options) -> Tensor", &empty_like_override);
  Tensor b = empty_like(a, at::kMSNPU);
  ASSERT_EQ(test_int, 2);

  EXPECT_ANY_THROW(add(a, b));
  register_extension_backend_op(
    Backend::MSNPU,
    "add(Tensor self, Tensor other, Scalar alpha) -> Tensor", &add_override);
  add(a, b);
  ASSERT_EQ(test_int, 3);

  // Ensure that non-MSNPU operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);

  // Attempt to register on a schema that has already has a function
  EXPECT_ANY_THROW(
    register_extension_backend_op(
      Backend::MSNPU,
      "empty(IntArrayRef size, TensorOptions options) -> Tensor", &empty_override)
  );
}
