#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExtensionBackendRegistration.h>

using namespace at;

int test_int;

Tensor empty_override(IntList size, const TensorOptions & options) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(), nullptr, false),
      FPGATensorId(),
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
  EXPECT_ANY_THROW(empty({5, 5}, at::kFPGA));
  register_extension_backend_op(
    Backend::FPGA,
    "empty(IntList size, TensorOptions options) -> Tensor", &empty_override);
  Tensor a = empty({5, 5}, at::kFPGA);
  ASSERT_EQ(test_int, 1);

  EXPECT_ANY_THROW(empty_like(a, at::kFPGA));
  register_extension_backend_op(
    Backend::FPGA,
    "empty_like(Tensor self, TensorOptions options) -> Tensor", &empty_like_override);
  Tensor b = empty_like(a, at::kFPGA);
  ASSERT_EQ(test_int, 2);

  EXPECT_ANY_THROW(add(a, b));
  register_extension_backend_op(
    Backend::FPGA,
    "add(Tensor self, Tensor other, Scalar alpha) -> Tensor", &add_override);
  add(a, b);
  ASSERT_EQ(test_int, 3);
}
