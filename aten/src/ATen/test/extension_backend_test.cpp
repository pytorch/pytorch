#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExtensionBackendRegistration.h>

using namespace at;

static int test_int;

Tensor get_dummy_tensor() {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 1)), nullptr, false),
      MSNPUTensorId(),
      false);
  return Tensor(std::move(tensor_impl));
}

Tensor empty_override(IntList size, const TensorOptions & options) {
  test_int = 1;
  return get_dummy_tensor();
}

Tensor empty_like_override(const Tensor & self, const TensorOptions & options) {
  test_int = 2;
  return self;
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 3;
  return a;
}

Tensor & s_copy__override(Tensor & self, const Tensor & src, bool non_blocking) {
  test_int = 4;
  return self;
}

Tensor _s_copy_from_override(const Tensor & self, const Tensor & dst, bool non_blocking) {
  test_int = 5;
  return get_dummy_tensor();
}

Tensor expand_override(const Tensor & self, IntList size, bool implicit) {
  return get_dummy_tensor();
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

  // copy has a double dispatch that needs to be tested
  register_extension_backend_op(
    Backend::MSNPU,
    "s_copy_(Tensor self, Tensor src, bool non_blocking) -> Tensor", &s_copy__override);
  register_extension_backend_op(
    Backend::MSNPU,
    "_s_copy_from(Tensor self, Tensor dst, bool non_blocking) -> Tensor", &_s_copy_from_override);
  // expand is needed for copy_
  register_extension_backend_op(
    Backend::MSNPU,
    "expand(Tensor self, IntList size, bool implicit) -> Tensor",
    &expand_override);
  Tensor cpu_tensor = empty({}, at::kCPU);
  Tensor msnpu_tensor = empty({}, at::kMSNPU);
  msnpu_tensor.copy_(cpu_tensor, false);
  ASSERT_EQ(test_int, 4);
  cpu_tensor.copy_(msnpu_tensor, false);
  ASSERT_EQ(test_int, 5);

  // Ensure that non-MSNPU operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);

  // Attempt to register on a schema that has already has a function
  EXPECT_ANY_THROW(
    register_extension_backend_op(
      Backend::MSNPU,
      "empty(IntArrayRef size, TensorOptions options) -> Tensor", &empty_override)
  );

  // Invalid registration: bad operator name
  EXPECT_ANY_THROW(
    register_extension_backend_op(
      Backend::MSNPU,
      "help(IntList size, TensorOptions options) -> Tensor", &add_override)
  );

  // Invalid registration: valid operator name but invalid schema
  EXPECT_ANY_THROW(
    register_extension_backend_op(
      Backend::MSNPU,
      "zeros(TensorOptions options) -> Tensor", &add_override)
  );

  // Invalid registration: valid schema but mismatched function pointer type
  EXPECT_ANY_THROW(
    register_extension_backend_op(
      Backend::MSNPU,
      "zeros(IntList size, TensorOptions options) -> Tensor", &add_override)
  );
}
