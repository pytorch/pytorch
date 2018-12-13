#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TypeExtensionBackend.h>

using namespace at;

using EmptyFnPtr = Tensor (*)(IntList, const TensorOptions &);

Tensor empty_override(IntList size, const TensorOptions & options) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(), nullptr, false),
      FPGATensorId(),
      false);
  return Tensor(std::move(tensor_impl));
}

using EmptyLikeFnPtr = Tensor (*)(const Tensor &, const TensorOptions &);

Tensor empty_like_override(const Tensor & self, const TensorOptions & options) {
  return self;
}

using AddFnPtr = Tensor (*)(const Tensor &, const Tensor&, Scalar);

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  return a;
}

TEST(BackendExtensionTest, TestRegisterOp) {
  EXPECT_ANY_THROW(empty({5, 5}, at::kFPGA));
  register_extension_backend_op(
    at::kFPGA,
    "empty(IntList size, TensorOptions options) -> Tensor", &empty_override);
  Tensor a = empty({5, 5}, at::kFPGA);

  EXPECT_ANY_THROW(empty_like(a, at::kFPGA));
  register_extension_backend_op(
    at::kFPGA,
    "empty_like(Tensor self, TensorOptions options) -> Tensor", &empty_like_override);
  Tensor b = empty_like(a, at::kFPGA);

  EXPECT_ANY_THROW(add(a, b));
  register_extension_backend_op(
    at::kFPGA,
    "add(Tensor self, Tensor other, Scalar alpha) -> Tensor", &add_override);
  add(a, b);
}
