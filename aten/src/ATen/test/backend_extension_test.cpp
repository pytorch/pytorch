#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/FPGAType.h>

using namespace at;
using AddFnPtr = Tensor (*)(const Tensor &, const Tensor&, Scalar);

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  return a;
}

TEST(BackendExtensionTest, TestRegisterOp) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(), nullptr, false),
      FPGATensorId(),
      false);
  auto tensor = Tensor(std::move(tensor_impl));
  EXPECT_ANY_THROW(add(tensor, tensor));

  FPGAType::FPGATypeDispatch<AddFnPtr>::register_function(
    "add(Tensor self, Tensor other, Scalar alpha) -> Tensor", &add_override);
  add(tensor, tensor);
}
