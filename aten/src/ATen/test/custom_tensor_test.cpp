#include "gtest/gtest.h"
#include "ATen/ATen.h"

#include <torch/csrc/autograd/variable.h>

struct TestTensor {
  at::Tensor c;
};

using namespace at;
void to_test_tensor(const c10::FunctionSchema&, torch::jit::Stack* stack) {
  (void)torch::jit::pop(*stack); // tensor type, assume it's correct
  auto self = torch::jit::pop(*stack).toTensor();
  auto tag = GET_CUSTOM_TENSOR_TAG("test_tensor");
  auto test_tensor = std::make_shared<TestTensor>();
  test_tensor->c = self;
  auto tensor = at::detail::make_tensor<CustomTensorImpl>(tag, 
      std::static_pointer_cast<void>(test_tensor),
      self.type_set(),
      self.dtype(),
      self.device()
  );
  auto t = torch::autograd::make_variable(tensor);
  torch::jit::push(*stack, t);
}

void from_test_tensor(const c10::FunctionSchema&, torch::jit::Stack* stack) {
  auto self = torch::jit::pop(*stack).toTensor();
  auto cst = static_cast<CustomTensorImpl*>(self.unsafeGetTensorImpl());
  auto test_tensor = std::static_pointer_cast<TestTensor>(cst->storage());
  auto tensor = test_tensor->c;
  auto t = torch::autograd::make_variable(tensor);
  torch::jit::push(*stack, t);
}

REGISTER_CUSTOM_TENSOR_METHOD(test_tensor, to_custom, to_test_tensor);
REGISTER_CUSTOM_TENSOR_METHOD(test_tensor, from_custom, from_test_tensor);

TEST(TestCustomTensor, Basic) {
  at::Tensor t = at::randn({2,2});
  auto t_s = t.to_custom("test_tensor");
  auto k = t_s.from_custom();
}
