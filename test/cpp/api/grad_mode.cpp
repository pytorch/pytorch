#include <torch/script.h>
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

using namespace torch::autograd;
using namespace torch::test;

TEST(GradModeTest, TestRequiresGradFunctionalOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor func_out = c * c;
    ASSERT_FALSE(func_out.requires_grad());
    ASSERT_TRUE(func_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradInplaceOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    c.mul_(2);
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

TEST(GradModeTest, TestRequiresGradViewOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor view_out = c.view({2, 3});
    ASSERT_EQ(view_out.requires_grad(), requires_grad);
    ASSERT_TRUE(view_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradViewOpExiting) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      torch::AutoGradMode mode(false);
      view_out = a.view({2, 3});  // go through kernels: VariableType, InplaceOrView, CPU
      assert_tensor_creation_meta(view_out, torch::autograd::CreationMeta::NO_GRAD_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());
    }

    tmp = view_out * view_out;
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    if (requires_grad) {
      ASSERT_THROWS_WITH(view_out.mul_(2),  // go through kernels: VariableType, InplaceOrView, CPU
        "A view was created in no_grad mode and is being modified inplace")
    } else {
        view_out.mul_(2);
    }

    tmp = view_out.view({2, 3});
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
    assert_tensor_creation_meta(tmp, torch::autograd::CreationMeta::NO_GRAD_MODE);
  }
}
