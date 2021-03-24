#include <torch/script.h>
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

using namespace torch::autograd;
namespace {
  torch::Tensor functional_op(torch::Tensor& x) {
    return x + x;
  }

  void inplace_op(torch::Tensor& x) {
    x.add_(1);
  }

  torch::Tensor view_op(torch::Tensor& x) {
    return x.view({2, 3});
  }

  void assert_tensor_creation_meta(torch::Tensor& x, torch::autograd::CreationMeta creation_meta) {
    ASSERT_EQ(static_cast<torch::autograd::DifferentiableViewMeta*>(x.unsafeGetTensorImpl()->autograd_meta())->get_creation_meta(), creation_meta);
  }

}

TEST(GradModeTest, TestRequiresGradFunctionalOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor func_out = functional_op(c);
    ASSERT_FALSE(func_out.requires_grad());
    ASSERT_TRUE(func_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradInplaceOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    inplace_op(c);
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

TEST(GradModeTest, TestRequiresGradViewOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor view_out = view_op(c);
    ASSERT_EQ(view_out.requires_grad(), requires_grad);
    ASSERT_TRUE(view_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradViewOpExiting) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s + 2;
    torch::Tensor view_out, tmp;

    {
      torch::AutoGradMode mode(false);
      view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
      assert_tensor_creation_meta(view_out, torch::autograd::CreationMeta::NO_GRAD_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());
    }

    tmp = functional_op(view_out);
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    if (requires_grad) {
      ASSERT_THROWS_WITH(inplace_op(view_out),  // go through kernels: VariableType, InplaceOrView, CPU
        "A view was created in no_grad mode and is being modified inplace")
    } else {
        inplace_op(view_out);
    }

    tmp = view_op(view_out);
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}


