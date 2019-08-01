#include <gtest/gtest.h>

#include <torch/autograd.h>

#include <torch/utils.h>
#include <test/cpp/api/support.h>

using namespace torch::autograd;

#define ASSERT_VARIABLE_EQ(a,b) ASSERT_TRUE(torch::allclose((a),(b)))

std::string graph_desc(std::shared_ptr<Node> node) {
  if (!node) {
    return "None";
  }
  auto result = node->name() + "(";
  auto next_edges = node->next_edges();
  for(auto& edge : next_edges) {
    result += graph_desc(edge.function);
  }
  return result+")";
}

TEST(CustomAutogradTest, CustomFunction) {
  struct MyFunction : public Function<MyFunction> {
    static variable_list forward(AutogradContext *ctx, Variable var1, int mul, Variable var2) {
      ctx->saved_data["mul"] = mul;
      ctx->save_for_backward({var1, var2});
      return {var1 + mul*var2 + var1*var2};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      int mul = ctx->saved_data["mul"].toInt();
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];
      variable_list output = {grad_output[0] + grad_output[0]*var2, Variable(), grad_output[0] * mul + grad_output[0] * var1};
      return output;
    }
  };

  Variable x = torch::randn({5,5}, torch::requires_grad());
  Variable y = torch::randn({5,5}, torch::requires_grad());
  auto res = MyFunction::apply(x,2,y)[0];
  auto go = torch::ones({}, torch::requires_grad());
  res.sum().backward(go, false, true);

  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({5,5}));
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({5,5})*2);
}

TEST(CustomAutogradTest, FunctionReturnsInput) {
  struct MyFunction : public Function<MyFunction> {
    static variable_list forward(AutogradContext *ctx, Variable var1) {
      return {var1};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      return {grad_output[0]*2};
    }
  };

  Variable x(torch::ones(1, torch::requires_grad()));
  MyFunction::apply(x)[0].backward(torch::ones(1) , true, true);
  ASSERT_VARIABLE_EQ(x.grad(), torch::full(1,2));
}

TEST(CustomAutogradTest, NoGradCustomFunction) {
  // Custom Function should respect grad mode
 struct MyOp : public Function<MyOp> {
   static variable_list forward(AutogradContext *ctx, Variable x) {
     return {x+1};
   }

   static variable_list backward(AutogradContext *ctx, variable_list dy) {
     return dy;
   }
 };

 auto x = torch::ones({5,5}, torch::requires_grad());
 {
    at::NoGradGuard no_grad;
    auto y = MyOp::apply(x)[0];
    ASSERT_FALSE(y.requires_grad());
 }
}

TEST(CustomAutogradTest, MarkNonDifferentiable) {
  struct MyFunction : public Function<MyFunction> {
    static variable_list forward(AutogradContext *ctx, Variable v) {
      Variable output = v > 0;
      ctx->mark_non_differentiable({output});
      return {output};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      return { (grad_output[0]*0.0) };
    }
  };

  auto x = torch::randn({5,5}, torch::requires_grad());
  auto mask = MyFunction::apply(x)[0];
  ASSERT_FALSE(mask.requires_grad());
  auto y = x.masked_fill(mask, 0);
  y.sum().backward();
}

TEST(CustomAutogradTest, ReturnLeafInplace) {
  struct Inplace : public Function<Inplace> {
    static variable_list forward(AutogradContext *ctx, Variable a, Variable b) {
      ctx->mark_dirty({a});
      return {a.add_(b), b+2};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      return {grad_output[0], grad_output[0] + grad_output[1]};
    }
  };

  Variable x = torch::randn({5,5});
  Variable y = torch::randn({5,5}, torch::requires_grad());

  auto out = Inplace::apply(x,y);
  auto &q = out[0];
  ASSERT_TRUE(torch::equal(q, x));
  ASSERT_TRUE(q.requires_grad());
  q.sum().backward();
  ASSERT_VARIABLE_EQ(y.grad(), torch::ones({5,5}));
}
