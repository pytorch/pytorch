#include <ATen/core/boxing/impl/test_helpers.h>
#include <gtest/gtest.h>

#include <ATen/core/op_registration/op_registration.h>
#include <torch/torch.h>

#include <torch/csrc/autograd/FunctionsManual.h>
#include <torch/csrc/autograd/functions/basic_ops.h>

#include <test/cpp/api/support.h>

using namespace torch::autograd;
using namespace torch::test;

#define ASSERT_VARIABLE_EQ(a, b) ASSERT_TRUE(torch::allclose((a), (b)))
#define EXPECT_VARIABLE_EQ(a, b) EXPECT_TRUE(torch::allclose((a), (b)))

std::string graph_desc(std::shared_ptr<Node> node) {
  if (!node) {
    return "None";
  }
  auto result = node->name() + "(";
  auto next_edges = node->next_edges();
  for (auto& edge : next_edges) {
    result += graph_desc(edge.function);
  }
  return result + ")";
}

Variable simple_fn(const Variable& x, const Variable& y) {
  return x + 2 * y + x * y;
}

TEST(AutogradAPITests, RegisterHookVoidReturnAcceptsUndefinedTensor) {
  auto x = at::zeros({}, at::kCPU);
  x.requires_grad_();
  x.register_hook([](at::TensorBase x) { return; });
  auto y = torch::autograd::UndefinedGrad().apply({x});
  y[0].backward();
}

TEST(AutogradAPITests, RegisterHookTensorReturnAcceptsUndefinedTensor) {
  auto x = at::zeros({}, at::kCPU);
  x.requires_grad_();
  x.register_hook([](at::Tensor x) -> at::Tensor { return x; });
  auto y = torch::autograd::UndefinedGrad().apply({x});
  y[0].backward();
}

TEST(AutogradAPITests, BackwardSimpleTest) {
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  auto res = simple_fn(x, y);
  backward({res.sum()}, {});

  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({2, 2}));
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({2, 2}) * 2);
}

TEST(AutogradAPITests, BackwardTest) {
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  auto res = simple_fn(x, y);
  backward({res}, {torch::ones({2, 2})}, {}, true);

  backward({res}, {torch::ones({2, 2})});

  ASSERT_VARIABLE_EQ(x.grad(), 2 * (y + torch::ones({2, 2})));
  ASSERT_VARIABLE_EQ(y.grad(), 2 * (x + torch::ones({2, 2}) * 2));
}

TEST(AutogradAPITests, GradSimpleTest) {
  // basic grad
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  auto res = simple_fn(x, y);
  auto grad_res = grad({res}, {x, y}, {torch::ones({2, 2})});

  ASSERT_VARIABLE_EQ(grad_res[0], y + torch::ones({2, 2}));
  ASSERT_VARIABLE_EQ(grad_res[1], x + torch::ones({2, 2}) * 2);
}

TEST(AutogradAPITests, GradTest) {
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  auto res = simple_fn(x, y);
  res.backward(torch::ones({2, 2}), false, true);

  Variable x_grad = y + torch::ones({2, 2});
  Variable y_grad = x + torch::ones({2, 2}) * 2;
  ASSERT_VARIABLE_EQ(x.grad(), x_grad);
  ASSERT_VARIABLE_EQ(y.grad(), y_grad);

  Variable grad_sum = 2 * x.grad() + y.grad();
  auto x_hv = grad({grad_sum}, {x}, {torch::ones({2, 2})}, {}, true);

  ASSERT_VARIABLE_EQ(x_hv[0], torch::ones({2, 2}));
  ASSERT_VARIABLE_EQ(x.grad(), x_grad);
  ASSERT_VARIABLE_EQ(y.grad(), y_grad);
}

TEST(AutogradAPITests, GradNonLeafTest) {
  Variable x_init = torch::randn({2, 2}, torch::requires_grad());
  Variable x = x_init;
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  Variable grad_output = torch::ones({2, 2});

  for (int i = 0; i < 5; ++i) {
    auto res = simple_fn(x, y);
    auto input_grads = grad({res}, {x}, {grad_output}, {}, true);

    Variable grad_x_expected = y + torch::ones({2, 2});
    ASSERT_VARIABLE_EQ(input_grads[0], grad_x_expected);
    ASSERT_FALSE(x.grad().defined());
    ASSERT_FALSE(y.grad().defined());
    x = x + 0.05 * input_grads[0];
  }

  float val_init = simple_fn(x_init, y).sum().item().toFloat();
  float val_final = simple_fn(x, y).sum().item().toFloat();
  ASSERT_TRUE(val_final > val_init);

  x.backward(grad_output, false, true);
  ASSERT_TRUE(x_init.grad().defined());
  ASSERT_TRUE(y.grad().defined());
}

TEST(AutogradAPITests, GradUnreachableTest) {
  Variable x = torch::ones({1}, torch::requires_grad());
  Variable y = torch::ones({1}, torch::requires_grad());

  Variable z = x * 2;
  Variable w = y * 2;

  auto grad_res = grad({x * 2}, {x, y}, {}, {}, false, true);
  ASSERT_VARIABLE_EQ(grad_res[0], x * 2);
  ASSERT_FALSE(grad_res[1].defined());

  // This is slightly different than the case above, because z doesn't even
  // have a grad accumulator allocated.
  z = torch::ones({1}, torch::requires_grad());
  grad_res = grad({x * 2}, {x, z}, {}, {}, false, true);

  ASSERT_VARIABLE_EQ(grad_res[0], x * 2);
  ASSERT_FALSE(grad_res[1].defined());

  // allow_unused=False, but grads contains None inside, should throw
  ASSERT_THROWS_WITH(
      grad({x * 2}, {x, y}, {}, {}, false, false), "Set allow_unused=True");
}

TEST(CustomAutogradTest, GradUnreachableDiscoveryTest) {
  // Test that certain nodes are not erroneously executed when an input
  // is unreachable. See #39784
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable var) {
      return var;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      ADD_FAILURE() << "This node should not be executed!";
      return grad_output;
    }
  };

  auto x = torch::randn(1, torch::requires_grad());
  auto x1 = torch::randn(1);
  auto x2 = MyFunction::apply(x + x1);

  auto y = torch::randn(1, torch::requires_grad());
  auto grad_res = torch::autograd::grad({x2}, {y}, {}, {}, false, true);
  ASSERT_FALSE(grad_res[0].defined());
}

TEST(AutogradAPITests, EmptyInput) {
  Variable x = torch::ones({1}, torch::requires_grad());
  ASSERT_THROWS_WITH(
      grad({x * 2}, /*inputs=*/{}, {x}), "grad requires non-empty inputs.");
}

TEST(AutogradAPITests, RetainGrad) {
  auto input = torch::rand({1, 3}, torch::requires_grad());
  auto h1 = input * 3;
  auto out = (h1 * h1).sum();

  {
    // Warning when grad is accessed for non-leaf tensor
    WarningCapture warnings;
    ASSERT_FALSE(h1.grad().defined());
    ASSERT_TRUE(warnings.str().find("is not a leaf") != std::string::npos);
  }
  // It should be possible to call retain_grad() multiple times
  h1.retain_grad();
  h1.retain_grad();
  {
    // If retain_grad is true for a non-leaf tensor,
    // there should not be any warning when grad is accessed
    WarningCapture warnings;
    ASSERT_FALSE(h1.grad().defined());
    ASSERT_FALSE(warnings.str().find("is not a leaf") != std::string::npos);
  }

  // Gradient should be accumulated
  // NOLINTNEXTLINE(bugprone-argument-comment)
  out.backward({}, /*keep_graph=*/true);
  ASSERT_VARIABLE_EQ(h1 * 2, h1.grad());
  // NOLINTNEXTLINE(bugprone-argument-comment)
  out.backward({}, /*keep_graph=*/true);
  ASSERT_VARIABLE_EQ(h1 * 4, h1.grad());

  {
    torch::NoGradGuard no_grad;
    input.grad().zero_();
  }
  // It should be a no-op for leaves
  input.retain_grad();
  input.retain_grad();
  out.backward();
  ASSERT_VARIABLE_EQ(input * 18, input.grad());
}

TEST(AutogradAPITests, AnomalyMode) {
  // Needs to have backtrace as warning and then throw an error
  torch::autograd::DetectAnomalyGuard detect_anomaly;
  {
    WarningCapture warnings;
    auto x = torch::tensor({5.0}, torch::requires_grad());
    auto y = x * x;
    auto z = y * y;
    y += 1;
    ASSERT_THROWS_WITH(z.backward(), "inplace");
    ASSERT_TRUE(
        warnings.str().find("Traceback of forward") != std::string::npos);
  }
  auto double_backward_produce_nan = [](bool should_throw) {
    auto x = torch::tensor({0.0}, torch::requires_grad());
    auto y = x.pow(1.5);
    auto gr =
        // NOLINTNEXTLINE(bugprone-argument-comment)
        grad({y}, {x}, {}, /*retain_graph=*/true, /*create_backward=*/true);
    if (should_throw) {
      WarningCapture warnings;
      ASSERT_THROWS_WITH(grad({gr[0]}, {x}, {torch::tensor({0.0})});
                         , "returned nan");
      auto msgs = warnings.messages();
      ASSERT_EQ(msgs.size(), 2);
      ASSERT_TRUE(
          msgs[0].find("Traceback of forward call that caused the error") !=
          std::string::npos);
      ASSERT_TRUE(
          msgs[1].find(
              "Traceback of forward call that induced the previous calculation") !=
          std::string::npos);
    } else {
      grad({gr[0]}, {x}, {torch::tensor({0.0})});
    }
  };

  double_backward_produce_nan(true);
  {
    torch::autograd::DetectAnomalyGuard detect_anomaly(/*check_nan=*/false);
    double_backward_produce_nan(false);
    {
      torch::autograd::DetectAnomalyGuard detect_anomaly(/*check_nan=*/true);
      double_backward_produce_nan(true);
    }
  }
  double_backward_produce_nan(true);
}

TEST(CustomAutogradTest, CustomFunctionReturnInputAsIsAndSavesIt) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(
        AutogradContext* ctx,
        Variable var1,
        Variable var2) {
      ctx->save_for_backward({var1, var2});
      return var1 * var2, var1;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {};
    }
  };

  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  MyFunction::apply(x, y);
}

TEST(CustomAutogradTest, CustomFunction) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(
        AutogradContext* ctx,
        Variable var1,
        int mul,
        Variable var2) {
      ctx->saved_data["mul"] = mul;
      ctx->save_for_backward({var1, var2});
      return var1 + mul * var2 + var1 * var2;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      int mul = ctx->saved_data["mul"].toInt();
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];
      variable_list output = {
          grad_output[0] + grad_output[0] * var2,
          Variable(),
          grad_output[0] * mul + grad_output[0] * var1};
      return output;
    }
  };

  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  auto res = MyFunction::apply(x, 2, y);
  auto go = torch::ones({}, torch::requires_grad());
  res.sum().backward(go, false, true);

  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({5, 5}));
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({5, 5}) * 2);
}

TEST(CustomAutogradTest, CustomFunctionWithTensorList) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, at::TensorList tensors) {
      torch::autograd::variable_list vars;
      for (const at::Tensor& tensor : tensors) {
        vars.push_back(tensor);
      }
      ctx->save_for_backward(vars);
      return tensors[0] + tensors[1] + tensors[0] * tensors[1];
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];
      variable_list output = {
          grad_output[0] + grad_output[0] * var2,
          grad_output[0] + grad_output[0] * var1};
      return output;
    }
  };

  at::Tensor x = torch::randn({5, 5}, torch::requires_grad());
  at::Tensor y = torch::randn({5, 5}, torch::requires_grad());
  torch::autograd::variable_list variables = {x, y};
  at::TensorList tensors = variables;
  auto res = MyFunction::apply(tensors);
  auto go = torch::ones({}, torch::requires_grad());
  res.sum().backward(go, false, true);

  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({5, 5}));
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({5, 5}));
}

TEST(CustomAutogradTest, GraphTaskTrimEdges) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(
        AutogradContext* ctx,
        Variable var1,
        Variable var2,
        int mul,
        bool needs_input1_grad,
        bool needs_input2_grad) {
      // setup the expected should and should not compute idx
      ctx->saved_data["needs_input1_grad"] = needs_input1_grad;
      ctx->saved_data["needs_input2_grad"] = needs_input2_grad;

      ctx->saved_data["mul"] = mul;
      ctx->save_for_backward({var1, var2});
      return var1 + mul * var2 + var1 * var2;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // Test `needs_input_grad` method is working correctly.
      // We have to test this within the backward function.
      auto needs_input1_grad = ctx->saved_data["needs_input1_grad"].toBool();
      auto needs_input2_grad = ctx->saved_data["needs_input2_grad"].toBool();
      IndexRange var1_idx = {0, 1};
      IndexRange var2_idx = {1, 2};
      EXPECT_EQ(ctx->needs_input_grad(0), needs_input1_grad);
      EXPECT_EQ(ctx->needs_input_grad(1), needs_input2_grad);
      EXPECT_EQ(ctx->needs_input_grad({var1_idx}), needs_input1_grad);
      EXPECT_EQ(ctx->needs_input_grad({var2_idx}), needs_input2_grad);
      EXPECT_EQ(
          ctx->needs_input_grad({var1_idx, var2_idx}),
          needs_input1_grad || needs_input2_grad);

      // calculate gradients
      int mul = ctx->saved_data["mul"].toInt();
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];

      Variable grad_var1, grad_var2;
      if (ctx->needs_input_grad(0)) {
        grad_var1 = grad_output[0] + grad_output[0] * var2;
      }
      if (ctx->needs_input_grad(1)) {
        grad_var2 = grad_output[0] * mul + grad_output[0] * var1;
      }
      variable_list output = {
          grad_var1,
          grad_var2,
          Variable(),
          Variable(),
          Variable(),
      };
      return output;
    }
  };

  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  auto go = torch::ones_like(x);
  Variable out;

  // grad_x
  out = MyFunction::apply(
      x,
      y,
      2,
      /* needs_input1_grad= */ true,
      /* needs_input2_grad= */ false);
  auto grad_x = torch::autograd::grad({out}, {x}, {go})[0];
  ASSERT_VARIABLE_EQ(grad_x, y + torch::ones({5, 5}));

  // grad_y
  out = MyFunction::apply(
      x,
      y,
      2,
      /* needs_input1_grad= */ false,
      /* needs_input2_grad= */ true);
  auto grad_y = torch::autograd::grad({out}, {y}, {go})[0];
  ASSERT_VARIABLE_EQ(grad_y, x + torch::ones({5, 5}) * 2);

  // grad_x and grad_y
  out = MyFunction::apply(
      x,
      y,
      2,
      /* needs_input1_grad= */ true,
      /* needs_input2_grad= */ true);
  auto grads = torch::autograd::grad({out}, {x, y}, {go});
  ASSERT_VARIABLE_EQ(grads[0], y + torch::ones({5, 5}));
  ASSERT_VARIABLE_EQ(grads[1], x + torch::ones({5, 5}) * 2);
}

TEST(CustomAutogradTest, FunctionReturnsInput) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable var1) {
      return var1;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {grad_output[0] * 2};
    }
  };

  Variable x(torch::ones(1, torch::requires_grad()));
  MyFunction::apply(x).backward(torch::ones(1), true, true);
  ASSERT_VARIABLE_EQ(x.grad(), torch::full(1, 2.));
}

TEST(CustomAutogradTest, FunctionReturnsUndefined) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable var) {
      return var * 2;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      at::Tensor undefined_tensor;
      return {undefined_tensor};
    }
  };

  auto x = torch::ones(1, torch::requires_grad());

  MyFunction::apply(x).backward();
  ASSERT_FALSE(x.grad().defined());

  MyFunction::apply(x.pow(2)).backward();
  ASSERT_FALSE(x.grad().defined());

  MyFunction::apply(x).sum().backward();
  ASSERT_FALSE(x.grad().defined());

  ASSERT_FALSE(torch::autograd::grad(
                   {MyFunction::apply(x)}, {x}, {}, false, false, true)[0]
                   .defined());
}

TEST(CustomAutogradTest, MaterializeGrads) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable var) {
      return var;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      EXPECT_VARIABLE_EQ(grad_output[0], torch::zeros(1));
      return grad_output;
    }
  };

  auto x = torch::ones(1, torch::requires_grad());
  UndefinedGrad().apply({MyFunction::apply(x)})[0].backward();
}

TEST(CustomAutogradTest, DontMaterializeGrads) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable var) {
      ctx->set_materialize_grads(false);
      return var;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      EXPECT_FALSE(grad_output[0].defined());
      return grad_output;
    }
  };

  auto x = torch::ones(1, torch::requires_grad());
  UndefinedGrad().apply({MyFunction::apply(x)})[0].backward();
}

TEST(CustomAutogradTest, NoGradCustomFunction) {
  // Custom Function should respect grad mode
  struct MyOp : public Function<MyOp> {
    static Variable forward(AutogradContext* ctx, Variable x) {
      return x + 1;
    }

    static variable_list backward(AutogradContext* ctx, variable_list dy) {
      return dy;
    }
  };

  auto x = torch::ones({5, 5}, torch::requires_grad());
  {
    at::NoGradGuard no_grad;
    auto y = MyOp::apply(x);
    ASSERT_FALSE(y.requires_grad());
  }
}

TEST(CustomAutogradTest, MarkDirty) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable v) {
      // Change the value inplace
      auto v_data = v.data_ptr<float>();
      v_data[0] = 2;
      ctx->mark_dirty({v});
      return v;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {(grad_output[0] * 2.0)};
    }
  };

  // Clone here because modifying leafs inplace is not allowed
  auto x = torch::randn({5, 5}, torch::requires_grad()).clone();
  auto version_before = x._version();
  auto out = MyFunction::apply(x);
  auto version_after = x._version();
  ASSERT_TRUE(version_after >= (version_before + 1));
  out.sum().backward();
}

TEST(CustomAutogradTest, MarkNonDifferentiable) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable v) {
      Variable output = v > 0;
      ctx->mark_non_differentiable({output});
      return output;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {(grad_output[0] * 0.0)};
    }
  };

  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto mask = MyFunction::apply(x);
  ASSERT_FALSE(mask.requires_grad());
  auto y = x.masked_fill(mask, 0);
  y.sum().backward();
}

TEST(CustomAutogradTest, MarkNonDifferentiableMixed) {
  struct MyFunction : public Function<MyFunction> {
    static variable_list forward(AutogradContext* ctx, Variable input) {
      Variable a = input + 1;
      Variable b = input + 2;
      ctx->mark_non_differentiable({a});
      return {a, b};
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      const Variable &grad_a = grad_output[0], &grad_b = grad_output[1];
      EXPECT_VARIABLE_EQ(grad_a, torch::zeros({5, 5}));
      EXPECT_VARIABLE_EQ(grad_b, torch::ones({5, 5}));
      return {grad_b};
    }
  };

  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto out = MyFunction::apply(x);

  ASSERT_FALSE(out[0].requires_grad());
  ASSERT_TRUE(out[1].requires_grad());
  out[1].sum().backward();
  ASSERT_VARIABLE_EQ(x.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, MarkNonDifferentiableNone) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable input) {
      auto output = input.clone();
      ctx->mark_non_differentiable({output});
      return output;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_outputs) {
      return {};
    }
  };

  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto r = MyFunction::apply(x * x);
  (r * x).sum().backward();
}

TEST(CustomAutogradTest, ReturnLeafInplace) {
  struct Inplace : public Function<Inplace> {
    static variable_list forward(AutogradContext* ctx, Variable a, Variable b) {
      ctx->mark_dirty({a});
      return {a.add_(b), b + 2};
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {grad_output[0], grad_output[0] + grad_output[1]};
    }
  };

  Variable x = torch::randn({5, 5});
  Variable y = torch::randn({5, 5}, torch::requires_grad());

  auto out = Inplace::apply(x, y);
  auto& q = out[0];
  ASSERT_TRUE(torch::equal(q, x));
  ASSERT_TRUE(q.requires_grad());
  q.sum().backward();
  ASSERT_VARIABLE_EQ(y.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, ReturnDuplicateInplace) {
  struct DoubleInplace : public Function<DoubleInplace> {
    static variable_list forward(AutogradContext* ctx, Variable x) {
      x.mul_(2);
      ctx->mark_dirty({x});
      return {x, x};
    }

    static variable_list backward(
        AutogradContext* ctsx,
        variable_list grad_outputs) {
      return {grad_outputs[0] * 2 + grad_outputs[1] * 2};
    }
  };

  auto x = torch::randn({5, 5}, torch::requires_grad());

  ASSERT_THROWS_WITH(
      DoubleInplace::apply(x), "leaf Variable that requires grad");
  // TODO ASSERT_THROWS_WITH(DoubleInplace::apply(x.clone()[0]), "only one
  // output");

  auto out = DoubleInplace::apply(x.clone());
  ASSERT_TRUE(torch::equal(out[0], out[1]));
}

TEST(CustomAutogradTest, ReturnDuplicate) {
  struct DoubleDuplicate : public Function<DoubleDuplicate> {
    static variable_list forward(AutogradContext* ctx, Variable x) {
      auto output = x * 2;
      return {output, output};
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_outputs) {
      return {grad_outputs[0] * 2 + grad_outputs[1] * 2};
    }
  };

  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto out = DoubleDuplicate::apply(x);
  ASSERT_TRUE(torch::equal(out[0], out[1]));
}

TEST(CustomAutogradTest, SaveEmptyForBackward) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable input) {
      ctx->save_for_backward({Variable(), input, Variable()});
      return input * input;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      auto saved = ctx->get_saved_variables();
      EXPECT_FALSE(saved[0].defined());
      EXPECT_FALSE(saved[2].defined());
      return {saved[1] * 2 * grad_output[0]};
    }
  };

  Variable x = torch::randn({5, 5}, torch::requires_grad());
  auto y = MyFunction::apply(x);
  y.sum().backward();
  ASSERT_VARIABLE_EQ(x.grad(), 2 * x);
}

TEST(CustomAutogradTest, InvalidGradients) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable x) {
      return x * 2;
    }

    static variable_list backward(
        AutogradContext* ctsx,
        variable_list grad_outputs) {
      return {
          torch::randn(10, torch::dtype(torch::kFloat).requires_grad(true))};
    }
  };

  auto input1 =
      torch::randn({5, 5}, torch::dtype(torch::kFloat).requires_grad(true));
  ASSERT_THROWS_WITH(
      MyFunction::apply(input1).sum().backward(), "expected shape");
  auto input2 =
      torch::randn(10, torch::dtype(torch::kDouble).requires_grad(true));
}

TEST(CustomAutogradTest, NoGradInput) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext*, Variable x) {
      return x;
    }

    static variable_list backward(
        AutogradContext*,
        variable_list grad_outputs) {
      return grad_outputs;
    }
  };

  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y;
  {
    at::NoGradGuard no_grad;
    y = MyFunction::apply(x);
  }

  ASSERT_TRUE(x.requires_grad());
  ASSERT_FALSE(y.grad_fn());
}

TEST(CustomAutogradTest, TooManyGrads) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext*, Variable input) {
      return input;
    }

    static variable_list backward(AutogradContext*, variable_list grad_output) {
      grad_output.insert(grad_output.end(), {Variable(), Variable()});
      return grad_output;
    }
  };
}

TEST(CustomAutogradTest, DepNoGrad) {
  struct F1 : public Function<F1> {
    static variable_list forward(AutogradContext* ctx, Variable input) {
      auto out = torch::randn(input.sizes());
      ctx->mark_non_differentiable({out});
      return {input, out};
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {grad_output[0]};
    }
  };

  struct F2 : public Function<F2> {
    static Variable forward(AutogradContext*, Variable input, Variable ignore) {
      return input;
    }

    static variable_list backward(AutogradContext*, variable_list grad_output) {
      return {grad_output[0], Variable()};
    }
  };

  auto x = torch::randn(5, torch::requires_grad());
  auto out = F1::apply(x);
  Variable &a = out[0], &b = out[1];
  b = b + 1; // Separate F1 and F2 by another operation
  ASSERT_TRUE(a.requires_grad());
  ASSERT_FALSE(b.requires_grad());

  auto c = F2::apply(a, b);
  c.backward(torch::ones(c.sizes()), false, false);
  ASSERT_VARIABLE_EQ(x.grad(), torch::ones(x.sizes()));
}

TEST(CustomAutogradTest, Reentrant) {
  static Variable y_data = torch::randn({2, 2});
  struct Reenter : public Function<Reenter> {
    static Variable forward(AutogradContext* ctx, Variable input) {
      Variable output;
      {
        at::AutoGradMode enable_grad(true);
        auto x = make_variable(input.tensor_data(), true);
        auto y = make_variable(y_data.tensor_data(), true);
        output = x * y;

        ctx->saved_data["x"] = x;
        ctx->saved_data["y"] = y;
        ctx->saved_data["output_var"] = output;
      }
      return output.detach();
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      {
        at::AutoGradMode enable_grad(true);
        auto out = ctx->saved_data["output_var"].toTensor();
        out.sum().backward();
      }
      return {ctx->saved_data["x"].toTensor().grad() * grad_output[0]};
    }
  };

  auto x = torch::randn({2, 2}, torch::requires_grad());
  auto out = Reenter::apply(x);
  out.sum().backward();
  ASSERT_VARIABLE_EQ(x.grad(), y_data);
}

// NOTE: If this fails for apparently unrelated reasons in TSAN be aware of
// the TSAN limit on mutex: https://github.com/google/sanitizers/issues/950
TEST(CustomAutogradTest, DeepReentrant) {
  struct DeepReenter : public Function<DeepReenter> {
    static Variable forward(AutogradContext* ctx, Variable x) {
      {
        at::AutoGradMode enable_grad(true);
        ctx->saved_data["x"] = make_variable(x.tensor_data(), true) - 1;
      }
      return ctx->saved_data["x"].toTensor().detach();
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      if (!at::native::is_nonzero(ctx->saved_data["x"].toTensor())) {
        return grad_output;
      }
      {
        at::AutoGradMode enable_grad(true);
        apply(ctx->saved_data["x"].toTensor())[0].sum().backward();
        return grad_output;
      }
    }
  };

  // This should not stack overflow
  auto v =
      torch::tensor({8193}, torch::dtype(torch::kFloat).requires_grad(true));
  DeepReenter::apply(v).sum().backward();
}

TEST(CustomAutogradTest, ReentrantPriority) {
  static std::vector<int> order;

  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext*, Variable x) {
      return x;
    }

    static variable_list backward(AutogradContext*, variable_list grad) {
      order.push_back(0);
      return grad;
    }
  };

  struct Reenter : public Function<Reenter> {
    static Variable forward(AutogradContext* ctx, Variable x) {
      {
        at::AutoGradMode enable_grad(true);
        ctx->saved_data["x"] = make_variable(x.tensor_data(), true) - 1;
      }
      return ctx->saved_data["x"].toTensor().detach();
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      order.push_back(1);
      if (!at::native::is_nonzero(ctx->saved_data["x"].toTensor())) {
        return grad_output;
      }
      {
        at::AutoGradMode enable_grad(true);
        apply(ctx->saved_data["x"].toTensor())[0].sum().backward();
        return grad_output;
      }
    }
  };

  auto a = MyFunction::apply(
      torch::tensor({6}, torch::dtype(torch::kFloat).requires_grad(true)));
  auto b = Reenter::apply(
      torch::tensor({9}, torch::dtype(torch::kFloat).requires_grad(true)));
  auto v = a * b;
  v.backward();

  // All the reentrant tasks should be prioritized over the MyFunction backward
  // task.
  ASSERT_EQ(order.size(), 10);
  ASSERT_EQ(std::count(order.begin(), order.end(), 1), 9);
  ASSERT_EQ(order.back(), 0);
  // Clear static variable in case test get executed in a loop
  order.clear();
}

TEST(CustomAutogradTest, Hooks) {
  Variable x = torch::ones({5, 5}, torch::requires_grad());
  Variable y = torch::ones({5, 5}) * 4;
  y.set_requires_grad(true);

  int counter = 0;

  std::function<void(int, Variable)> bw_hook(
      [&counter](int inc, Variable grad) { counter += inc; });

  Variable z = x * x + x * 2 + x * y + y;
  x.register_hook([&bw_hook](Variable grad) { bw_hook(0, grad); });
  auto hook_1 =
      z.register_hook([&bw_hook](Variable grad) { bw_hook(1, grad); });
  z.backward(torch::ones({5, 5}), true, true);
  ASSERT_EQ(counter, 1);

  auto hook_2 =
      z.register_hook([&bw_hook](Variable grad) { bw_hook(2, grad); });
  z.backward(torch::ones({5, 5}), true, true);
  ASSERT_EQ(counter, 4);

  z.remove_hook(hook_2);
  z.backward(torch::ones({5, 5}), true, true);
  ASSERT_EQ(counter, 5);

  std::function<Variable(Variable)> bw_hook_modify(
      [](Variable grad) { return grad.mul(2); });

  z.remove_hook(hook_1);
  z.register_hook(bw_hook_modify);
  y.grad().zero_();
  z.backward(torch::ones({5, 5}), true, false);
  ASSERT_VARIABLE_EQ(y.grad(), (x + 1) * 2);

  y.register_hook(bw_hook_modify);
  y.grad().zero_();
  z.backward(torch::ones({5, 5}), false, false);
  ASSERT_VARIABLE_EQ(y.grad(), (x + 1) * 4);

  ASSERT_THROWS_WITH(y.remove_hook(3), "Invalid index");
}

TEST(CustomAutogradTest, HooksInplace) {
  auto a = torch::ones({5, 5}, torch::requires_grad()).clone();

  int hook1_count = 0;
  auto hook1 = ([&hook1_count](Variable grad) {
    hook1_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 2);
  });

  int hook2_count = 0;
  auto hook2 = ([&hook2_count](Variable grad) {
    hook2_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}));
  });

  a.register_hook(hook1);
  a.mul_(2);
  a.register_hook(hook2);

  auto out = (a + 1).sum();
  out.backward();

  ASSERT_EQ(hook1_count, 1);
  ASSERT_EQ(hook2_count, 1);
}

TEST(CustomAutogradTest, HooksInplaceWithRetainsGrad) {
  auto a = torch::ones({5, 5}, torch::requires_grad()).clone();

  int hook1_count = 0;
  auto hook1 = ([&hook1_count](Variable grad) {
    hook1_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 2);
  });

  int hook2_count = 0;
  auto hook2 = ([&hook2_count](Variable grad) {
    hook2_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 2);
  });

  int hook3_count = 0;
  auto hook3 = ([&hook3_count](Variable grad) {
    hook3_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}));
  });

  a.register_hook(hook1);
  a.retain_grad();
  a.register_hook(hook2);

  a.mul_(2);
  a.register_hook(hook3);

  auto out = (a + 1).sum();
  out.backward();

  ASSERT_EQ(hook1_count, 1);
  ASSERT_EQ(hook2_count, 1);
  ASSERT_EQ(hook3_count, 1);

  ASSERT_TRUE(a.retains_grad());
  ASSERT_VARIABLE_EQ(a.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, HooksInplaceTwiceWithRetainsGrad) {
  auto a = torch::ones({5, 5}, torch::requires_grad()).clone();

  int hook1_count = 0;
  auto hook1 = ([&hook1_count](Variable grad) {
    hook1_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 4);
  });

  int hook2_count = 0;
  auto hook2 = ([&hook2_count](Variable grad) {
    hook2_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 4);
  });

  int hook3_count = 0;
  auto hook3 = ([&hook3_count](Variable grad) {
    hook3_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}));
  });

  a.register_hook(hook1);
  a.retain_grad();
  a.register_hook(hook2);

  a.mul_(2);
  a.mul_(2);
  a.register_hook(hook3);

  auto out = (a + 1).sum();
  out.backward();

  ASSERT_EQ(hook1_count, 1);
  ASSERT_EQ(hook2_count, 1);
  ASSERT_EQ(hook3_count, 1);

  ASSERT_TRUE(a.retains_grad());
  ASSERT_VARIABLE_EQ(a.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, HookNone) {
  struct NoneGradientFunction : public Function<NoneGradientFunction> {
    static variable_list forward(AutogradContext* ctx, Variable x, Variable y) {
      return {x, y};
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad) {
      return {grad[0], Variable()};
    }
  };

  bool was_called = false;

  auto hook = ([&was_called](Variable grad) {
    ASSERT_TRUE(grad.defined());
    was_called = true;
  });

  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto y = torch::randn({5, 5});

  auto out = NoneGradientFunction::apply(x, y);
  Variable rx = x[0], ry = x[1];

  rx.register_hook(hook);
  ry.register_hook(hook);
  (rx + ry).sum().backward();
  ASSERT_TRUE(was_called);
}

TEST(CustomAutogradTest, BackwardWithInputs) {
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  Variable z = x * x + x * y + y * y;
  Variable x_grad_expected = 2 * x + y;
  Variable y_grad_expected = x + 2 * y;

  z.backward(torch::ones({5, 5}), false, false, {x});

  ASSERT_VARIABLE_EQ(x.grad(), x_grad_expected);
  ASSERT_FALSE(y.grad().defined());
}

TEST(CustomAutogradTest, BackwardWithEmptyInputs) {
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  Variable z = x * x + x * y + y * y;
  Variable x_grad_expected = 2 * x + y;
  Variable y_grad_expected = x + 2 * y;
  ASSERT_THROWS_WITH(
      z.backward(torch::ones({5, 5}), false, false, std::vector<Variable>{}),
      "cannot be empty");
}

TEST(CustomAutogradTest, BackwardWithNonLeafInputs) {
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  Variable z = x * x;
  Variable w = y * z + x * y + y * y;

  Variable x_grad_expected = 2 * x * y + y;
  Variable z_grad_expected = y;

  w.backward(torch::ones({5, 5}), false, false, std::vector<Variable>{x, z});

  ASSERT_VARIABLE_EQ(x.grad(), x_grad_expected);
  ASSERT_VARIABLE_EQ(z.grad(), z_grad_expected);
  ASSERT_FALSE(y.grad().defined());
}

TEST(CustomAutogradTest, BackwardWithCreateGraphWarns) {
  c10::WarningUtils::WarnAlways guard(true);

  torch::Tensor x = torch::randn({5, 5}).set_requires_grad(true);
  auto z = x * x;
  {
    WarningCapture warnings;
    z.backward(torch::ones({5, 5}), c10::nullopt, true);
    ASSERT_TRUE(
        warnings.str().find("Using backward() with create_graph=True") !=
        std::string::npos);
  }

  {
    WarningCapture warnings;
    torch::autograd::backward({z}, {torch::ones({5, 5})}, c10::nullopt, true);
    ASSERT_TRUE(
        warnings.str().find("Using backward() with create_graph=True") !=
        std::string::npos);
  }
}

/**
 * Tests for AutogradNotImplementedFallback
 * - Check that we created the NotImplemented kernel when inputs require grad
 *   but when no inputs require grad, we should not create this node
 * - check_inplace logic
 * - view ops
 * - TODO: Tests for debug-only checks? Don't need for now because CI doesn't
 * test non-NDEBUG builds.
 * - tensorlist input and output
 * - multiple outputs / non-tensor output
 * - rebase_history vs set_history
 */
namespace {

torch::Tensor inplace_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return self.add_(other);
}

std::tuple<torch::Tensor, torch::Tensor> two_arg_inplace_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  other.add_(self);
  self.add_(other);
  return std::tuple<torch::Tensor, torch::Tensor>(self, other);
}

std::tuple<torch::Tensor, torch::Tensor> two_pairs_of_view_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  // This is not allowed. We test below that this calling into the boxed kernel
  // will raise an error
  return std::tuple<torch::Tensor, torch::Tensor>(self, other);
}

std::tuple<torch::Tensor, torch::Tensor> non_first_view_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  // This is not allowed. We test below that this calling into the boxed kernel
  // will raise an error
  return std::tuple<torch::Tensor, torch::Tensor>(self.clone(), other);
}

int64_t ret_single_non_tensor(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return 12;
}

torch::Tensor opt_op(
    const torch::Tensor& self,
    const c10::optional<at::Tensor>& other) {
  if (other.has_value()) {
    return self + other.value();
  } else {
    return self.clone();
  }
}

torch::Tensor my_custom_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return self + other;
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> ret_tuple_non_tensor(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  auto a = self - other;
  auto b = self + other;
  return std::tuple<torch::Tensor, torch::Tensor, int64_t>(a, b, 12);
}

torch::Tensor view_op(const torch::Tensor& self) {
  return self.alias();
}

torch::Tensor view_op_with_extra_arg(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return self.alias();
}

std::vector<torch::Tensor> ret_tensor_vector_view(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return {self.alias(), self.alias()};
}

std::vector<at::Tensor> ret_tensor_vector(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  std::vector<at::Tensor> out;
  out.push_back(self + other);
  out.push_back(self - other);
  return out;
}

torch::Tensor tensorlist_op(const torch::Tensor& self, at::TensorList other) {
  const auto& res = self.clone();
  for (const auto& t : other) {
    res.add_(t);
  }
  return res;
}

#define REGISTER_TEST_OP(name, schema, fn)                                 \
  auto m = MAKE_TORCH_LIBRARY(_test);                                      \
  m.def(schema);                                                           \
  auto m_autograd = MAKE_TORCH_LIBRARY_IMPL(_test, Autograd);              \
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);                        \
  auto m_inplaceorview = MAKE_TORCH_LIBRARY_IMPL(_test, ADInplaceOrView);  \
  m_cpu.impl(name, c10::DispatchKey::CPU, TORCH_FN(fn));                   \
  m_autograd.impl(                                                         \
      name, c10::DispatchKey::Autograd, autogradNotImplementedFallback()); \
  m_inplaceorview.impl(                                                    \
      name,                                                                \
      c10::DispatchKey::ADInplaceOrView,                                   \
      autogradNotImplementedInplaceOrViewFallback());

template <typename F>
void assertBasicChecks(F op) {
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});
  auto c = torch::tensor({1.}, {torch::kFloat32});

  // If any inputs require grad,
  auto out1 = op(a, b);
  ASSERT_THROWS_WITH(out1.backward(), "is not implemented");

  // # Should not have grad_fn if none require grad
  auto out2 = op(b, c);
  ASSERT_THROWS_WITH(
      out2.backward(),
      "element 0 of tensors does not require grad and does not have a grad_fn");

  // TODO: Forward AD Tests?
}

} // namespace

TEST(TestAutogradNotImplementedFallback, RetSingleNonTensor) {
  REGISTER_TEST_OP(
      "ret_single_non_tensor",
      "_test::ret_single_non_tensor(Tensor self, Tensor other) -> int",
      ret_single_non_tensor);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_single_non_tensor", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<int64_t, const torch::Tensor&, const torch::Tensor&>(
        opHandle, _1, _2);
  };

  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  ASSERT_EQ(op(a, b), ret_single_non_tensor(a, b));
}

TEST(TestAutogradNotImplementedFallback, InplaceOp) {
  REGISTER_TEST_OP(
      "inplace_op",
      "_test::inplace_op(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      inplace_op);
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::inplace_op", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };

  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  // Check in-place
  ASSERT_THROWS_WITH(
      op(a, b),
      "a leaf Variable that requires grad is being used in an in-place operation");
  op(b, a);
  a = a.clone();
  b = b.clone();
  auto c = op(a, b);
  ASSERT_TRUE(torch::allclose(c, inplace_op(a, b)));

  // Test in-place on view
  auto base =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();
  auto view = base.view(-1);
  auto t = torch::tensor({1.}, {torch::kFloat32});

  torch::Tensor v_nograd;
  {
    c10::NoGradGuard guard;
    v_nograd = base.view(-1);
    op(v_nograd, t);
  }

  ASSERT_THROWS_WITH(op(v_nograd, t), "A view was created in no_grad mode");
  ASSERT_EQ(op(view, t).unsafeGetTensorImpl(), view.unsafeGetTensorImpl());
  ASSERT_THAT(
      op(view, t).grad_fn()->name(), ::testing::HasSubstr("AsStridedBackward"));
}

TEST(TestAutogradNotImplementedFallback, DoubleInplaceOp) {
  REGISTER_TEST_OP(
      "two_arg_inplace_op",
      "_test::two_arg_inplace_op(Tensor(a!) self, Tensor(b!) other) -> (Tensor(a!), Tensor(b!))",
      two_arg_inplace_op);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::two_arg_inplace_op", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  // Both are modified in-place!
  ASSERT_THROWS_WITH(
      op(a, b),
      "a leaf Variable that requires grad is being used in an in-place operation");
  ASSERT_THROWS_WITH(
      op(b, a),
      "a leaf Variable that requires grad is being used in an in-place operation");

  auto c =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();
  auto d =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();

  auto saved_version_c = c._version();
  auto saved_version_d = d._version();
  op(c, d);
  ASSERT_NE(c._version(), saved_version_c);
  ASSERT_NE(d._version(), saved_version_d);
}

TEST(TestAutogradNotImplementedFallback, OptOp) {
  REGISTER_TEST_OP(
      "opt_op", "_test::opt_op(Tensor self, Tensor? other) -> Tensor", opt_op);
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::opt_op", "");
  auto op = [&](const torch::Tensor& _1,
                const c10::optional<torch::Tensor>& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const c10::optional<torch::Tensor>&>(opHandle, _1, _2);
  };

  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  ASSERT_TRUE(torch::allclose(op(a, b), opt_op(a, b)));
  ASSERT_TRUE(torch::allclose(op(a, {}), opt_op(a, {})));
}

TEST(TestAutogradNotImplementedFallback, OutOfPlaceAddition) {
  REGISTER_TEST_OP(
      "my_custom_op",
      "_test::my_custom_op(Tensor self, Tensor other) -> Tensor",
      my_custom_op);
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::my_custom_op", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };

  assertBasicChecks(op);
}

TEST(TestAutogradNotImplementedFallback, RetTupleNonTensor) {
  REGISTER_TEST_OP(
      "ret_tuple_non_tensor",
      "_test::ret_tuple_non_tensor(Tensor self, Tensor other) -> (Tensor, Tensor, int)",
      ret_tuple_non_tensor);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_tuple_non_tensor", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    torch::Tensor out0;
    torch::Tensor out1;
    int64_t out2;
    auto out = callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor, int64_t>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
    std::tie(out0, out1, out2) = std::move(out);
    return out0;
  };

  assertBasicChecks(op);
}

TEST(TestAutogradNotImplementedFallback, ViewOp) {
  REGISTER_TEST_OP(
      "view_op", "_test::view_op(Tensor(a) self) -> Tensor(a)", view_op);
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::view_op", "");
  auto op = [&](const torch::Tensor& _1) {
    return callOpUnboxed<torch::Tensor, const torch::Tensor&>(opHandle, _1);
  };
  auto b = torch::tensor({1.}, {torch::kFloat32});
  auto v = op(b);
  ASSERT_TRUE(v.is_view());
  ASSERT_EQ(v._base().unsafeGetTensorImpl(), b.unsafeGetTensorImpl());

  auto b1 =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();
  auto v1 = op(b1);
  ASSERT_TRUE(v1.is_view());
  ASSERT_EQ(v1._base().unsafeGetTensorImpl(), b1.unsafeGetTensorImpl());

  // Test inplace on view
  auto t = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);

  // raise on rebase_history when it refreshes grad_fn
  ASSERT_THROWS_WITH(
      v1.add_(t), "which does not have a derivative implemented is forbidden");
  // base should not be aware of the views, so this is still okay
  b1.add_(t);
  ASSERT_THROWS_WITH(
      v1.grad_fn(),
      "which does not have a derivative implemented is forbidden");
}

TEST(TestAutogradNotImplementedFallback, ViewOpWithExtraArg) {
  REGISTER_TEST_OP(
      "view_op_with_extra_arg",
      "_test::view_op_with_extra_arg(Tensor(a) self, Tensor other) -> Tensor(a)",
      view_op_with_extra_arg);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::view_op_with_extra_arg", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  assertBasicChecks(op);
  auto a = torch::tensor({1.}, {torch::kFloat32});
  auto b = torch::tensor({2.}, {torch::kFloat32});
  auto out1 = op(a, b);
  ASSERT_TRUE(out1.is_view());
  ASSERT_EQ(out1._base().unsafeGetTensorImpl(), a.unsafeGetTensorImpl());
}

TEST(TestAutogradNotImplementedFallback, RetTensorVectorView) {
  REGISTER_TEST_OP(
      "ret_tensor_vector_view",
      "_test::ret_tensor_vector_view(Tensor(a) self, Tensor other) -> Tensor[](a)",
      ret_tensor_vector_view);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_tensor_vector_view", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::vector<at::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  auto a = torch::tensor({1.}, {torch::kFloat32});
  auto b = torch::tensor({1.}, {torch::kFloat32});
  auto out = op(a, b);
  ASSERT_TRUE(out[0].is_view());
  ASSERT_EQ(out[0]._base().unsafeGetTensorImpl(), a.unsafeGetTensorImpl());
  ASSERT_TRUE(out[1].is_view());
  ASSERT_EQ(out[1]._base().unsafeGetTensorImpl(), a.unsafeGetTensorImpl());
}

TEST(TestAutogradNotImplementedFallback, DoubleViewOP) {
  REGISTER_TEST_OP(
      "two_pairs_of_view_op",
      "_test::two_pairs_of_view_op(Tensor(a) self, Tensor(b) other) -> (Tensor(a), Tensor(b))",
      two_pairs_of_view_op);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::two_pairs_of_view_op", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});
  ASSERT_THROWS_WITH(
      op(a, b),
      "Expected only a single output in the operator schema to have a non-write alias annotation");
}

TEST(TestAutogradNotImplementedFallback, NonFirstViewOP) {
  REGISTER_TEST_OP(
      "non_first_view_op",
      "_test::non_first_view_op(Tensor self, Tensor(b) other) -> (Tensor, Tensor(b))",
      non_first_view_op);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::non_first_view_op", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});
  ASSERT_THROWS_WITH(
      op(a, b), "can only create view relationships between the first");
}

TEST(TestAutogradNotImplementedFallback, RetTensorVector) {
  REGISTER_TEST_OP(
      "ret_tensor_vector",
      "_test::ret_tensor_vector(Tensor self, Tensor other) -> Tensor[]",
      ret_tensor_vector);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_tensor_vector", "");
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::vector<at::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2)[0];
  };
  assertBasicChecks(op);
}

TEST(TestAutogradNotImplementedFallback, TensorlistOp) {
  REGISTER_TEST_OP(
      "tensorlist_op",
      "_test::tensorlist_op(Tensor self, Tensor[] other) -> Tensor",
      tensorlist_op);
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::tensorlist_op", "");
  auto op = [&](torch::Tensor _1, at::TensorList _2) {
    return callOpUnboxed<torch::Tensor, const torch::Tensor&, at::TensorList>(
        opHandle, _1, _2);
  };

  auto a = torch::tensor({1.}, {torch::kFloat32});
  auto b = torch::tensor({1.}, {torch::kFloat32});
  auto c = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  std::vector<torch::Tensor> vec = {b, c};
  auto out = op(a, vec);

  ASSERT_THROWS_WITH(
      torch::autograd::grad({out}, {vec[0]}),
      "One of the differentiated Tensors does not require grad");
  ASSERT_THROWS_WITH(
      torch::autograd::grad({out}, {vec[1]}), "is not implemented");

  ASSERT_TRUE(at::allclose(op(a, vec), tensorlist_op(a, vec)));
}

// TODO add these tests if needed
// test_once_differentiable
// test_sparse_backward
// test_save_output_nr
// test_free_deep_graph_pyfunction
// test_naughty_anomaly_access
// test_naughty_autograd-function_stashing_ctx
// test_custom_autograd_repeated_grad_grad
// test_return_leaf
// test_anomaly_detect_nan
// test_no_grad_copy
