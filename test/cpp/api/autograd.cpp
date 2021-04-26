#include <gtest/gtest.h>

#include <torch/torch.h>

#include <torch/csrc/autograd/functions/basic_ops.h>

#include <test/cpp/api/support.h>

using namespace torch::autograd;
using namespace torch::test;

#define ASSERT_VARIABLE_EQ(a,b) ASSERT_TRUE(torch::allclose((a),(b)))
#define EXPECT_VARIABLE_EQ(a,b) EXPECT_TRUE(torch::allclose((a),(b)))

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

Variable simple_fn(const Variable& x, const Variable& y) {
  return x + 2 * y + x * y;
}

TEST(AutogradAPITests, BackwardSimpleTest) {
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  auto res = simple_fn(x, y);
  backward({res.sum()}, {});

  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({2, 2}));
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({2, 2})*2);
}

TEST(AutogradAPITests, BackwardTest) {
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  auto res = simple_fn(x, y);
  backward({res}, {torch::ones({2, 2})}, {}, true);

  backward({res}, {torch::ones({2, 2})});

  ASSERT_VARIABLE_EQ(x.grad(), 2* (y + torch::ones({2, 2})));
  ASSERT_VARIABLE_EQ(y.grad(), 2 * (x + torch::ones({2, 2})*2));
}

TEST(AutogradAPITests, GradSimpleTest) {
  // basic grad
  Variable x = torch::randn({2,2}, torch::requires_grad());
  Variable y = torch::randn({2,2}, torch::requires_grad());
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

  for (int i = 0; i < 5; ++ i) {
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
  ASSERT_THROWS_WITH(grad({x * 2}, {x, y}, {}, {}, false, false), "Set allow_unused=True");
}

TEST(CustomAutogradTest, GradUnreachableDiscoveryTest) {
  // Test that certain nodes are not erroneously executed when an input
  // is unreachable. See #39784
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable var) {
      return var;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
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

TEST(AutogradAPITests, RetainGrad) {
  auto input = torch::rand({1, 3}, torch::requires_grad());
  auto h1 = input * 3;
  auto out = (h1 * h1).sum();

  // It should be possible to call retain_grad() multiple times
  h1.retain_grad();
  h1.retain_grad();

  // Gradient should be accumulated
  out.backward({}, /*keep_graph=*/true);
  ASSERT_VARIABLE_EQ(h1 * 2, h1.grad());
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
  {
    WarningCapture warnings;
    // Double backward
    auto x = torch::tensor({0.0}, torch::requires_grad());
    auto y = x.pow(1.5);
    auto gr =
        grad({y}, {x}, {}, /*retain_graph=*/true, /*create_backward=*/true);
    ASSERT_THROWS_WITH(grad({gr[0]}, {x}, {torch::tensor({0.0})});, "returned nan");
    auto msgs = warnings.messages();
    ASSERT_EQ(msgs.size(), 2);
    ASSERT_TRUE(
        msgs[0].find("Traceback of forward call that caused the error") !=
        std::string::npos);
    ASSERT_TRUE(
        msgs[1].find(
            "Traceback of forward call that induced the previous calculation") !=
        std::string::npos);
  }
}

TEST(CustomAutogradTest, CustomFunction) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable var1, int mul, Variable var2) {
      ctx->saved_data["mul"] = mul;
      ctx->save_for_backward({var1, var2});
      return var1 + mul*var2 + var1*var2;
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
  auto res = MyFunction::apply(x,2,y);
  auto go = torch::ones({}, torch::requires_grad());
  res.sum().backward(go, false, true);

  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({5,5}));
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({5,5})*2);
}

TEST(CustomAutogradTest, FunctionReturnsInput) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable var1) {
      return var1;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      return {grad_output[0]*2};
    }
  };

  Variable x(torch::ones(1, torch::requires_grad()));
  MyFunction::apply(x).backward(torch::ones(1) , true, true);
  ASSERT_VARIABLE_EQ(x.grad(), torch::full(1, 2.));
}

TEST(CustomAutogradTest, FunctionReturnsUndefined) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable var) {
      return var * 2;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
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
    {MyFunction::apply(x)}, {x}, {}, false, false, true)[0].defined());
}

TEST(CustomAutogradTest, MaterializeGrads) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable var) {
      return var;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      EXPECT_VARIABLE_EQ(grad_output[0], torch::zeros(1));
      return grad_output;
    }
  };

  auto x = torch::ones(1, torch::requires_grad());
  UndefinedGrad().apply({MyFunction::apply(x)})[0].backward();
}

TEST(CustomAutogradTest, DontMaterializeGrads) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable var) {
      ctx->set_materialize_grads(false);
      return var;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
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
   static Variable forward(AutogradContext *ctx, Variable x) {
     return x+1;
   }

   static variable_list backward(AutogradContext *ctx, variable_list dy) {
     return dy;
   }
 };

 auto x = torch::ones({5,5}, torch::requires_grad());
 {
    at::NoGradGuard no_grad;
    auto y = MyOp::apply(x);
    ASSERT_FALSE(y.requires_grad());
 }
}

TEST(CustomAutogradTest, MarkDirty) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable v) {
      // Change the value inplace
      auto v_data = v.data_ptr<float>();
      v_data[0] = 2;
      ctx->mark_dirty({v});
      return v;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      return { (grad_output[0]*2.0) };
    }
  };

  // Clone here because modifying leafs inplace is not allowed
  auto x = torch::randn({5,5}, torch::requires_grad()).clone();
  auto version_before = x._version();
  auto out = MyFunction::apply(x);
  auto version_after = x._version();
  ASSERT_TRUE(version_after >= (version_before + 1));
  out.sum().backward();
}

TEST(CustomAutogradTest, MarkNonDifferentiable) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable v) {
      Variable output = v > 0;
      ctx->mark_non_differentiable({output});
      return output;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      return { (grad_output[0]*0.0) };
    }
  };

  auto x = torch::randn({5,5}, torch::requires_grad());
  auto mask = MyFunction::apply(x);
  ASSERT_FALSE(mask.requires_grad());
  auto y = x.masked_fill(mask, 0);
  y.sum().backward();
}

TEST(CustomAutogradTest, MarkNonDifferentiableMixed) {
  struct MyFunction : public Function<MyFunction> {
    static variable_list forward(AutogradContext *ctx, Variable input) {
      Variable a = input+1;
      Variable b = input+2;
      ctx->mark_non_differentiable({a});
      return {a,b};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      const Variable &grad_a = grad_output[0], &grad_b = grad_output[1];
      EXPECT_VARIABLE_EQ(grad_a, torch::zeros({5,5}));
      EXPECT_VARIABLE_EQ(grad_b, torch::ones({5,5}));
      return {grad_b};
    }
  };

  auto x = torch::randn({5,5}, torch::requires_grad());
  auto out = MyFunction::apply(x);

  ASSERT_FALSE(out[0].requires_grad());
  ASSERT_TRUE(out[1].requires_grad());
  out[1].sum().backward();
  ASSERT_VARIABLE_EQ(x.grad(), torch::ones({5,5}));
}

TEST(CustomAutogradTest, MarkNonDifferentiableNone) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable input) {
      auto output = input.clone();
      ctx->mark_non_differentiable({output});
      return output;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
      return {};
    }
  };

  auto x = torch::randn({5,5}, torch::requires_grad());
  auto r = MyFunction::apply(x * x);
  (r * x).sum().backward();
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

TEST(CustomAutogradTest, ReturnDuplicateInplace) {
  struct DoubleInplace : public Function<DoubleInplace> {
    static variable_list forward(AutogradContext *ctx, Variable x) {
      x.mul_(2);
      ctx->mark_dirty({x});
      return {x,x};
    }

    static variable_list backward(AutogradContext *ctsx, variable_list grad_outputs) {
      return {grad_outputs[0]*2 + grad_outputs[1]*2};
    }
  };

  auto x = torch::randn({5,5}, torch::requires_grad());

  ASSERT_THROWS_WITH(DoubleInplace::apply(x), "leaf Variable that requires grad");
  // TODO ASSERT_THROWS_WITH(DoubleInplace::apply(x.clone()[0]), "only one output");

  auto out = DoubleInplace::apply(x.clone());
  ASSERT_TRUE(torch::equal(out[0],out[1]));
}

TEST(CustomAutogradTest, ReturnDuplicate) {
  struct DoubleDuplicate : public Function<DoubleDuplicate> {
    static variable_list forward(AutogradContext *ctx, Variable x) {
      auto output = x*2;
      return {output, output};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
      return {grad_outputs[0]*2 + grad_outputs[1]*2};
    }
  };

  auto x = torch::randn({5,5}, torch::requires_grad());
  auto out = DoubleDuplicate::apply(x);
  ASSERT_TRUE(torch::equal(out[0],out[1]));
}

TEST(CustomAutogradTest, SaveEmptyForBackward) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable input) {
      ctx->save_for_backward({Variable(), input, Variable()});
      return input*input;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      auto saved = ctx->get_saved_variables();
      EXPECT_FALSE(saved[0].defined());
      EXPECT_FALSE(saved[2].defined());
      return {saved[1] * 2 * grad_output[0]};
    }
  };

  Variable x = torch::randn({5,5}, torch::requires_grad());
  auto y = MyFunction::apply(x);
  y.sum().backward();
  ASSERT_VARIABLE_EQ(x.grad(), 2*x);
}

TEST(CustomAutogradTest, InvalidGradients) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext *ctx, Variable x) {
      return x*2;
    }

    static variable_list backward(AutogradContext *ctsx, variable_list grad_outputs) {
      return {torch::randn(10, torch::dtype(torch::kFloat).requires_grad(true))};
    }
  };

  auto input1 = torch::randn({5,5}, torch::dtype(torch::kFloat).requires_grad(true));
  ASSERT_THROWS_WITH(
    MyFunction::apply(input1).sum().backward(), "expected shape");
  auto input2 = torch::randn(10, torch::dtype(torch::kDouble).requires_grad(true));
}

TEST(CustomAutogradTest, NoGradInput) {
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext*, Variable x) {
      return x;
    }

    static variable_list backward(AutogradContext*, variable_list grad_outputs) {
      return grad_outputs;
    }
  };

  Variable x = torch::randn({5,5}, torch::requires_grad());
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
    static variable_list forward(AutogradContext *ctx, Variable input) {
      auto out = torch::randn(input.sizes());
      ctx->mark_non_differentiable({out});
      return {input, out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
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
  b = b+1; // Separate F1 and F2 by another operation
  ASSERT_TRUE(a.requires_grad());
  ASSERT_FALSE(b.requires_grad());

  auto c = F2::apply(a,b);
  c.backward(torch::ones(c.sizes()), false, false);
  ASSERT_VARIABLE_EQ(x.grad(), torch::ones(x.sizes()));
}

TEST(CustomAutogradTest, Reentrant) {
  static Variable y_data = torch::randn({2, 2});
  struct Reenter : public Function<Reenter> {
    static Variable forward(AutogradContext *ctx, Variable input) {
      Variable output;
      {
        at::AutoGradMode enable_grad(true);
        auto x = make_variable(input.tensor_data(), true);
        auto y = make_variable(y_data.tensor_data(), true);
        output = x*y;

        ctx->saved_data["x"] = x;
        ctx->saved_data["y"] = y;
        ctx->saved_data["output_var"] = output;
      }
      return output.detach();
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_output) {
      {
        at::AutoGradMode enable_grad(true);
        auto out = ctx->saved_data["output_var"].toTensor();
        out.sum().backward();
      }
      return {ctx->saved_data["x"].toTensor().grad() * grad_output[0]};
    }
  };

  auto x = torch::randn({2,2}, torch::requires_grad());
  auto out = Reenter::apply(x);
  out.sum().backward();
  ASSERT_VARIABLE_EQ(x.grad(), y_data);
}


// NOTE: If this fails for apparently unrelated reasons in TSAN be aware of
// the TSAN limit on mutex: https://github.com/google/sanitizers/issues/950
TEST(CustomAutogradTest, DeepReentrant) {
  struct DeepReenter : public Function<DeepReenter> {
    static Variable forward(AutogradContext *ctx, Variable x) {
      {
        at::AutoGradMode enable_grad(true);
        ctx->saved_data["x"] = make_variable(x.tensor_data(), true) -1;
      }
      return ctx->saved_data["x"].toTensor().detach();
    }

    static variable_list backward(AutogradContext*ctx, variable_list grad_output) {
      if (!ctx->saved_data["x"].toTensor().is_nonzero()) {
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
  auto v = torch::tensor({8193}, torch::dtype(torch::kFloat).requires_grad(true));
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
    static Variable forward(AutogradContext *ctx, Variable x) {
      {
        at::AutoGradMode enable_grad(true);
        ctx->saved_data["x"] = make_variable(x.tensor_data(), true) -1;
      }
      return ctx->saved_data["x"].toTensor().detach();
    }

    static variable_list backward(AutogradContext*ctx, variable_list grad_output) {
      order.push_back(1);
      if (!ctx->saved_data["x"].toTensor().is_nonzero()) {
        return grad_output;
      }
      {
        at::AutoGradMode enable_grad(true);
        apply(ctx->saved_data["x"].toTensor())[0].sum().backward();
        return grad_output;
      }
    }
  };

  auto a = MyFunction::apply(torch::tensor({6}, torch::dtype(torch::kFloat).requires_grad(true)));
  auto b = Reenter::apply(torch::tensor({9}, torch::dtype(torch::kFloat).requires_grad(true)));
  auto v = a*b;
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
  Variable x = torch::ones({5,5}, torch::requires_grad());
  Variable y = torch::ones({5,5})*4;
  y.set_requires_grad(true);

  int counter = 0;

  std::function<void(int, Variable)> bw_hook([&counter](int inc, Variable grad){
    counter += inc;
  });

  Variable z = x * x + x * 2 + x * y + y;
  x.register_hook([&bw_hook](Variable grad){
    bw_hook(0, grad);
  });
  auto hook_1 = z.register_hook([&bw_hook](Variable grad){
    bw_hook(1, grad);
  });
  z.backward(torch::ones({5,5}), true, true);
  ASSERT_EQ(counter, 1);

  auto hook_2 = z.register_hook([&bw_hook](Variable grad){
    bw_hook(2, grad);
  });
  z.backward(torch::ones({5,5}), true, true);
  ASSERT_EQ(counter, 4);

  z.remove_hook(hook_2);
  z.backward(torch::ones({5,5}), true, true);
  ASSERT_EQ(counter, 5);

  std::function<Variable(Variable)> bw_hook_modify([](Variable grad){
    return grad.mul(2);
  });

  z.remove_hook(hook_1);
  z.register_hook(bw_hook_modify);
  y.grad().zero_();
  z.backward(torch::ones({5,5}), true, false);
  ASSERT_VARIABLE_EQ(y.grad(), (x+1)*2);

  y.register_hook(bw_hook_modify);
  y.grad().zero_();
  z.backward(torch::ones({5,5}), false, false);
  ASSERT_VARIABLE_EQ(y.grad(), (x+1)*4);

  ASSERT_THROWS_WITH(y.remove_hook(3), "Invalid index");
}

TEST(CustomAutogradTest, HookNone) {
  struct NoneGradientFunction : public Function<NoneGradientFunction> {
    static variable_list forward(AutogradContext *ctx, Variable x, Variable y) {
      return {x,y};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad) {
      return {grad[0], Variable()};
    }
  };

  bool was_called = false;

  auto hook = ([&was_called](Variable grad){
    ASSERT_TRUE(grad.defined());
    was_called = true;
  });

  auto x = torch::randn({5,5}, torch::requires_grad());
  auto y = torch::randn({5,5});

  auto out = NoneGradientFunction::apply(x,y);
  Variable rx = x[0], ry = x[1];

  rx.register_hook(hook);
  ry.register_hook(hook);
  (rx+ry).sum().backward();
  ASSERT_TRUE(was_called);
}

TEST(CustomAutogradTest, BackwardWithInputs) {
  Variable x = torch::randn({5,5}, torch::requires_grad());
  Variable y = torch::randn({5,5}, torch::requires_grad());
  Variable z = x * x + x * y + y * y;
  Variable x_grad_expected = 2 * x + y;
  Variable y_grad_expected = x + 2 * y;

  z.backward(torch::ones({5, 5}), false, false, {x});

  ASSERT_VARIABLE_EQ(x.grad(), x_grad_expected);
  ASSERT_FALSE(y.grad().defined());
}

TEST(CustomAutogradTest, BackwardWithEmptyInputs) {
  Variable x = torch::randn({5,5}, torch::requires_grad());
  Variable y = torch::randn({5,5}, torch::requires_grad());
  Variable z = x * x + x * y + y * y;
  Variable x_grad_expected = 2 * x + y;
  Variable y_grad_expected = x + 2 * y;
  ASSERT_THROWS_WITH(z.backward(torch::ones({5, 5}), false, false, std::vector<Variable>{}), "cannot be empty");
}

TEST(CustomAutogradTest, BackwardWithNonLeafInputs) {
  Variable x = torch::randn({5,5}, torch::requires_grad());
  Variable y = torch::randn({5,5}, torch::requires_grad());
  Variable z = x * x;
  Variable w = z + x * y + y * y;
  ASSERT_THROWS_WITH(w.backward(torch::ones({5, 5}), false, false, {z}), "is not a leaf Tensor");
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
