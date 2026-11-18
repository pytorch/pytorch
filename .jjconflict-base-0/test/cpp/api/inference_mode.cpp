#include <gtest/gtest.h>
#include <test/cpp/api/support.h>
#include <torch/script.h>

using namespace torch::autograd;
using namespace torch::test;

namespace {
torch::Tensor functional_op(torch::Tensor& x) {
  return x * x;
}

void inplace_op(torch::Tensor& x) {
  x.mul_(1);
}

torch::Tensor view_op(torch::Tensor& x) {
  return x.view({2, 3});
}

/*
  Only the following combos of Autograd & ADInplaceOrView keys on tensors are
  valid:
    - Autograd=true, ADInplaceOrView=true (normal tensor)
    - Autograd=false, ADInplaceOrView=false (inference tensor)
  Tensors created in InferenceMode are mostly inference tensors. The only
  exception is that view of normal tensors created in InferenceMode still
  produce normal tensor.
*/
void assert_TLS_states(bool inference_mode) {
  ASSERT_EQ(InferenceMode::is_enabled(), inference_mode);
  ASSERT_FALSE(c10::impl::tls_is_dispatch_key_excluded(
      c10::DispatchKey::ADInplaceOrView));
  ASSERT_FALSE(c10::impl::tls_is_dispatch_keyset_included(
      c10::autograd_dispatch_keyset));
  ASSERT_EQ(
      c10::impl::tls_is_dispatch_keyset_excluded(c10::autograd_dispatch_keyset),
      inference_mode);
  ASSERT_EQ(
      c10::impl::tls_is_dispatch_key_included(
          c10::DispatchKey::ADInplaceOrView),
      !inference_mode);
  ASSERT_EQ(GradMode::is_enabled(), !inference_mode);
}
} // namespace

TEST(InferenceModeTest, TestTLSState) {
  assert_TLS_states(false);
  {
    InferenceMode guard;
    assert_TLS_states(true);
    {
      InferenceMode guard(false);
      assert_TLS_states(false);
    }
    assert_TLS_states(true);
  }
  assert_TLS_states(false);
}

TEST(InferenceModeTest, TestInferenceTensorCreation) {
  {
    InferenceMode guard;
    // New tensor created through constructors are inference tensors.
    torch::Tensor c = torch::ones({1, 2, 3});
    ASSERT_FALSE(c.requires_grad());
    ASSERT_TRUE(c.is_inference());

    // requires_grad doesn't change inference tensor behavior inside
    // InferenceMode.
    torch::Tensor tmp = torch::ones({1, 2, 3}).set_requires_grad(true);
    ASSERT_TRUE(tmp.requires_grad());
    ASSERT_TRUE(tmp.is_inference());

    tmp = torch::ones({1, 2, 3}).set_requires_grad(false);
    ASSERT_FALSE(tmp.requires_grad());
    ASSERT_TRUE(tmp.is_inference());
  }
}

TEST(InferenceModeTest, TestExistingAutogradSession) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s.clone();

  // Save `a` in an existing autograd session
  torch::Tensor out = a * a;
  {
    InferenceMode guard;
    inplace_op(a);
  }
  // Performing backward should trigger error since `a`'s version has been
  // bumped.
  ASSERT_THROWS_WITH(
      out.backward(torch::ones_like(out)),
      "one of the variables needed for gradient computation has been modified by an inplace operation")
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeFunctionalOp) {
  c10::InferenceMode guard;
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor func_out = functional_op(c); // go through kernels: CPU
    ASSERT_TRUE(func_out.is_inference());
    ASSERT_FALSE(func_out.requires_grad());
  }
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeInplaceOp) {
  c10::InferenceMode guard;
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    inplace_op(c); // go through kernels: CPU
    ASSERT_TRUE(c.is_inference());
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeViewOp) {
  c10::InferenceMode guard;
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor view_out = view_op(c); // go through kernels: CPU
    ASSERT_TRUE(view_out.is_inference());
    // Note this is different from NoGradMode but makes sense.
    ASSERT_FALSE(view_out.requires_grad());
    ASSERT_FALSE(view_out.is_view());
  }
}

TEST(InferenceModeTest, TestInferenceTensorInNormalModeFunctionalOp) {
  torch::Tensor inference_tensor;
  for (bool requires_grad : {true, false}) {
    {
      InferenceMode guard;
      inference_tensor =
          torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }

    // Due to issue #54614, this might run slower compared to InferenceMode
    // since intermediate tensors are normal tensors, and they might dispatch to
    // VariableType kernels. This is fine since users can easily fix it by
    // moving it inside InferenceMode block.
    torch::Tensor tmp =
        functional_op(inference_tensor); // go through kernels:
                                         // ADInplaceOrView(fallthrough), CPU
    ASSERT_FALSE(tmp.is_inference());
    ASSERT_FALSE(tmp.requires_grad());
  }
}

TEST(InferenceModeTest, TestInferenceTensorInNormalModeInplaceOp) {
  torch::Tensor inference_tensor;
  for (bool requires_grad : {true, false}) {
    {
      InferenceMode guard;
      inference_tensor =
          torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }
    ASSERT_THROWS_WITH(
        inplace_op(
            inference_tensor), // go through kernels: ADInplaceOrView, CPU
        "Inplace update to inference tensor outside InferenceMode is not allowed");
  }
}

TEST(InferenceModeTest, TestInferenceTensorInNormalModeViewOp) {
  torch::Tensor inference_tensor;
  for (bool requires_grad : {true, false}) {
    {
      InferenceMode guard;
      inference_tensor =
          torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }
    torch::Tensor out =
        view_op(inference_tensor); // go through kernels: ADInplaceOrView, CPU
    ASSERT_TRUE(out.is_inference());
    ASSERT_FALSE(out.requires_grad());
    ASSERT_FALSE(out.is_view());
    ASSERT_TRUE(out.is_leaf());
  }
}

TEST(InferenceModeTest, TestNormalTensorInplaceOutputInInferenceMode) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();

    {
      c10::InferenceMode guard;

      inplace_op(a); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(a.is_inference());
      ASSERT_EQ(a.requires_grad(), requires_grad);

      // inplace -> inplace
      inplace_op(a); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(a.is_inference());
      ASSERT_EQ(a.requires_grad(), requires_grad);

      // inplace -> inplace -> view
      torch::Tensor view_out =
          view_op(a); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(view_out.is_inference());
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
    }
  }
}

TEST(InferenceModeTest, TestNormalTensorInplaceOutputInNormalMode) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();

    {
      c10::InferenceMode guard;

      inplace_op(a); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(a.is_inference());
      ASSERT_EQ(a.requires_grad(), requires_grad);
    }

    torch::Tensor tmp = functional_op(a); // go through kernels: VariableType,
                                          // ADInplaceOrView(fallthrough), CPU
    ASSERT_FALSE(tmp.is_inference());
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    inplace_op(a); // go through kernels: VariableType, ADInplaceOrView, CPU
    ASSERT_FALSE(a.is_inference());
    ASSERT_EQ(a.requires_grad(), requires_grad);

    tmp = view_op(a); // go through kernels: VariableType, ADInplaceOrView, CPU
    ASSERT_FALSE(tmp.is_inference());
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestNormalTensorViewOutputInInferenceMode) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      c10::InferenceMode guard;
      // View ops on normal tensor produce normal tensors as output.
      // - For view ops it has both dispatch keys since due to the way we create
      //   view Tensors in alias_with_sizes_and_strides:
      //   ```
      //     auto impl = c10::make_intrusive<TensorImpl>(
      //     Storage(self.storage()), self.key_set(), self.dtype());
      //   ```
      //   In addition, these view output tensors are normal in the sense they
      //   have both Autograd and ADInplaceOrView keys. But they're still
      //   special since they'll have CreationMeta::INFERENCE_MODE. In other
      //   words they behave exactly the same as a view tensor created in
      //   no_grad mode.

      view_out = view_op(a); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(view_out.is_inference());
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());

      // view -> view
      tmp = view_op(view_out); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(tmp.is_inference());
      assert_tensor_creation_meta(tmp, CreationMeta::INFERENCE_MODE);
      ASSERT_EQ(tmp.requires_grad(), requires_grad);
      ASSERT_TRUE(tmp.is_leaf());

      // view -> view -> inplace
      inplace_op(tmp); // kernels: ADInplaceOrView, CPU
      assert_tensor_creation_meta(tmp, CreationMeta::INFERENCE_MODE);
      ASSERT_FALSE(tmp.is_inference());
      ASSERT_EQ(tmp.requires_grad(), requires_grad);
      ASSERT_TRUE(tmp.is_leaf());
      ASSERT_EQ(a._version(), tmp._version());
    }
  }
}

TEST(InferenceModeTest, TestNormalTensorViewOutputInNormalMode) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      c10::InferenceMode guard;
      view_out = view_op(a); // go through kernels: ADInplaceOrView, CPU
      ASSERT_FALSE(view_out.is_inference());
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());
    }

    tmp = functional_op(view_out);
    ASSERT_FALSE(view_out.is_inference());
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    if (requires_grad) {
      ASSERT_THROWS_WITH(
          inplace_op(view_out), // go through kernels: VariableType,
                                // ADInplaceOrView, CPU
          "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out);
    }

    tmp = view_op(view_out);
    ASSERT_FALSE(view_out.is_inference());
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorFunctionalOp) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      InferenceMode guard;
      c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }

    // add(Tensor, Tensor) is safe with inference tensor since it doesn't save
    // any variable for backward.
    torch::Tensor out = c.add(s); // go through kernels: VariableType,
                                  // ADInplaceOrView(fallthrough), CPU
    ASSERT_FALSE(out.is_inference());
    ASSERT_EQ(out.requires_grad(), requires_grad);
    if (requires_grad) {
      // leaf inference tensor with requires_grad=true can still have gradient.
      // Note this behavior is different from NoGradMode which has empty grad.
      out.backward(torch::ones_like(out));
      assert_tensor_equal(c.grad(), torch::ones_like(c));
    }

    if (requires_grad) {
      // mul(self, other) saves variable when requires_grad=true
      ASSERT_THROWS_WITH(
          c.mul(s), "Inference tensors cannot be saved for backward.");

      // Inference tensor in TensorList input
      // stack does not capture anymore, so disabled
      // TODO: find alternative Function that captures a list (maybe custom fn)
      /*
      std::vector<torch::Tensor> inputs = {s, c};
      ASSERT_THROWS_WITH(
          torch::stack(inputs), // go through kernels: VariableType(ERROR)!,
                                // ADInplaceOrView(fallthrough), CPU
          "Inference tensors cannot be saved for backward.")
      */
    }
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorInplaceOp) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor c;
    {
      InferenceMode guard;
      c = torch::ones({1, 2, 3});
    }

    if (requires_grad) {
      ASSERT_THROWS_WITH(
          a.mul_(c), // go through kernels: VariableType(ERROR!), InferenceMode,
                     // CPU
          "Inference tensors cannot be saved for backward.");

      ASSERT_THROWS_WITH(
          torch::mul_out(
              /*out=*/c, s, s), // go through kernels: VariableType(ERROR!),
                                // ADInplaceOrView, CPU
          "out=... arguments don't support automatic differentiation, but one of the arguments requires grad")
    } else {
      a.mul_(c);

      ASSERT_THROWS_WITH(
          torch::mul_out(/*out=*/c, s, s), // go through kernels: VariableType,
                                           // ADInplaceOrView(ERROR!), CPU
          "Inplace update to inference tensor outside InferenceMode is not allowed");
    }
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorViewOp) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      InferenceMode guard;
      c = torch::ones({1, 2, 3});
    }

    // view_as is a composite op which calls view() with only one tensor
    // argument. So there isn't a mixed inference tensor and normal tensor
    // inputs for view ops.
    torch::Tensor tmp1 =
        c.view_as(s); // go through kernels: ADInplaceOrView, CPU
    ASSERT_TRUE(tmp1.is_inference());
    ASSERT_FALSE(tmp1.requires_grad());

    // This is fine since it's equivalent as s.view(c.sizes()) which
    // isn't a mixed input scenario.
    torch::Tensor tmp2 =
        s.view_as(c); // go through kernels: VariableType, ADInplaceOrView, CPU
    ASSERT_FALSE(tmp2.is_inference());
    ASSERT_EQ(tmp2.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestHandleDirectViewOnRebase) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out;
    {
      InferenceMode guard;
      view_out = view_op(a); // go through kernels: ADInplaceOrView, CPU
    }
    if (requires_grad) {
      ASSERT_THROWS_WITH(
          inplace_op(view_out),
          "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out);
    }
  }
}

TEST(InferenceModeTest, TestHandleInDirectViewOnRebase) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out;
    {
      InferenceMode guard;
      view_out = view_op(a); // go through kernels: ADInplaceOrView, CPU
    }
    inplace_op(a);
    if (requires_grad) {
      ASSERT_THROWS_WITH(
          view_out.grad_fn(),
          "A view was created in inference mode and its base or another view of its base has been modified inplace");
    } else {
      view_out.grad_fn();
    }
  }
}

TEST(InferenceModeTest, TestCreationMetaPropagation) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor b, c;
  {
    InferenceMode guard;
    b = s.view_as(s);
  }
  ASSERT_THROWS_WITH(
      b.add_(1),
      "A view was created in inference mode and is being modified inplace");
  {
    AutoGradMode mode(false);
    c = b.view_as(b);
  }
  ASSERT_THROWS_WITH(
      c.add_(1),
      "A view was created in inference mode and is being modified inplace");
}

TEST(InferenceModeTest, TestCreationMetaPropagationInput) {
  torch::Tensor s = torch::ones({2, 2, 3}).set_requires_grad(true);
  auto s_view = s.view_as(s);
  std::vector<at::Tensor> b, c;
  {
    InferenceMode guard;
    b = s_view.split_with_sizes({1, 1});

    s = s.view_as(s);
    c = s.split_with_sizes({1, 1});
  }
  for (auto& b_el : b) {
    assert_tensor_creation_meta(b_el, CreationMeta::INFERENCE_MODE);
    ASSERT_THROWS_WITH(
        b_el.add_(1),
        "A view was created in inference mode and is being modified inplace");
  }
  for (auto& c_el : c) {
    assert_tensor_creation_meta(c_el, CreationMeta::INFERENCE_MODE);
    ASSERT_THROWS_WITH(
        c_el.add_(1),
        "A view was created in inference mode and is being modified inplace");
  }
}

TEST(InferenceModeTest, TestInplaceCopyOnInferenceTensor) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor t;
    {
      InferenceMode guard;
      t = torch::ones({1, 2, 3});
      t.copy_(s);
      ASSERT_TRUE(t.is_inference());
      ASSERT_FALSE(t.requires_grad());
    }

    ASSERT_THROWS_WITH(
        t.copy_(s),
        "Inplace update to inference tensor outside InferenceMode is not allowed");
  }
}

TEST(InferenceModeTest, TestSetRequiresGradInNormalMode) {
  torch::Tensor t;
  {
    InferenceMode guard;
    t = torch::ones({1, 2, 3});
  }
  t.set_requires_grad(false);
  ASSERT_THROWS_WITH(
      t.set_requires_grad(true),
      "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
}

TEST(InferenceModeTest, TestAccessVersionCounter) {
  torch::Tensor t;
  {
    InferenceMode guard;
    t = torch::ones({1, 2, 3});
    ASSERT_THROWS_WITH(
        t.unsafeGetTensorImpl()->version_counter().current_version(),
        "Inference tensors do not track version counter.");
    t.unsafeGetTensorImpl()->bump_version();
  }
  ASSERT_THROWS_WITH(
      t.unsafeGetTensorImpl()->version_counter().current_version(),
      "Inference tensors do not track version counter.");
  ASSERT_THROWS_WITH(
      t.unsafeGetTensorImpl()->bump_version(),
      "Inplace update to inference tensor outside InferenceMode is not allowed.");
  // Suggested workaround
  torch::Tensor c = t.clone();
  uint32_t v = c.unsafeGetTensorImpl()->version_counter().current_version();
  c.unsafeGetTensorImpl()->bump_version();
  ASSERT_EQ(
      c.unsafeGetTensorImpl()->version_counter().current_version(), v + 1);
}

TEST(InferenceModeTest, TestInplaceUpdateInferenceTensorWithNormalTensor) {
  torch::Tensor s = torch::ones({1, 2, 3});
  torch::Tensor t;
  {
    InferenceMode guard;
    t = torch::ones({1, 2, 3});
    // Testing both copy_ from VariableTypeManual and add_ from generated code.
    s.copy_(t);
    s.add_(t);
    t.add_(s);
    t.copy_(s);
  }
  s.copy_(t);
  s.add_(t);
  ASSERT_THROWS_WITH(
      t.copy_(s),
      "Inplace update to inference tensor outside InferenceMode is not allowed");

  ASSERT_THROWS_WITH(
      t.add_(s),
      "Inplace update to inference tensor outside InferenceMode is not allowed");
}

TEST(InferenceModeTest, TestComplexViewInInferenceMode) {
  torch::Tensor s = torch::ones({3, 3, 2});
  torch::Tensor t = torch::view_as_complex(s);
  {
    InferenceMode guard;
    torch::Tensor tmp;

    tmp = torch::view_as_real(t);
    ASSERT_FALSE(tmp.is_inference());
    tmp = torch::view_as_complex(s);
    ASSERT_FALSE(tmp.is_inference());

    torch::Tensor e = torch::ones({3, 3, 2});
    tmp = torch::view_as_complex(e);
    ASSERT_TRUE(tmp.is_inference());
    tmp = torch::view_as_real(tmp);
    ASSERT_TRUE(tmp.is_inference());
  }
}

TEST(InferenceModeTest, TestComplexViewInNormalMode) {
  torch::Tensor s;
  {
    InferenceMode guard;
    s = torch::ones({3, 3, 2});
  }
  torch::Tensor tmp = torch::view_as_complex(s);
  ASSERT_TRUE(tmp.is_inference());
  tmp = torch::view_as_real(tmp);
  ASSERT_TRUE(tmp.is_inference());
}

TEST(InferenceModeTest, TestCustomFunction) {
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

  {
    InferenceMode guard;
    torch::Tensor var1 = torch::ones({3, 3}).set_requires_grad(true);
    auto var2 = var1.clone();
    int mul = 2;
    // If InferenceMode didn't set NoGradGuard automatically, this line
    // would error out when trying to save `var1` and `var2` for backward.
    auto y = MyFunction::apply(var1, mul, var2);
    torch::Tensor expected = var1 + mul * var2 + var1 * var2;
    assert_tensor_equal(y, expected);
  }
}

TEST(InferenceModeTest, TestLegacyAutoNonVariableTypeModeWarning) {
  c10::WarningUtils::WarnAlways warn_always(true);
  WarningCapture warnings;
  at::AutoNonVariableTypeMode guard;
  ASSERT_TRUE(
      warnings.str().find("AutoNonVariableTypeMode is deprecated") !=
      std::string::npos);
}
