#include <torch/script.h>
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

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
    Only the following combos of Autograd & InplaceOrView keys on tensors are valid:
      - Autograd=true, InplaceOrView=true (normal tensor)
      - Autograd=false, InplaceOrView=false (inference tensor)
    Tensors created in InferenceMode are mostly inference tensors. The only exception
    is that view of normal tensors created in InferenceMode still produce normal tensor.
  */
  bool is_inference_tensor(torch::Tensor& x) {
    c10::DispatchKeySet ks = x.key_set();
    bool has_Autograd = ks.has(c10::DispatchKey::AutogradCPU);
    bool has_InplaceOrView = ks.has(c10::DispatchKey::InplaceOrView);
    // They must be either both true or false.
    bool is_inference_tensor = !has_Autograd && !has_InplaceOrView && x.is_leaf();
    return is_inference_tensor;
  }

  void assert_TLS_states(bool inference_mode) {
    ASSERT_EQ(InferenceMode::is_enabled(), inference_mode);
    ASSERT_FALSE(c10::impl::tls_is_dispatch_key_excluded(c10::DispatchKey::InplaceOrView));
    ASSERT_FALSE(c10::impl::tls_is_dispatch_keyset_included(c10::autograd_dispatch_keyset));
    ASSERT_EQ(c10::impl::tls_is_dispatch_keyset_excluded(c10::autograd_dispatch_keyset), inference_mode);
    ASSERT_EQ(c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::InplaceOrView), !inference_mode);
  }
}

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
    ASSERT_TRUE(is_inference_tensor(c));

    // requires_grad doesn't change inference tensor behavior inside InferenceMode.
    torch::Tensor tmp = torch::ones({1, 2, 3}).set_requires_grad(true);
    ASSERT_TRUE(tmp.requires_grad());
    ASSERT_TRUE(is_inference_tensor(tmp));

    tmp = torch::ones({1, 2, 3}).set_requires_grad(false);
    ASSERT_FALSE(tmp.requires_grad());
    ASSERT_TRUE(is_inference_tensor(tmp));
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
  // perform backward on `a` should trigger error since `a`'s version has been bumped.
  ASSERT_THROWS_WITH(out.backward(torch::ones_like(out)),
    "one of the variables needed for gradient computation has been modified by an inplace operation")
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeFunctionalOp) {
  c10::InferenceMode guard;
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor func_out = functional_op(c);  // go through kernels: CPU
    ASSERT_TRUE(is_inference_tensor(func_out));
    ASSERT_FALSE(func_out.requires_grad());
  }
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeInplaceOp) {
  c10::InferenceMode guard;
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    inplace_op(c);  // go through kernels: CPU
    ASSERT_TRUE(is_inference_tensor(c));
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeViewOp) {
  c10::InferenceMode guard;
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor view_out = view_op(c);  // go through kernels: CPU
    ASSERT_TRUE(is_inference_tensor(view_out));
    // Note this is different from NoGradMode but makes sense.
    ASSERT_FALSE(view_out.requires_grad());
  }
}

TEST(InferenceModeTest, TestInferenceTensorInNormalModeFunctionalOp) {
  torch::Tensor inference_tensor;
  for (bool requires_grad: {true, false}) {
    {
      InferenceMode guard;
      inference_tensor = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }
    // Functional ops on inference tensors might run slower outside InferenceMode than inside.
    // But it's fine that we don't care about perf of this case that much.
    //
    // An alternative behavior we perfer but didn't implement is throwing an error by forcing
    // this op go through VariableType kernel and hit the assert_no_inference_tensor check.
    // But to do that we'll have to add c10::autograd_dispatch_keyset to the globally enabled set,
    // but doing that might accidentally call autograd kernel from a backend that doesn't match tensor input.
    // Thus we allow functional ops run without throwing an error.
    torch::Tensor tmp = functional_op(inference_tensor); // go through kernels: InplaceOrView(fallthrough), CPU
    ASSERT_FALSE(is_inference_tensor(tmp));
    ASSERT_FALSE(tmp.requires_grad());
  }
}

TEST(InferenceModeTest, TestInferenceTensorInNormalModeInplaceOp) {
  torch::Tensor inference_tensor;
  for (bool requires_grad: {true, false}) {
    {
      InferenceMode guard;
      inference_tensor = torch::ones({1, 2, 3});
    }
    ASSERT_THROWS_WITH(inplace_op(inference_tensor), // go through kernels: InplaceOrView(ERROR!), CPU
      "inplace/view ops on inference tensor outside InferenceMode");
  }
}

TEST(InferenceModeTest, TestInferenceTensorInNormalModeViewOp) {
  torch::Tensor inference_tensor;
  for (bool requires_grad: {true, false}) {
    {
      InferenceMode guard;
      inference_tensor = torch::ones({1, 2, 3});
    }
    ASSERT_THROWS_WITH(view_op(inference_tensor), // go through kernels: InplaceOrView(ERROR!), CPU
      "inplace/view ops on inference tensor outside InferenceMode")
  }
}

TEST(InferenceModeTest, TestNormalTensorInplaceOpInInferenceMode) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();

    {
      c10::InferenceMode guard;

      inplace_op(a);  // go through kernels: InplaceOrView, CPU
      ASSERT_FALSE(is_inference_tensor(a));
      ASSERT_EQ(a.requires_grad(), requires_grad);

      // inplace -> inplace
      inplace_op(a);  // go through kernels: InplaceOrView, CPU
      ASSERT_FALSE(is_inference_tensor(a));
      ASSERT_EQ(a.requires_grad(), requires_grad);

      // inplace -> inplace -> view
      torch::Tensor view_out = view_op(a);
      ASSERT_FALSE(is_inference_tensor(view_out));
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
    }
  }
}

TEST(InferenceModeTest, TestNormalTensorInplaceOutputInNormalMode) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();

    {
      c10::InferenceMode guard;

      inplace_op(a);  // go through kernels: InplaceOrView, CPU
      ASSERT_FALSE(is_inference_tensor(a));
      ASSERT_EQ(a.requires_grad(), requires_grad);
    }

    torch::Tensor tmp = functional_op(a);  // go through kernels: VariableType, InplaceOrView(fallthrough), CPU
    ASSERT_FALSE(is_inference_tensor(tmp));
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    inplace_op(a); // go through kernels: VariableType, InplaceOrView, CPU
    ASSERT_FALSE(is_inference_tensor(a));
    ASSERT_EQ(a.requires_grad(), requires_grad);

    tmp = view_op(a);  // go through kernels: VariableType, InplaceOrView, CPU
    ASSERT_FALSE(is_inference_tensor(tmp));
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestNormalTensorViewOpInInferenceMode) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
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
      //   have both Autograd and InplaceOrView keys. But they're still special
      //   since they'll have CreationMeta::INFERENCE_MODE. In other words they behave
      //   exactly the same as a view tensor created in no_grad mode.

      view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
      ASSERT_FALSE(is_inference_tensor(view_out));
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      ASSERT_TRUE(view_out.requires_grad());
      ASSERT_TRUE(view_out.is_leaf());

      // view -> view
      tmp = view_op(view_out);  // go through kernels: InplaceOrView, CPU
      ASSERT_FALSE(is_inference_tensor(tmp));
      assert_tensor_creation_meta(tmp, CreationMeta::INFERENCE_MODE);
      ASSERT_TRUE(tmp.requires_grad());
      ASSERT_TRUE(tmp.is_leaf());

      // view -> view -> inplace
      inplace_op(tmp);  // kernels: InplaceOrView, CPU
      assert_tensor_creation_meta(tmp, CreationMeta::INFERENCE_MODE);
      ASSERT_FALSE(is_inference_tensor(tmp));
      ASSERT_TRUE(tmp.requires_grad());
      ASSERT_TRUE(tmp.is_leaf());
      ASSERT_EQ(a._version(), tmp._version());
    }
  }
}

TEST(InferenceModeTest, TestNormalTensorViewOutputInNormalMode) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      c10::InferenceMode guard;
      view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
      ASSERT_FALSE(is_inference_tensor(view_out));
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());
    }

    tmp = functional_op(view_out);
    ASSERT_FALSE(is_inference_tensor(view_out));
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    if (requires_grad) {
      ASSERT_THROWS_WITH(inplace_op(view_out),  // go through kernels: VariableType, InplaceOrView, CPU
        "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out);
    }

    tmp = view_op(view_out);
    ASSERT_FALSE(is_inference_tensor(view_out));
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorFunctionalOp) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      InferenceMode guard;
      c = torch::ones({1, 2, 3});
    }

    ASSERT_THROWS_WITH(c.add(s), // go through kernels: VariableType(ERROR!), InplaceOrView(fallthrough), CPU
      "Inference tensor cannot participate in autograd")

    // Inference tensor in TensorList input
    std::vector<torch::Tensor> inputs = {s, c};
    ASSERT_THROWS_WITH(torch::cat(inputs), // go through kernels: VariableType(ERROR)!, InplaceOrView(fallthrough), CPU
      "Inference tensor cannot participate in autograd")
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorInplaceOp) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      InferenceMode guard;
      c = torch::ones({1, 2, 3});
    }

    ASSERT_THROWS_WITH(c.add_(s), // go through kernels: VariableType(ERROR!), InplaceOrView, CPU
      "Inference tensor cannot participate in autograd")

    ASSERT_THROWS_WITH(torch::add_out(c, s, s), // go through kernels: VariableType(ERROR!), InplaceOrView, CPU
      "Inference tensor cannot participate in autograd")
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorViewOp) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      InferenceMode guard;
      c = torch::ones({1, 2, 3});
    }

    // view_as is a composite op which calls view() with only one tensor argument.
    // So there isn't a mixed inference tensor and normal tensor inputs for view ops.
    ASSERT_THROWS_WITH(c.view_as(s), // go through kernels: InplaceOrView(ERROR!), CPU
      "inplace/view ops on inference tensor outside InferenceMode")

    // This is fine since it's equivalent as s.view(c.sizes()) which
    // isn't a mixed input scenario.
    s.view_as(c); // go through kernels: VariableType, InplaceOrView, CPU
  }
}

TEST(InferenceModeTest, TestHandleDirectViewOnRebase) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out;
    {
      InferenceMode guard;
      view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
    }
    if (requires_grad) {
      ASSERT_THROWS_WITH(inplace_op(view_out),
        "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out);
    }
  }
}

TEST(InferenceModeTest, TestHandleInDirectViewOnRebase) {
  for (bool requires_grad: {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out;
    {
      InferenceMode guard;
      view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
    }
    inplace_op(a);
    if (requires_grad) {
      ASSERT_THROWS_WITH(view_out.grad_fn(),
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
  ASSERT_THROWS_WITH(b.add_(1),
    "A view was created in inference mode and is being modified inplace");
  {
    AutoGradMode mode(false);
    c = b.view_as(b);
  }
  ASSERT_THROWS_WITH(c.add_(1),
    "A view was created in inference mode and is being modified inplace");
}
