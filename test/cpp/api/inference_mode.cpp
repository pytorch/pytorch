#include <torch/script.h>
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

using namespace torch::autograd;

torch::Tensor functional_op(torch::Tensor& x) {
  return x + x;
}

void inplace_op(torch::Tensor& x) {
  x.add_(1);
}

torch::Tensor view_op(torch::Tensor& x) {
  return x.view({2, 3});
}

/*
  Only the following combos of Autograd & InplaceOrView keys on tensors are valid:
    - Autograd=true, InplaceOrView=true (normal tensor)
    - Autograd=false, InplaceOrView=false (inference tensor)
  Tensors created in InferenceMode are mostly inference tensors. There're a few exceptions
  which listed below, even though these tensors are created in InferenceMode, they're
  normal tensors:
    - tensors which are view of normal tensors.
    - output of inplace ops on normal tensors(which is the input itself).
*/
bool is_inference_tensor(torch::Tensor& x) {
  c10::DispatchKeySet ks = x.key_set();
  bool has_Autograd = ks.has(c10::DispatchKey::AutogradCPU);
  bool has_InplaceOrView = ks.has(c10::DispatchKey::InplaceOrView);
  // They must be either both true or false.
  //ASSERT_TRUE(has_Autograd == has_InplaceOrView);
  return !has_Autograd && !has_InplaceOrView;
}

void assert_tensor_creation_meta(torch::Tensor& x, torch::autograd::CreationMeta creation_meta) {
  ASSERT_EQ(static_cast<torch::autograd::DifferentiableViewMeta*>(x.unsafeGetTensorImpl()->autograd_meta())->get_creation_meta(), creation_meta);
}

void assert_TLS_states(bool inference_mode) {
  ASSERT_EQ(InferenceMode::is_enabled(), inference_mode);
  ASSERT_EQ(c10::impl::is_all_dispatch_keyset_excluded(c10::autograd_dispatch_keyset), inference_mode);
  ASSERT_EQ(c10::impl::tls_is_dispatch_key_included(c10::DispatchKey::InplaceOrView), !inference_mode);
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
    // New tensor created in InferenceMode (inference tensor) doesn't have Autograd & InplaceOrView keys.
    // Even if you ask it to requires_grad (which doesn't make sense in InferenceMode),
    // it'll be silently ignored. This is not an error to make adding InferenceMode
    // guard to existing code a smooth user experience.
    torch::Tensor c = torch::ones({1, 2, 3});
    ASSERT_TRUE(is_inference_tensor(c));
    ASSERT_FALSE(c.requires_grad());

    torch::Tensor tmp = torch::ones({1, 2, 3}).set_requires_grad(true);
    ASSERT_TRUE(tmp.requires_grad()); // requires_grad is silently ignored when it's an inference tensor.
    ASSERT_TRUE(is_inference_tensor(tmp));

    tmp = torch::ones({1, 2, 3}).set_requires_grad(false);
    ASSERT_FALSE(tmp.requires_grad());
    ASSERT_TRUE(is_inference_tensor(tmp));
  }
}

TEST(InferenceModeTest, TestExistingAutogradSession) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s + 2;

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

TEST(InferenceModeTest, TestInferenceTensorInInferenceMode) {
  c10::InferenceMode guard;
  torch::Tensor c = torch::ones({1, 2, 3});

  torch::Tensor func_out = functional_op(c);  // go through kernels: CPU
  ASSERT_TRUE(is_inference_tensor(func_out));

  inplace_op(c);  // go through kernels: CPU
  ASSERT_TRUE(is_inference_tensor(c));

  torch::Tensor view_out = view_op(c);  // go through kernels: CPU
  ASSERT_TRUE(is_inference_tensor(view_out));
}

TEST(InferenceModeTest, TestInferenceTensorInNormalMode) {
  torch::Tensor inference_tensor;
  {
    InferenceMode guard;
    inference_tensor = torch::ones({1, 2, 3});
  }
  // Functional ops on inference tensors might run slower outside InferenceMode than inside.
  // But it's fine that we don't care about perf of this case that much.
  //
  // An alternative behavior we perfer but didn't implement is throwing an error by forcing
  // this op go through VariableType kernel and hit the assert_no_inference_tensor check.
  // But to do that we'll have to add c10::autograd_dispatch_keyset to the globally enabled set,
  // but doing that might accidentally call autograd kernel from a backend that doesn't match tensor input.
  // Thus we allow functional ops run without throwing an error.
  //
  torch::Tensor tmp = functional_op(inference_tensor); // go through kernels: InplaceOrView(fallthrough), CPU
  ASSERT_FALSE(is_inference_tensor(tmp));
  ASSERT_FALSE(tmp.requires_grad());

  // If you hit InplaceOrView kernel without Autograd keys being excluded, throw an error.
  // If we have new intermediate tensor created in backend kernels, ops on it will dispatch to VariableType kernel
  // unexpectedly.
  // Erroring out in InplaceOrView kernel is much easier and less error prone.
  ASSERT_THROWS_WITH(inplace_op(inference_tensor), // go through kernels: InplaceOrView(ERROR!), CPU
    "inplace/view ops on inference tensor outside InferenceMode")
  ASSERT_THROWS_WITH(view_op(inference_tensor), // go through kernels: InplaceOrView(ERROR!), CPU
    "inplace/view ops on inference tensor outside InferenceMode")
}

// TODO: Currently requires_grad=false tensor behave exactly the same as requires_grad=true tensor
//       inside dispatcher.
//       In the ideal future requires_grad=false tensor don't go through VariableType kernel
//       and we should add separate tests for require_grad=false tensor.
TEST(InferenceModeTest, TestRequiresGradTrueTensorFunctionalOp) {
  torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor func_out;

  {
    c10::InferenceMode guard;
    // Functional op on normal tensor in InferenceMode produces
    // inference tensor as output.
    func_out = functional_op(a);  // go through kernels: InplaceOrView, CPU
    ASSERT_TRUE(is_inference_tensor(func_out));
    ASSERT_FALSE(func_out.requires_grad());
  }

  // func_out should behave exactly the same as inference tensor.
  torch::Tensor tmp = functional_op(func_out); // go through kernels: InplaceOrView(fallthrough), CPU
  ASSERT_FALSE(is_inference_tensor(tmp));
  ASSERT_FALSE(tmp.requires_grad());

  ASSERT_THROWS_WITH(inplace_op(func_out), // go through kernels: InplaceOrView(ERROR!), CPU
    "inplace/view ops on inference tensor outside InferenceMode")
  ASSERT_THROWS_WITH(view_op(func_out), // go through kernels: InplaceOrView(ERROR!), CPU
    "inplace/view ops on inference tensor outside InferenceMode")
}

TEST(InferenceModeTest, TestRequiresGradTrueTensorInplaceOp) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s + 2;

  {
    c10::InferenceMode guard;

    // Inplace ops on normal tensor produce normal tensors as output.
    // In other words they have Autograd=true, InplaceOrView=true.
    // It's straightforward since output is the same as input.
    inplace_op(a);  // go through kernels: InplaceOrView, CPU
    ASSERT_FALSE(is_inference_tensor(a));
    ASSERT_TRUE(a.requires_grad());

    // inplace -> inplace
    inplace_op(a);  // go through kernels: InplaceOrView, CPU
    ASSERT_FALSE(is_inference_tensor(a));
    ASSERT_TRUE(a.requires_grad());

    // inplace -> inplace -> view
    torch::Tensor view_out = view_op(a);
    ASSERT_FALSE(is_inference_tensor(view_out));
    ASSERT_TRUE(view_out.requires_grad());
  }

  torch::Tensor tmp = functional_op(a);  // go through kernels: VariableType, InplaceOrView(fallthrough), CPU
  ASSERT_FALSE(is_inference_tensor(tmp));
  ASSERT_TRUE(tmp.requires_grad());

  inplace_op(a); // go through kernels: VariableType, InplaceOrView, CPU
  ASSERT_FALSE(is_inference_tensor(a));
  ASSERT_TRUE(a.requires_grad());

  tmp = view_op(a);  // go through kernels: VariableType, InplaceOrView, CPU
  ASSERT_FALSE(is_inference_tensor(tmp));
  ASSERT_TRUE(tmp.requires_grad());
}

TEST(InferenceModeTest, TestRequiresGradTrueTensorViewOp) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s + 2;
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
    //   In addition, these view output tensors although are normal in the sense
    //   that it has both Autograd and InplaceOrView keys, they're still special
    //   since they'll have CreationMeta::NO_VARIABLE_TYPE_VIEW since they don't
    //   have proper setup for backward node (skipping VariableType kernel).
    //   In other words, they'll be specially handled if participating autograd
    //   outside InferenceMode.

    view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
    ASSERT_FALSE(is_inference_tensor(view_out));
    assert_tensor_creation_meta(view_out, torch::autograd::CreationMeta::NO_VARIABLE_TYPE_VIEW);
    ASSERT_TRUE(view_out.requires_grad());
    ASSERT_FALSE(view_out.is_leaf());

    // view -> view
    tmp = view_op(view_out);  // go through kernels: InplaceOrView, CPU
    ASSERT_FALSE(is_inference_tensor(tmp));
    assert_tensor_creation_meta(tmp, torch::autograd::CreationMeta::NO_VARIABLE_TYPE_VIEW);
    ASSERT_TRUE(tmp.requires_grad());
    ASSERT_FALSE(tmp.is_leaf());

    // view -> view -> inplace
    inplace_op(tmp);  // kernels: InplaceOrView, CPU
    assert_tensor_creation_meta(tmp, torch::autograd::CreationMeta::NO_VARIABLE_TYPE_VIEW);
    ASSERT_FALSE(is_inference_tensor(tmp));
    ASSERT_TRUE(tmp.requires_grad());
  }

  ASSERT_THROWS_WITH(functional_op(view_out), // go through kernels: VariableType, InplaceOrView(fallthrough), CPU
    "A view created in InferenceMode without proper grad_fn setup")

  ASSERT_THROWS_WITH(inplace_op(view_out),  // go through kernels: VariableType, InplaceOrView, CPU
    "A view created in InferenceMode without proper grad_fn setup")

  ASSERT_THROWS_WITH(view_op(view_out), // go through kernels: VariableType, InferenceMode, CPU
    "A view created in InferenceMode without proper grad_fn setup")
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensor) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor c;
  {
    InferenceMode guard;
    c = torch::ones({1, 2, 3});

    torch::Tensor d = s + c; // go through kernels: InplaceOrView (fallthrough), CPU
    ASSERT_TRUE(is_inference_tensor(d));
    ASSERT_FALSE(d.requires_grad());
  }

  ASSERT_THROWS_WITH(c.add(s), // go through kernels: VariableType(ERROR!), InplaceOrView(fallthrough), CPU
    "inference tensor cannot participate in autograd")

  ASSERT_THROWS_WITH(c.add_(s), // go through kernels: VariableType(ERROR!), InplaceOrView, CPU
    "inference tensor cannot participate in autograd")

  // view_as is a composite op which calls view() with only one tensor argument.
  // So there isn't a mixed inference tensor and normal tensor inputs for view ops.
  ASSERT_THROWS_WITH(c.view_as(s), // go through kernels: InplaceOrView(ERROR!), CPU
    "inplace/view ops on inference tensor outside InferenceMode")

  // This is fine since it's equivalent as s.view(c.sizes()) which
  // isn't a mixed input scenario.
  s.view_as(c); // go through kernels: VariableType, InplaceOrView, CPU

  ASSERT_THROWS_WITH(torch::add_out(c, s, s), // go through kernels: VariableType(ERROR!), InplaceOrView, CPU
    "inference tensor cannot participate in autograd")
}

TEST(InferenceModeTest, TestHandleViewOnRebase) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s + 2;
  torch::Tensor view_out;
  {
    InferenceMode guard;
    view_out = view_op(a);  // go through kernels: InplaceOrView, CPU
  }
  inplace_op(a);
  ASSERT_THROWS_WITH(view_out.grad_fn(),
    "A view of normal tensor was created in InferenceMode")

}
