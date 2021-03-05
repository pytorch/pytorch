#include <torch/script.h>
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

torch::Tensor functional_op(torch::Tensor& x) {
  return x + x;
}

torch::Tensor& inplace_op(torch::Tensor& x) {
  return x.add_(1);
}

torch::Tensor view_op(torch::Tensor& x) {
  return x.view({2, 3});
}

void assert_keys(torch::Tensor& x, bool autograd, bool inplace) {
  c10::DispatchKeySet ks = x.key_set();
  bool has_autograd = ks.has(c10::DispatchKey::AutogradCPU);
  bool has_inplace = ks.has(c10::DispatchKey::InplaceOrView);
  assert(has_autograd == autograd);
  assert(has_inplace == inplace);
}

void assert_special_creation_meta(torch::Tensor& x) {
  assert(static_cast<torch::autograd::DifferentiableViewMeta*>(x.unsafeGetTensorImpl()->autograd_meta())->get_creation_meta() == torch::autograd::CreationMeta::NO_VARIABLE_TYPE);
}

/*
Only the following combos of Autograd & InplaceOrView keys on tensors are valid:
- Autograd=true, InplaceOrView=true
- Autograd=false, InplaceOrView=false
*/
TEST(InferenceModeTest, TestBasic) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s + 2;
  torch::Tensor b = torch::ones({1, 2, 3}).set_requires_grad(false);

  // Save `a` in an existing autograd session
  torch::Tensor out = a * a;

  torch::Tensor requires_grad_true_func_out, requires_grad_true_inplace_out, requires_grad_true_view_out;
  torch::Tensor requires_grad_false_func_out, requires_grad_false_inplace_out, requires_grad_false_view_out;
  torch::Tensor inference_tensor_func_out, inference_tensor_inplace_out, inference_tensor_view_out;

  {
    c10::AutoInferenceMode guard(true);
    // New tensor created in AutoInferenceMode doesn't have Autograd & InplaceOrView keys.
    // Even if you ask it to requires_grad (which doesn't make sense in AutoInferenceMode), it'll be silently ignored.
    // But this is not an error to make adding AutoInferenceMode guard to existing code a smooth user experience.
    torch::Tensor c = torch::ones({1, 2, 3});
    assert_keys(c, false, false);
    assert(c.requires_grad() == false);
    torch::Tensor tmp = torch::ones({1, 2, 3}).set_requires_grad(true);
    assert(tmp.requires_grad() == false);  // Question: Do we want this to show true/false?
    assert_keys(tmp, false, false);
    tmp = torch::ones({1, 2, 3}).set_requires_grad(false);
    assert_keys(tmp, false, false);

    // 1. copy constructor preserves original keyset.
    tmp = a;
    assert_keys(tmp, true, true);
    tmp = b;
    assert_keys(tmp, true, true);
    tmp = c;
    assert_keys(tmp, false, false);

    // 2. Entering AutoInferenceMode
    //   2.1. requires_grad=true tensor in AutoInferenceMode
    requires_grad_true_func_out = functional_op(a);  // kernels: InplaceOrView, CPU
    assert_keys(requires_grad_true_func_out, false, false);

    // All Autograd=true, InplaceOrView=true created through view op in AutoInferenceMode
    // has CreationMeta::NO_VARIABLE_TYPE, which should be banned in autograd later
    // after exiting AutoInferenceMode.
    // Note InplaceOrView op returns the same Tensor as input, so it doesn't need special handling.
    requires_grad_true_inplace_out = inplace_op(a);  // kernels: InplaceOrView, CPU
    // requires_grad_true_inplace_out has the same TensorImpl as `a` so both keys are present.
    assert_keys(requires_grad_true_inplace_out, true, true);

    // Chained inplace ops
    tmp = inplace_op(requires_grad_true_inplace_out);
    assert_keys(requires_grad_true_inplace_out, true, true);

    requires_grad_true_view_out = view_op(a);  // kernels: InplaceOrView, CPU
    // requires_grad_true_view_out has both keys due to the way we create view Tensors in alias_with_sizes_and_strides
    //     auto impl = c10::make_intrusive<TensorImpl>(
    //     Storage(self.storage()), self.key_set(), self.dtype());
    assert_keys(requires_grad_true_view_out, true, true);
    assert_special_creation_meta(requires_grad_true_view_out);

    // Chained view ops
    tmp = view_op(requires_grad_true_view_out);  // kernels: InplaceOrView, CPU
    assert_keys(tmp, true, true);
    assert_special_creation_meta(requires_grad_true_view_out);


    // view -> view -> inplace
    tmp = inplace_op(tmp);  // kernels: InplaceOrView, CPU
    assert_keys(tmp, true, true);

    //  2.2 requires_grad=false tensor in AutoInferenceMode
    //      same as requires_grad=true case
    requires_grad_false_func_out = functional_op(a);  // kernels: InplaceOrView, CPU
    assert_keys(requires_grad_false_func_out, false, false);
    requires_grad_false_inplace_out = inplace_op(a);  // kernels: InplaceOrView, CPU
    assert_keys(requires_grad_false_inplace_out, true, true);
    requires_grad_false_view_out = view_op(a);  // kernels: InplaceOrView, CPU
    assert_keys(requires_grad_false_view_out, true, true);
    assert(requires_grad_false_view_out.is_leaf() == false);


    // 3. InferenceTensor in AutoInferenceMode
    inference_tensor_func_out = functional_op(c);  // kernels: CPU
    assert_keys(inference_tensor_func_out, false, false);
    inference_tensor_inplace_out = inplace_op(c);  // kernels: CPU
    assert_keys(inference_tensor_inplace_out, false, false);
    inference_tensor_view_out = view_op(c);  // kernels: CPU
    assert_keys(inference_tensor_view_out, false, false);

  }

  // 4. Exiting InferneceOnlyMode
  // Copying inference Tensor preserves key_set
  torch::Tensor tmp = inference_tensor_func_out;
  assert_keys(tmp, false, false);
  assert(inference_tensor_func_out.requires_grad() == false);

  // Proposed idea: if you hit InplaceOrView kernel without VariableType being excluded, throw an error.

  //   4.1 Tensor with Autograd=false && InplaceOrView=false behavior outside AutoInferenceMode
  //       e.g. requires_grad_true_func_out/requires_grad_false_func_out


  // We cannot add c10::autograd_dispatch_keyset to gloablly enabled, because that will
  // accidentally call autograd kernel from a backend that might not match tensor input.
  // This will run slower than running in AutoInferenceMode but it's fine that we don't
  // care about perf of this case that much.
  tmp = functional_op(requires_grad_true_func_out); // kernels: InplaceOrView(fallthrough), CPU
  assert_keys(tmp, true, true);


  // Note: Why do we choose to error out instead of properly handling it?
  //
  // If we have new intermediate tensor created in backend kernels, ops on it will dispatch to VariableType kernel
  // unexpectedly.
  // Erroring out is much easier and less error prone.
  //
  // It should go through InplaceOrView and CPU and never goes to VariableType,
  // but in fact when dispatching from add(Tensor, Scalar) to add(Tensor, Tensor)
  // it goes back to dispatcher and reach VariableType, thus the error! (which doesn't make sense)
  ASSERT_THROWS_WITH(inplace_op(requires_grad_true_func_out), // kernels: InplaceOrView, CPU
    "Cannot handle Inference Tensor outside InferenceMode")
  ASSERT_THROWS_WITH(view_op(requires_grad_true_func_out), // kernels: InplaceOrView, CPU
  "Cannot handle Inference Tensor outside InferenceMode")

  // 4.2 Tensor with Autograd=true && InplaceOrView=true but with CreationMeta::NoVariableType created inside AutoInferenceMode
  //     behavior outside AutoInferenceMode.
  //       e.g. requires_grad_false_view_out

  tmp = functional_op(requires_grad_false_view_out);

  // TODO: update error message to remove no_grad
  ASSERT_THROWS_WITH(inplace_op(requires_grad_false_view_out),  // kernels: InplaceOrView, CPU
    "This view is created in InferenceMode without proper autograd setup")

  // TODO: maybe we should just throw here?
  tmp = view_op(requires_grad_false_view_out);
  assert_special_creation_meta(requires_grad_false_view_out);

  // 5. perform backward on `a` should trigger error since `a`'s version has been bumped.
  ASSERT_THROWS_WITH(out.backward(torch::ones_like(out)),
    "one of the variables needed for gradient computation has been modified by an inplace operation")


  // 6. Mix normal tensor and inference tensor as inputs
  ASSERT_THROWS_WITH(requires_grad_true_func_out.add(a),
    "inference tensor cannot participate in autograd")

}

