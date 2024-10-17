# type: ignore
import torch
import torch.utils.cpp_extension

def compiler_fn(gm):
    # return gm
    return torch.compile(gm, backend="eager", fullgraph=False)

# ===========================================================
# Basic test with a hook that has side effects


# Test case 1: a hook
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 2
z = y.sum()

im_grad = []

def hook(grad):
    im_grad.append(grad)
    return 2 * grad

y.register_hook(hook)

with torch._dynamo.compiled_autograd.enable(compiler_fn):
    z.backward()

assert torch.allclose(x.grad, 4 * x)
assert torch.allclose(im_grad[0], torch.ones_like(y))

# ===========================================================
# Unsupported C++ autograd node should graph break.
# This is better than the current compiled autograd behavior of "error out"
# and brings us a step closer to having "compiled autograd on by default".
# In theory we can also add a config that automatically treats
# it as an opaque callable, but such a config is unsound.

cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = false;
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    // not traceable
    *grad_output[0].data_ptr<float>() = 3.14;
    return grad_output;
  }
};
torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}
TORCH_LIBRARY_FRAGMENT(mylib, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""

module = torch.utils.cpp_extension.load_inline(
    name="mylib",
    cpp_sources=cpp_source,
    functions="custom_op_backed_by_autograd_fn",
    verbose=True,
)

x = torch.ones(2, 2, requires_grad=True)
out = torch.ops.mylib.custom_op_backed_by_autograd_fn(
    x
)
loss = out.sum()
with torch._dynamo.compiled_autograd.enable(compiler_fn):
    loss.backward()

expected = torch.ones_like(x) * 3.14
assert torch.allclose(x.grad, expected)

# ===========================================================
# Tests that we don't bake in "guessed" metadata.
# This test case would have erroed out in the previous
# compiled autograd.

import torch
import torch.utils.cpp_extension
cpp_source2 = """
struct CustomOpAutogradFunction2 : public torch::autograd::Function<CustomOpAutogradFunction2> {
  static constexpr bool is_traceable = true;
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    if (grad_output[0].is_contiguous()) {
        return {2 * grad_output[0]};
    } else {
        return {3 * grad_output[0]};
    }
  }
};
torch::Tensor custom_op_backed_by_autograd_fn2(torch::Tensor x) {
  return CustomOpAutogradFunction2::apply(x);
}
TORCH_LIBRARY_FRAGMENT(mylib, m) {
    m.def("custom_op_backed_by_autograd_fn2", custom_op_backed_by_autograd_fn2);
}
"""

module = torch.utils.cpp_extension.load_inline(
    name="mylib",
    cpp_sources=cpp_source2,
    functions="custom_op_backed_by_autograd_fn2",
    verbose=True,
)


x = torch.tensor([[1., 2., 3.], [4, 5, 6]], requires_grad=True)
y = torch.ops.mylib.custom_op_backed_by_autograd_fn2(x)
z = y.clone()
w = z.sum()

def hook(grad):
    # return a contiguous grad.
    # The previous compiled autograd would have "guessed" that
    # the tensor is not contiguous.
    assert not grad.is_contiguous()
    return grad.contiguous()

z.register_hook(hook)

with torch._dynamo.compiled_autograd.enable(lambda x: x):
    w.backward()

assert torch.allclose(x.grad, 2 * torch.ones_like(x))
