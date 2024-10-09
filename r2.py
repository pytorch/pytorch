# type: ignore
import torch
import torch.utils.cpp_extension

"""
Python compiled autograd POC.

This is a re-implementation of compiled autograd done mostly in Python.
The main benefit of this is that it specializes on less than the existing
compiiled autograd implementation (thereby having better support for Tensor subclasses)
and we are able to graph break on unsupported C++ autograd nodes.

Scroll to the bottom of this file to see "user code" (some test cases)
"""

def compiled_autograd(tensor, *, compiler=lambda f: f):
    """Executes the equivalent of tensor.backward(), but in a compiled way.

    There are two phases:
    1. First, we parse the autograd graph and build a function that essentially
    "runs the backward" in Python.
    2. Next, we run the compiler (usually torch.compile) over said function.

    The function can run user-defined hooks (Dynamo is able to inline into them),
    and is also able to graph-break on unsupported things (like unsupported
    C++ autograd nodes).
    """
    nodes = get_execution_order(tensor)

    def lift_saved_values():
        result = []
        for node in nodes:
            for attr in dir(node):
                if attr.startswith("_raw"):
                    non_raw_attr = attr[4:]
                    result.append(getattr(node, non_raw_attr))
            if is_accumulate_grad(node):
                result.append(node.variable)
        return result

    saved_values = lift_saved_values()

    # NB: the real compiled autograd has its own cache on the
    # (backward graph, saved_values) here.

    # Phase 1: construct a function that executes the backward in Python
    # This function is stateless and accepts the saved_values as inputs.
    func = construct_backward_function(nodes)
    # Phase 2: run torch.compile on the function
    compiler(func)(torch.tensor(1.), saved_values)


def get_execution_order(tensor):
    """Returns a list of autograd nodes, in the order that the c++ autograd
    engine would have evaluated them in.
    """
    execution_order = None

    def hook(_):
        nonlocal execution_order
        execution_order = torch._C._current_graph_task_execution_order()
        # Super hacky, we can make this API much nicer.
        raise StopIteration

    handle = tensor.register_hook(hook)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            tensor.backward()
    except StopIteration:
        pass
    finally:
        handle.remove()
    return execution_order


def create_apply_node_with_saved(node):
    """Given a node, return a function that accepts (grad_outputs, saved_values)
    and produces the grad_inputs.
    """
    if is_accumulate_grad(node):
        # AccumulateGrad handled elsewhere
        return None

    if node.is_traceable():
        dec = torch._dynamo.allow_in_graph
    else:
        dec = torch._dynamo.disable

    @dec
    def apply_node_with_saved(grads, *saved):
        new_saved_values = [torch._C._autograd.SavedTensor(x) for x in saved]
        assert num_saved(node) == len(saved)
        swap_saved_values(node, new_saved_values)
        try:
            result = node(*grads)
            return result
        finally:
            swap_saved_values(node, new_saved_values)
    return apply_node_with_saved


def is_accumulate_grad(node):
    return node.__class__.__name__ == "AccumulateGrad"


def visit_node(node, visit_saved, visit_param=None):
    for attr in dir(node):
        if attr.startswith("_raw"):
            old_saved_value = getattr(node, attr)
            visit_saved(old_saved_value)
    if is_accumulate_grad(node):
        if visit_param is not None:
            visit_param(node.variable)


def swap_saved_values(node, saved):
    counter = 0

    def visit(old_saved_value):
        nonlocal counter
        if counter >= len(saved):
            breakpoint()
        new_saved_value = saved[counter]
        torch._C._swap_saved(old_saved_value, new_saved_value)
        counter += 1

    visit_node(node, visit)


def num_saved(node):
    counter = 0

    def visit_saved(val):
        nonlocal counter
        counter += 1

    def visit_param(param):
        nonlocal counter
        counter += 1

    visit_node(node, visit_saved, visit_param)
    return counter


def construct_backward_function(nodes):
    # Do some preprocessing here
    apply_with_saved = {idx: create_apply_node_with_saved(node) for idx, node in enumerate(nodes)}
    num_saved_values = {idx: num_saved(node) for idx, node in enumerate(nodes)}
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    next_functions = {
        idx: tuple((node_to_idx[next_node], next_idx) for next_node, next_idx in node.next_functions)
        for idx, node in enumerate(nodes)
    }
    is_accumulate_grad_node = {idx: is_accumulate_grad(node) for idx, node in enumerate(nodes)}
    pre_hooks = {idx: node.pre_hooks() for idx, node in enumerate(nodes)}

    def execute_autograd(input_buffer, saved_values):
        """Captures the autograd nodes to be executed.
        This function is essentially a re-implementation of the autograd engine
        in Python.
        """
        all_input_buffers = {0: [input_buffer]}
        saved_values_begin = 0
        for idx, node in enumerate(nodes):
            input_buffers = all_input_buffers[idx]

            saved_values_end = saved_values_begin + num_saved_values[idx]

            # We support prehooks.
            if len(pre_hooks[idx]) > 0:
                assert len(pre_hooks[idx][0].values()) == 1
                pre_hook = list(pre_hooks[idx][0].values())[0]
                input_buffers = pre_hook(input_buffers)

            if is_accumulate_grad_node[idx]:
                # Rewrite AccumulateGrad nodes into a special op.
                assert len(input_buffers) == 1
                param = saved_values[saved_values_begin]
                param_grad = input_buffers[0].expand_as(param)
                grad_inputs = [torch.ops.inductor.accumulate_grad_.default(param, param_grad)]
            else:
                grad_inputs = apply_with_saved[idx](input_buffers, *saved_values[saved_values_begin:saved_values_end])
            saved_values_begin = saved_values_end

            # Handle gradient accumulation and passing to the next node
            for grad_input, (next_node_idx, idx) in zip(grad_inputs, next_functions[idx]):
                if grad_input is None:
                    continue
                if next_node_idx not in all_input_buffers:
                    all_input_buffers[next_node_idx] = []
                if idx == len(all_input_buffers[next_node_idx]):
                    all_input_buffers[next_node_idx].append(grad_input)
                elif idx < len(all_input_buffers[next_node_idx]):
                    all_input_buffers[next_node][idx] += grad_input
                else:
                    raise AssertionError("help")

    return execute_autograd


"""
===========================================================

                     BEGIN USER CODE

===========================================================
"""

# ===========================================================
# Basic test

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = torch.randn(3, requires_grad=True)
value = 2.

out = torch.addcmul(a, b, c, value=value)

loss = out.sum()
compiled_autograd(loss, compiler=torch.compile(backend="aot_eager", fullgraph=True))

assert torch.allclose(a.grad, torch.ones_like(a))
assert torch.allclose(b.grad, c * value)
assert torch.allclose(c.grad, b * value)


# ===========================================================
# Hooks with side effects work.

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
c = torch.randn(3, requires_grad=True)
value = 2.

out = torch.addcmul(a, b, c, value=value)
stuff = []

def hook(grads):
    stuff.append(grads[0])
    return grads

out.grad_fn.register_prehook(hook)

loss = out.sum()
compiled_autograd(loss, compiler=torch.compile(backend="eager", fullgraph=True))
assert len(stuff) == 1
assert torch.allclose(stuff[0], torch.ones_like(stuff[0]))


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

TORCH_LIBRARY(test_non_traceable_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
"""

module = torch.utils.cpp_extension.load_inline(
    name="test_non_traceable_autograd_cpp_node",
    cpp_sources=cpp_source,
    functions="custom_op_backed_by_autograd_fn",
    verbose=True,
)

x = torch.ones(10, 10, requires_grad=True)
out = torch.ops.test_non_traceable_autograd_cpp_node.custom_op_backed_by_autograd_fn(
    x
)
loss = out.sum()
compiled_autograd(loss, compiler=torch.compile(backend="eager"))
expected = torch.ones_like(x) * 3.14
assert torch.allclose(x.grad, expected)
