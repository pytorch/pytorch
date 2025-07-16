(control_flow_operators)=

# Control Flow Operators

Control flow operators in PyTorch enable dynamic, data-dependent execution paths within your models. Unlike traditional static computation graphs, these operators allow your model to make decisions and execute different operations based on the values or shapes of tensors at runtime. This capability is essential for exporting or efficient compiling complex algorithms that require conditional logic or iterative processing.

| Control Flow Operator | torch.compile | export | autograd |
|-----------------------|---------------|--------|----------|
| cond                  | Y             | Y      | Y        |
| while_loop            | Y             | Y      | WIP      |
| scan                  | Y             | Y      | Y        |
| associative_scan      | Y             | Y      | WIP      |
| map                   | Y             | Y      | Y        |


```{warning}
Control flow operators are prototype features in PyTorch. They have limited support for input and output types, and some operators don't support training currently. Please look forward to more stable implementations in future versions of PyTorch.
Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

(cond)=

## torch.cond

`torch.cond` is a structured control flow operator that enables if-else like control flow. It can logically be seen as implemented as follows:

```python
def cond(
    pred: Union[bool, torch.Tensor],
    true_fn: Callable,
    false_fn: Callable,
    operands: Tuple[torch.Tensor]
):
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)
```

Its unique power lies in its ability to express **data-dependent control flow**: it lowers to a conditional
operator (`torch.ops.higher_order.cond`), which preserves predicate, true function and false functions.
This unlocks great flexibility in writing and deploying models that change model architecture based on
the **value** or **shape** of inputs or intermediate outputs of tensor operations.

### Examples

Below is an example that uses cond to branch based on input shape:

```python
import torch

def true_fn(x: torch.Tensor):
    return x.cos() + x.sin()

def false_fn(x: torch.Tensor):
    return x.sin()

class DynamicShapeCondPredicate(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_fn(x: torch.Tensor):
            return x.cos()

        def false_fn(x: torch.Tensor):
            return x.sin()

        return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

dyn_shape_mod = DynamicShapeCondPredicate()
```

We can eagerly run the model and expect the results vary based on input shape:

```python
inp = torch.randn(3)
inp2 = torch.randn(5)
assert torch.equal(dyn_shape_mod(inp), false_fn(inp))
assert torch.equal(dyn_shape_mod(inp2), true_fn(inp2))
```

We can export the model for further transformations and deployment:

```python
inp = torch.randn(4, 3)
dim_batch = torch.export.Dim("batch", min=2)
ep = torch.export.export(DynamicShapeCondPredicate(), (inp,), {}, dynamic_shapes={"x": {0: dim_batch}})
print(ep)
```

This gives us an exported program as shown below:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[s0, 3]):
        sym_size: Sym(s0) = torch.ops.aten.sym_size.int(arg0_1, 0)
        gt: Sym(s0 > 4) = sym_size > 4;  sym_size = None
        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        conditional: f32[s0, 3] = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
        return (conditional,)

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1)
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            add: f32[s0, 3] = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
            return add

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin
```

Notice that `torch.cond` is lowered to `torch.ops.higher_order.cond`, its predicate becomes a Symbolic expression over the shape of input,
and branch functions becomes two sub-graph attributes of the top level graph module.

Here is another example that showcases how to express a data-dependent control flow:

```python
class DataDependentCondPredicate(torch.nn.Module):
    """
    A basic usage of cond based on data dependent predicate.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(x.sum() > 4.0, true_fn, false_fn, (x,))
```

The exported program we get after export:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[s0, 3]):
        sum_1: f32[] = torch.ops.aten.sum.default(arg0_1)
        gt: b8[] = torch.ops.aten.gt.Scalar(sum_1, 4.0);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        conditional: f32[s0, 3] = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
        return (conditional,)

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1)
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            add: f32[s0, 3] = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
            return add

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin
```

(while_loop)=

## torch._higher_order_ops.while_loop

`torch._higher_order_ops.while_loop` is a structured control flow operator that enables while-loop like control flow. It can logically be seen as implemented as follows:

```python
def while_loop(
    cond_fn: Callable,
    body_fn: Callable,
    carried_inputs: Tuple[Union[torch.Tensor, int, float, bool]]
):
    val = carried_inputs
    while cond_fn(*val):
        val = body_fn(*val)
    return val
```

It lowers to a higher-order operator (`torch.ops.higher_order.while_loop`), which preserves the condition function, body function, and carried inputs. It enables writing and deploying models that need to perform iterative computations based on the **value** or **shape** of inputs or intermediate outputs of tensor operations.

### Examples

Below is a simple example that uses while_loop to perform an iterative computation:

```python
import torch

class SimpleWhileLoop(torch.nn.Module):
    def forward(self, counter, a, b):
        def cond_fn(i, x, y):
            return i > 0

        def body_fn(i, x, y):
            return i - 1, x + y, y - x

        return torch._higher_order_ops.while_loop(cond_fn, body_fn, [counter, a, b])

counter = torch.tensor(5)  # Will run 5 iterations
a = torch.randn(3, 4)
b = torch.randn(3, 4)

model = SimpleWhileLoop()
final_counter, final_a, final_b = model(counter, a, b)
```

We can compare eager execution with `torch.compile`:

```python
# Eagerly run the model
model = SimpleWhileLoop()
eager_counter, eager_a, eager_b = model(counter, a, b)

# Compile and run the model
compiled_model = torch.compile(model)
compiled_counter, compiled_a, compiled_b = compiled_model(counter, a, b)

# Verify the results match
assert torch.equal(eager_counter, compiled_counter)
assert torch.equal(eager_a, compiled_a)
assert torch.equal(eager_b, compiled_b)
print("Eager and compiled results match!")
```

We can also export the model for further transformations and deployment:

```python
inp_counter = torch.tensor(5)
inp_a = torch.randn(4, 3)
inp_b = torch.randn(4, 3)
dim_batch = torch.export.Dim("batch", min=2)
ep = torch.export.export(
    SimpleWhileLoop(),
    (inp_counter, inp_a, inp_b),
    {},
    dynamic_shapes={"a": {0: dim_batch}, "b": {0: dim_batch}}
)
print(ep)
```

This gives us an exported program that preserves the while loop structure:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: i64[], arg1_1: f32[s0, 3], arg2_1: f32[s0, 3]):
        # ...
        conditional: (i64[], f32[s0, 3], f32[s0, 3]) = torch.ops.higher_order.while_loop(cond_graph_0, body_graph_0, [arg0_1, arg1_1, arg2_1]);
        # ...
        return conditional

    class cond_graph_0(torch.nn.Module):
        def forward(self, arg0_1: i64[], arg1_1: f32[s0, 3], arg2_1: f32[s0, 3]):
            gt: b8[] = torch.ops.aten.gt.Scalar(arg0_1, 0);
            return gt

    class body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: i64[], arg1_1: f32[s0, 3], arg2_1: f32[s0, 3]):
            add: i64[] = torch.ops.aten.add.Scalar(arg0_1, -1);
            add_1: f32[s0, 3] = torch.ops.aten.add.Tensor(arg1_1, arg2_1);
            sub: f32[s0, 3] = torch.ops.aten.sub.Tensor(arg2_1, arg1_1);
            return (add, add_1, sub)
```

Notice that `torch._higher_order_ops.while_loop` is lowered to `torch.ops.higher_order.while_loop`, with the condition function and body function becoming subgraphs of the main module.

Here's another example that demonstrates a data-dependent loop condition:

```python
class DataDependentWhileLoop(torch.nn.Module):
    def forward(self, x):
        def cond_fn(x):
            return x.sum() < 10

        def body_fn(x):
            return x + 1

        return torch._higher_order_ops.while_loop(cond_fn, body_fn, (x,))

# Create input
x = torch.ones(3, 4)  # Sum is 12, so loop won't execute

# Run the model
model = DataDependentWhileLoop()
result = model(x)
```

### Working with PyTree Structures

`while_loop` supports PyTree structures for carried inputs, allowing for more complex data structures:

```python
class PytreeWhileLoop(torch.nn.Module):
    def forward(self, it, pytree_input):
        def cond_fn(it, pytree_input):
            return it > 0

        def body_fn(it, pytree_input):
            x = pytree_input[0][0]
            y = pytree_input[1]["x"]
            z = pytree_input[1]["y"]
            new_x = y.sin()
            new_y = z.cos()
            new_z = x + 1
            return it - 1, ([new_x], {"x": new_y, "y": new_z})

        return torch._higher_order_ops.while_loop(cond_fn, body_fn, (it, pytree_input))

# Create inputs
it = torch.tensor(3)  # Will run 3 iterations
pytree_input = (
    [torch.randn(3, 4)],
    {"x": torch.randn(3, 4), "y": torch.randn(3, 4)}
)

# Run the model
model = PytreeWhileLoop()
final_it, final_pytree = model(it, pytree_input)
```

One requirement for using PyTree structures is that the pytree structure of carried inputs must match that of the output of the body function at corresponding position since the body function's output is used as the input to the next iteration of the loop.


(scan)=

## torch._higher_order_ops.scan

`torch._higher_order_ops.scan` performs an inclusive scan with a combine function. It can be used to implement cumulative operations over sequences of data, such as cumulative sums, products, or more complex operations. The semantic can be roughly summarized using following pseudocode:

```python
def scan(combine_fn, init, xs):
    carry = init
    ys = []
    for x in xs:
        carry, y = combind_fn(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)
```

It's useful for implementing algorithms that require sequential processing with state, such as recurrent neural networks, cumulative operations, or sequential transformations.

### Examples

Here's a simple example that uses scan to compute a cumulative sum:

```python
import torch

def add(carry, x):
    next_carry = y = carry + x
    return next_carry, y.clone()

init = torch.zeros(1)
xs = torch.arange(5)

# Returns (torch.tensor([10.]), torch.tensor([0., 1., 3., 6., 10.]))
last_carry, cumsum = torch._higher_order_ops.scan(add, init=init, xs=xs)
```

Here's a more complex example using scan with PyTree structures and neural network modules:

```python
class ScanLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, init, xs):
        def combine_fn(carry, x):
            prev_sz = x.size()
            x = self.linear(x.view(-1, x.size(-1)))
            x_view = x.view(*prev_sz)
            # Return the new carry and output
            return x_view, x_view.clone()

        return scan(combine_fn, init, xs, dim=0)

init = torch.randn(2, 4, 4)
xs = torch.randn(5, 2, 4, 4)  # 5 is the scan dimension

model = ScanLinearModel()
final_carry, outputs = model(init, xs)
```

We can compile the model using `torch.compile`:

```python
compiled_model = torch.compile(model)
final_carry_compiled, outputs_compiled = compiled_model(init, xs)

assert torch.allclose(final_carry, final_carry_compiled)
assert torch.allclose(outputs, outputs_compiled)

```

We can export the model for further transformations and deployment:

```python
init = torch.randn(2, 4, 4)
xs = torch.randn(5, 2, 4, 4)  # 5 is the scan dimension
dim_batch = torch.export.Dim("batch", min=1)
dim_seq = torch.export.Dim("seq", min=1)

ep = torch.export.export(
    ScanLinearModel(),
    (init, xs),
    {},
    dynamic_shapes={
        "init": {0: dim_batch},
        "xs": {0: dim_seq, 1: dim_batch}
    }
)
print(ep)
```

This gives us an exported program that preserves the scan structure:
```
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, p_linear_weight: "f32[4, 4]", p_linear_bias: "f32[4]", init: "f32[s80, 4, 4]", xs: "f32[s83, s80, 4, 4]"):
             # File: /home/yidi/local/pytorch/test.py:19 in forward, code: return scan(combine_fn, init, xs, dim=0)
            movedim: "f32[s83, s80, 4, 4]" = torch.ops.aten.movedim.int(xs, 0, 0);  xs = None

             # File: <eval_with_key>.23:10 in forward, code: scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_leaves_init_0_], [l_leaves_xs_0_], [l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_weight_, l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_bias_]);  scan_combine_fn_0 = l_leaves_init_0_ = l_leaves_xs_0_ = l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_weight_ = l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_bias_ = None
            scan_combine_graph_0 = self.scan_combine_graph_0
            scan = torch.ops.higher_order.scan(scan_combine_graph_0, [init], [movedim], (p_linear_weight, p_linear_bias));  scan_combine_graph_0 = init = movedim = p_linear_weight = p_linear_bias = None
            getitem_2: "f32[s80, 4, 4]" = scan[0]
            getitem_3: "f32[s83, s80, 4, 4]" = scan[1];  scan = None
            return (getitem_2, getitem_3)

        class scan_combine_graph_0(torch.nn.Module):
            def forward(self, arg0_1: "f32[s80, 4, 4]", arg1_1: "f32[s80, 4, 4]", arg2_1: "f32[4, 4]", arg3_1: "f32[4]"):
                 # File: <eval_with_key>.21:9 in forward, code: view = child_1.view(-1, 4);  child_1 = None
                view: "f32[4*s80, 4]" = torch.ops.aten.view.default(arg1_1, [-1, 4])

                 # File: <eval_with_key>.21:10 in forward, code: x = torch._C._nn.linear(view, l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_weight_, l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_bias_);  view = l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_weight_ = l_combine_fn_keywords_combine_fn_closure_0_cell_contents_modules_linear_parameters_bias_ = None
                linear: "f32[4*s80, 4]" = torch.ops.aten.linear.default(view, arg2_1, arg3_1);  view = arg2_1 = arg3_1 = None

                 # File: <eval_with_key>.21:11 in forward, code: x_view = x.view(getitem, 4, 4);  x = getitem = None
                sym_size_int: "Sym(s80)" = torch.ops.aten.sym_size.int(arg1_1, 0);  arg1_1 = None
                view_1: "f32[s80, 4, 4]" = torch.ops.aten.view.default(linear, [sym_size_int, 4, 4]);  linear = sym_size_int = None

                 # File: <eval_with_key>.21:12 in forward, code: child_2 = x_view.clone()
                clone: "f32[s80, 4, 4]" = torch.ops.aten.clone.default(view_1)
                return [view_1, clone]
```


### Working with PyTree Structures

`scan` supports PyTree structures for both the carry and the inputs/outputs:

```python
class PytreeScanModel(torch.nn.Module):
    def forward(self, init, xs):
        def combine_fn(carry, x):
            # Carry and x are both pytrees
            new_carry = {
                "param": carry["param"] @ x + carry["bias"],
                "bias": carry["bias"].sin(),
            }
            # Return new carry and output as pytrees
            return new_carry, {
                "result": new_carry["param"].clone(),
                "extra": {"bias": new_carry["bias"].clone()}
            }

        return scan(
            combine_fn,
            {"param": init["weight"], "bias": init["bias"]},
            xs,
            dim=0
        )

init = {
    "weight": torch.randn(4, 4),
    "bias": torch.randn(4)
}
xs = torch.randn(5, 4, 4)  # 5 is the scan dimension

model = PytreeScanModel()
final_carry, outputs = model(init, xs)
```

## torch._higher_order_ops.associative_scan

`torch._higher_order_ops.associative_scan` is a structured control flow operator that performs an inclusive scan with an associative combine function. Unlike the regular `scan` operator, `associative_scan` can achieve better performance by exploiting parallelism when the combine function is associative.

### Examples

Here's a simple example that uses associative_scan to compute a cumulative sum:

```python
import torch
from torch._higher_order_ops.associative_scan import associative_scan

def add(x: torch.Tensor, y: torch.Tensor):
    return x + y

x = torch.arange(5, device='cuda')  # [0, 1, 2, 3, 4]

# Compute cumulative sum using associative scan
result = associative_scan(add, x, dim=0)
# Result: tensor([0, 1, 3, 6, 10])

# This is equivalent to torch.cumsum(x, 0)
expected = torch.cumsum(x, 0)
assert torch.equal(result, expected)
```

We can compile the associative scan for better performance:

```python
def add(x: torch.Tensor, y: torch.Tensor):
    return x + y

# Compile the associative scan function
compiled_associative_scan = torch.compile(associative_scan, fullgraph=True)

x = torch.randn(1000, device='cuda')
result_compiled = compiled_associative_scan(add, x, dim=0, combine_mode="pointwise")
result_eager = associative_scan(add, x, dim=0, combine_mode="pointwise")

torch.testing.assert_close(result_compiled, result_eager)
```

### Working with PyTree Structures

`associative_scan` supports PyTree structures for both inputs and outputs:

```python
def combine_pytree(carry, x):
    return {
        "sum": carry["sum"] + x["value"],
        "product": carry["product"] * x["factor"]
    }

# PyTree input
xs = {
    "value": torch.randn(5, device='cuda'),
    "factor": torch.ones(5, device='cuda') + 0.1
}

# This would require implementing the combine function to handle the PyTree structure
# The associative_scan will handle flattening and unflattening automatically
```
### Combine Modes

The `combine_mode` parameter controls how the combine function is executed:

- **`"pointwise"`** (default): More efficient mode that requires:
  - The combine function to be pure and contain only pointwise operations
  - All input tensors to be on CUDA devices
  - No lifted arguments or closures in the combine function

- **`"generic"`**: More flexible mode that:
  - Works with any associative combine function
  - Allows more complex operations in the combine function

```python
# Pointwise mode - more efficient for simple operations
def simple_add(x, y):
    return x + y

result_pointwise = associative_scan(
    simple_add,
    torch.randn(100, device='cuda'),
    dim=0,
    combine_mode="pointwise"
)

# Generic mode - more flexible for complex operations
def complex_combine(x, y):
    return torch.matmul(x, y.transpose(-2, -1))

result_generic = associative_scan(
    complex_combine,
    torch.randn(10, 4, 4, device='cuda'),
    dim=0,
    combine_mode="generic"
)
```

(map)=

## torch._higher_order_ops.map

`torch._higher_order_ops.map` is a structured control flow operator that applies a function to each element along the first dimension of the input tensors. It's conceptually similar to Python's built-in `map` function but operates on tensor dimensions. Roughly, it's equivalent to the following Python code:

```python
def map(f, xs, *args):
    return torch.stack([f(xs[i], *args) for i in range(xs.size(0))])
```

### Examples

Here's a simple example that uses map to apply a function to each element along the first dimension:

```python
import torch

def add_and_sin(x):
    return x.sin() + torch.ones_like(x)

# Input tensor with shape [3, 4] - will map over the first dimension
x = torch.randn(3, 4)

# Result shape: [3, 4] - same as input
result = torch._higher_order_ops.map(add_and_sin, x)
```

```python
def add_with_constant(x, constant):
    return x + constant

x = torch.randn(5, 3, 4)
constant = torch.tensor(2.0)

# The constant will be passed to each invocation of the function
result = torch._higher_order_ops.map(add_with_constant, x, constant)
# Result shape: [5, 3, 4]
```

Here's an example using map with neural network modules:

```python
class SimpleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        def process_element(elem):
            return self.linear(elem).sin()

        # x has shape [batch_size, seq_len, 4]
        # We want to apply the linear layer to each sequence element
        x_reshaped = x.transpose(0, 1)  # [seq_len, batch_size, 4]
        result = torch._higher_order_ops.map(process_element, x_reshaped)  # [seq_len, batch_size, 2]
        return result.transpose(0, 1)  # [batch_size, seq_len, 2]

model = SimpleLinear()
x = torch.randn(8, 10, 4)  # batch_size=8, seq_len=10, features=4
output = model(x)  # [8, 10, 2]
```

We can compile a function with map for better performance:

```python

def f(x):
    def process_fn(x):
        return x.cos() * 2 + x.sin()
    return torch._higher_order_ops.map(process_fn, x)

# Compile the map function
compiled_f = torch.compile(f)

x = torch.randn(100, 50, 25)
result_compiled = compiled_f(process_fn, x)
result_eager = f(x)

assert torch.allclose(result_compiled, result_eager)
```

### Working with PyTree Structures

`map` supports PyTree structures for both inputs and outputs, allowing you to work with complex nested data structures:

```python
class PytreeMapModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, z):
        def process_pytree(inputs):
            # inputs is a dict with keys "x", "y_z"
            x = inputs["x"]
            y, (z,) = inputs["y_z"]
            return self.linear(x).sin(), (self.linear(y), z.cos())

        # Create PyTree input structure
        pytree_input = {"x": x, "y_z": (y, (z,))}

        return torch._higher_order_ops.map(process_pytree, pytree_input)

model = PytreeMapModel()
x = torch.randn(2, 5, 3)  # batch_size=2
y = torch.randn(2, 5, 3)
z = torch.randn(2, 4, 3)

result = model(x, y, z)
# result is a tuple: (tensor of shape [2, 5, 5], (tensor of shape [2, 5, 5], tensor of shape [2, 4, 3]))
```

## Constraints
Control flow operators have the following constraints:

- **Structure Match**: Inputs and outputs must match structure in two cases:
  - **Output match**: For `cond`, both `true_fn` and `false_fn` must return the same container structure (e.g., both return a list of 2 tensors). This prevents downstream divergence that could create exponential branches.
  - **Carry match**: For loops (`while_loop`, `scan`, `associative_scan`), carries must have matching structure across iterations. Since the IR assumes fixed input structure for `combine_fn`, `while_loop` outputs must match next iteration inputs, and `scan`'s init must match the output carry structure.

- **Tensor Match**: Corresponding tensors in the structure must have matching device, dtype, dimensionality and etc.

- **No side effects**: Function arguments to control flow operators must not create side effects such as mutating objects created outside of the subgraph (appending to a lists, deleting dictionary keys, or setting global attributes).

- **Aliasing and mutation**: There shouldn't be any aliasing or mutation between the inputs and outputs of the control flow operators.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.cond.cond
.. autofunction:: torch._higher_order_ops.while_loop.while_loop
.. autofunction:: torch._higher_order_ops.scan.scan
.. autofunction:: torch._higher_order_ops.associative_scan.associative_scan
.. autofunction:: torch._higher_order_ops.map.map
```
