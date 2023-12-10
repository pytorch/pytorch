.. _cond:

Control Flow - Cond
====================

`torch.cond` is a structured control flow operator. It can be used to specify if-else like control flow
and can logically be seen as implemented as follows.

.. code-block:: python

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

Its unique power lies in its aibilty of expressing **data-dependent control flow**: it lowers to a conditional
operator (`torch.ops.higher_order.cond`), which preserves predicate, true function and false functions.
This unlocks great flexibilty in writing and deploying models that change model architecture based on
the **value** or **shape** of inputs or intermediate outputs of tensor operations.

.. warning::
    `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
    doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
    Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

Examples
~~~~~~~~

Below is an example that uses cond to branch based on input shape:

.. code-block:: python

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

We can eagerly run the model and expect the results vary based on input shape:

.. code-block:: python

    inp = torch.randn(3)
    inp2 = torch.randn(5)
    assert torch.equal(dyn_shape_mod(inp), false_fn(inp))
    assert torch.equal(dyn_shape_mod(inp2), true_fn(inp2))

We can export the model for further transformations and deployment:

.. code-block:: python

    inp = torch.randn(4, 3)
    dim_batch = torch.export.Dim("batch", min=2)
    ep = torch.export.export(DynamicShapeCondPredicate(), (inp,), {}, dynamic_shapes={"x": {0: dim_batch}})
    print(ep)

This gives us an exported program as shown below:

.. code-block::

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

Notice that `torch.cond` is lowered to `torch.ops.higher_order.cond`, its predicate becomes a Symbolic expression over the shape of input,
and branch functions becomes two sub-graph attributes of the top level graph module.

Here is another exmaple that showcases how to express a data-dependet control flow:

.. code-block:: python

    class DataDependentCondPredicacte(torch.nn.Module):
        """
        A basic usage of cond based on data dependent predicate.
        """
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cond(x.sum() > 4.0, true_fn, false_fn, (x,))

The exported program we get after export:

.. code-block::

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


Invariants of torch.ops.higher_order.cond
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several useful invariants for `torch.ops.higher_order.cond`:

- For predicate:
    - Dynamicness of predicate is preserved (e.g. `gt` shown in the above example)
    - If the predicate in user-program is constant (e.g. a python bool constant), the `pred` of the operator will be a constant.

- For branches:
    - The input and output signature will be a flattened tuple.
    - They are `torch.fx.GraphModule`.
    - Closures in original function becomes explicit inputs. No closures.
    - No mutations on inputs or globals are allowed.

- For operands:
    - It will also be a flat tuple.

- Nesting of `torch.cond` in user program becomes nested graph modules.


API Reference
-------------
.. autofunction:: torch._higher_order_ops.cond.cond
