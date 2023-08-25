Control Flow - Cond
====================

Export has a `cond` API used to help specify if-else like control flow within
some code, similar to JAX's control flow operators.

torch.cond
----------

The `cond` function represents an “if” statement in other programming languages.
It can logically be seen as implemented as follows:

.. code-block:: python

    def cond(
        pred: Union[bool, torch.Tensor],
        true_branch: Callable,
        false_branch: Callable,
        operands: Tuple[torch.Tensor]
    ):
        if pred:
            return true_branch(*operands)
        else:
            return false_branch(*operands)

Parameters
~~~~~~~~~~

- `pred (Union[bool, torch.Tensor])`: A boolean expression or a tensor with one element,
  indicating which branch function to apply, or a boolean expression.

- `true_branch (Callable)`: A callable function (a -> b) that is within the
  scope that is being traced.

- `false_branch (Callable)`: A callable function (a -> b) that is within the
  scope that is being traced. The true branch and false branch must have
  consistent input and outputs, meaning the inputs have to be the same, and
  the outputs have to be the same type and shape.

- `operands (Tuple[torch.Tensor])`: A list of inputs to the true/false
  branches.

Returns
~~~~~~~

- Value (b) of either `true_branch(*operands)` or `false_branch(*operands)`,
  depending on the value of `pred`.

Restrictions
~~~~~~~~~~~~

- The conditional statement (aka `pred`) must meet one of the following constraints:

  - It's a `torch.Tensor` with only one element, e.g. `torch.tensor(10)` or
    `torch.tensor([[10]])`, etc.

  - It's a boolean expression, e.g. `x.shape[0] > 10` or `x.dim() > 1 and x.shape[1] > 10`

- The branch function (aka `true_branch`/`false_branch`) must meet all of the following constraints:

  - The function signature must match with operands.

  - The function must return a tensor with the same metadata, e.g. shape,
    dtype, etc.

  - The function cannot have in-place mutations on inputs or global variables. (Note: in-place tensor operations such as `add_` for intermediate results are allowed in a branch)

Temporal Limitations
~~~~~~~~~~~~~~~~~~~~

- `Cond` is only supported for **inference** right now. Autograd will be allowed in the future.

- The **operands** must be a **tuple of tensors**. Pytree of tensors will be allowed in the future.

- The **output** of branches must be a **single Tensor**. Pytree of tensors will be allowed in the future.

Example
~~~~~~~

An example of how to use the `cond()` operator:

.. code-block:: python
    import torch
    from torch.export import export, dynamic_dim
    from torch._higer_order_ops.cond import cond_compiled as cond

    class DynamicShapeCondPredicate(torch.nn.Module):
        """
        A basic usage of control flow based on dynamic shape predicate.
        """

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            def true_fn(x: torch.Tensor):
                return x.cos()

            def false_fn(x: torch.Tensor):
                return x.sin()

            return cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    inp = torch.randn(4, 3)
    ep = export(DynamicShapeCondPredicate(), (inp,), {}, constraints=[dynamic_dim(inp, 0)]

torch.ops.higher_order.cond
===========================

`Cond` will be lowered to `torch.ops.higher_order.cond` in the IR. For the above example, if we run `ep.graph_module.print_readable()`, we can see the actual operator in the exported graph is `torch.ops.higher_order.cond` as shown below:

.. code-block:: python

    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sym_size: Sym(s0) = torch.ops.aten.sym_size.int(arg0_1, 0)
            gt: Sym(s0 > 4) = sym_size > 4;  sym_size = None
            submodule_0 = self.submodule_0
            submodule_1 = self.submodule_1
            cond: f32[s0, 3] = torch.ops.higher_order.cond(gt, submodule_0, submodule_1, [arg0_1]);  gt = submodule_0 = submodule_1 = arg0_1 = None
            return (cond,)

    # True graph module
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
            return cos

    # False graph module
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin

Note that even though this operator is directly usable but using the user-facing api is recommended.
Invariants
-----------

Compared with the user-facing API, there are several useful invariants for `torch.ops.higher_order.cond`.

- For predicate:
    - Dynamicness of predicate is preserved via sym_sizes (e.g. `gt` shown in the above example)
    - If the predicate in user-program is constant (e.g. boolean expression of shape of a static sized tensor or a python bool constant), the `pred` in IR node will be a constant.

- For branches:
    - The input and output signature will be a flattened tuple.
    - They are `torch.fx.GraphModule`.
    - Tensors used in the GraphModule are explicit inputs. No closures.
    - No mutations for inputs/globals.

- For operands:
    - It will also be a flat tuple.

- Nesting of `cond` in user program becomes nested graph modules.

See examples of advanced usage of `cond` operator in ExportDB.
