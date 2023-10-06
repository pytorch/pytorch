.. _torch.export:

torch.export
=====================

.. warning::
    This feature is a prototype under active development and there WILL BE
    BREAKING CHANGES in the future.


Overview
--------

:func:`torch.export.export` takes an arbitrary Python callable (a
:class:`torch.nn.Module`, a function or a method) and produces a traced graph
representing only the Tensor computation of the function in an Ahead-of-Time
(AOT) fashion, which can subsequently be executed with different outputs or
serialized.

::

    import torch
    from torch.export import export

    def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

    example_args = (torch.randn(10, 10), torch.randn(10, 10))

    exported_program: torch.export.ExportedProgram = export(
        f, args=example_args
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[10, 10], arg1_1: f32[10, 10]):
                # code: a = torch.sin(x)
                sin: f32[10, 10] = torch.ops.aten.sin.default(arg0_1);

                # code: b = torch.cos(y)
                cos: f32[10, 10] = torch.ops.aten.cos.default(arg1_1);

                # code: return a + b
                add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos);
                return (add,)

        Graph signature: ExportGraphSignature(
            parameters=[],
            buffers=[],
            user_inputs=['arg0_1', 'arg1_1'],
            user_outputs=['add'],
            inputs_to_parameters={},
            inputs_to_buffers={},
            buffers_to_mutate={},
            backward_signature=None,
            assertion_dep_token=None,
        )
        Range constraints: {}
        Equality constraints: []

``torch.export`` produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found
:ref:`here <export.ir_spec>`.

* **Soundness**: It is guaranteed to be a sound representation of the original
  program, and maintains the same calling conventions of the original program.

* **Normalized**: There are no Python semantics within the graph. Submodules
  from the original programs are inlined to form one fully flattened
  computational graph.

* **Defined Operator Set**: The graph produced contains only a small defined
  :ref:`Core ATen IR <torch.compiler_ir>` opset and registered custom
  operators.

* **Graph properties**: The graph is purely functional, meaning it does not
  contain operations with side effects such as mutations or aliasing. It does
  not mutate any intermediate values, parameters, or buffers.

* **Metadata**: The graph contains metadata captured during tracing, such as a
  stacktrace from user's code.

Under the hood, ``torch.export`` leverages the following latest technologies:

* **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature
  called the Frame Evaluation API to safely trace PyTorch graphs. This
  provides a massively improved graph capturing experience, with much fewer
  rewrites needed in order to fully trace the PyTorch code.

* **AOT Autograd** provides a functionalized PyTorch graph and ensures the graph
  is decomposed/lowered to the small defined Core ATen operator set.

* **Torch FX (torch.fx)** is the underlying representation of the graph,
  allowing flexible Python-based transformations.


Existing frameworks
^^^^^^^^^^^^^^^^^^^

:func:`torch.compile` also utilizes the same PT2 stack as ``torch.export``, but
is slightly different:

* **JIT vs. AOT**: :func:`torch.compile` is a JIT compiler whereas
  which is not intended to be used to produce compiled artifacts outside of
  deployment.

* **Partial vs. Full Graph Capture**: When :func:`torch.compile` runs into an
  untraceable part of a model, it will "graph break" and fall back to running
  the program in the eager Python runtime. In comparison, ``torch.export`` aims
  to get a full graph representation of a PyTorch model, so it will error out
  when something untraceable is reached. Since ``torch.export`` produces a full
  graph disjoint from any Python features or runtime, this graph can then be
  saved, loaded, and run in different environments and languages.

* **Usability tradeoff**: Since :func:`torch.compile` is able to fallback to the
  Python runtime whenever it reaches something untraceable, it is a lot more
  flexible. ``torch.export`` will instead require users to provide more
  information or rewrite their code to make it traceable.

Compared to :func:`torch.fx.symbolic_trace`, ``torch.export`` traces using
TorchDynamo which operates at the Python bytecode level, giving it the ability
to trace arbitrary Python constructs not limited by what Python operator
overloading supports. Additionally, ``torch.export`` keeps fine-grained track of
tensor metadata, so that conditionals on things like tensor shapes do not
fail tracing. In general, ``torch.export`` is expected to work on more user
programs, and produce lower-level graphs (at the ``torch.ops.aten`` operator
level). Note that users can still use :func:`torch.fx.symbolic_trace` as a
preprocessing step before ``torch.export``.

Compared to :func:`torch.jit.script`, ``torch.export`` does not capture Python
control flow or data structures, but it supports more Python language features
than TorchScript (as it is easier to have comprehensive coverage over Python
bytecodes). The resulting graphs are simpler and only have straight line control
flow (except for explicit control flow operators).

Compared to :func:`torch.jit.trace`, ``torch.export`` is sound: it is able to
trace code that performs integer computation on sizes and records all of the
side-conditions necessary to show that a particular trace is valid for other
inputs.


Exporting a PyTorch Model
-------------------------

An Example
^^^^^^^^^^

The main entrypoint is through :func:`torch.export.export`, which takes a
callable (:class:`torch.nn.Module`, function, or method) and sample inputs, and
captures the computation graph into an :class:`torch.export.ExportedProgram`. An
example:

::

    import torch
    from torch.export import export

    # Simple module for demonstration
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            )
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

        def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
            a = self.conv(x)
            a.add_(constant)
            return self.maxpool(self.relu(a))

    example_args = (torch.randn(1, 3, 256, 256),)
    example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

    exported_program: torch.export.ExportedProgram = export(
        M(), args=example_args, kwargs=example_kwargs
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[16, 3, 3, 3], arg1_1: f32[16], arg2_1: f32[1, 3, 256, 256], arg3_1: f32[1, 16, 256, 256]):

                # code: a = self.conv(x)
                convolution: f32[1, 16, 256, 256] = torch.ops.aten.convolution.default(
                    arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
                );

                # code: a.add_(constant)
                add: f32[1, 16, 256, 256] = torch.ops.aten.add.Tensor(convolution, arg3_1);

                # code: return self.maxpool(self.relu(a))
                relu: f32[1, 16, 256, 256] = torch.ops.aten.relu.default(add);
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(
                    relu, [3, 3], [3, 3]
                );
                getitem: f32[1, 16, 85, 85] = max_pool2d_with_indices[0];
                return (getitem,)

        Graph signature: ExportGraphSignature(
            parameters=['L__self___conv.weight', 'L__self___conv.bias'],
            buffers=[],
            user_inputs=['arg2_1', 'arg3_1'],
            user_outputs=['getitem'],
            inputs_to_parameters={
                'arg0_1': 'L__self___conv.weight',
                'arg1_1': 'L__self___conv.bias',
            },
            inputs_to_buffers={},
            buffers_to_mutate={},
            backward_signature=None,
            assertion_dep_token=None,
        )
        Range constraints: {}
        Equality constraints: []

Inspecting the ``ExportedProgram``, we can note the following:

* The :class:`torch.fx.Graph` contains the computation graph of the original
  program, along with records of the original code for easy debugging.

* The graph contains only ``torch.ops.aten`` operators found in the
  :ref:`Core ATen IR <torch.compiler_ir>` opset and custom operators, and is
  fully functional, without any inplace operators such as ``torch.add_``.

* The parameters (weight and bias to conv) are lifted as inputs to the graph,
  resulting in no ``get_attr`` nodes in the graph, which previously existed in
  the result of :func:`torch.fx.symbolic_trace`.

* The :class:`torch.export.ExportGraphSignature` models the input and output
  signature, along with specifying which inputs are parameters.

* The resulting shape and dtype of tensors produced by each node in the graph is
  noted. For example, the ``convolution`` node will result in a tensor of dtype
  ``torch.float32`` and shape (1, 16, 256, 256).


Expressing Dynamism
^^^^^^^^^^^^^^^^^^^

By default ``torch.export`` will trace the program assuming all input shapes are
**static**, and specializing the exported program to those dimensions. However,
some dimensions, such as a batch dimension, can be dynamic and vary from run to
run. Such dimensions must be specified by using the
:func:`torch.export.Dim` API to create them and by passing them into
:func:`torch.export.export` through the ``dynamic_shapes`` argument. An example:

::

    import torch
    from torch.export import Dim, export

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(
                torch.nn.Linear(64, 32), torch.nn.ReLU()
            )
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example_args = (torch.randn(32, 64), torch.randn(32, 128))

    # Create a dynamic batch size
    batch = Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    exported_program: torch.export.ExportedProgram = export(
        M(), args=example_args, dynamic_shapes=dynamic_shapes
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[32, 64], arg1_1: f32[32], arg2_1: f32[64, 128], arg3_1: f32[64], arg4_1: f32[32], arg5_1: f32[s0, 64], arg6_1: f32[s0, 128]):

                # code: out1 = self.branch1(x1)
                permute: f32[64, 32] = torch.ops.aten.permute.default(arg0_1, [1, 0]);
                addmm: f32[s0, 32] = torch.ops.aten.addmm.default(arg1_1, arg5_1, permute);
                relu: f32[s0, 32] = torch.ops.aten.relu.default(addmm);

                # code: out2 = self.branch2(x2)
                permute_1: f32[128, 64] = torch.ops.aten.permute.default(arg2_1, [1, 0]);
                addmm_1: f32[s0, 64] = torch.ops.aten.addmm.default(arg3_1, arg6_1, permute_1);
                relu_1: f32[s0, 64] = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None

                # code: return (out1 + self.buffer, out2)
                add: f32[s0, 32] = torch.ops.aten.add.Tensor(relu, arg4_1);
                return (add, relu_1)

        Graph signature: ExportGraphSignature(
            parameters=[
                'branch1.0.weight',
                'branch1.0.bias',
                'branch2.0.weight',
                'branch2.0.bias',
            ],
            buffers=['L__self___buffer'],
            user_inputs=['arg5_1', 'arg6_1'],
            user_outputs=['add', 'relu_1'],
            inputs_to_parameters={
                'arg0_1': 'branch1.0.weight',
                'arg1_1': 'branch1.0.bias',
                'arg2_1': 'branch2.0.weight',
                'arg3_1': 'branch2.0.bias',
            },
            inputs_to_buffers={'arg4_1': 'L__self___buffer'},
            buffers_to_mutate={},
            backward_signature=None,
            assertion_dep_token=None,
        )
        Range constraints: {s0: RangeConstraint(min_val=2, max_val=9223372036854775806)}
        Equality constraints: [(InputDim(input_name='arg5_1', dim=0), InputDim(input_name='arg6_1', dim=0))]

Some additional things to note:

* Through the :func:`torch.export.Dim` API and the ``dynamic_shapes`` argument, we specified the first
  dimension of each input to be dynamic. Looking at the inputs ``arg5_1`` and
  ``arg6_1``, they have a symbolic shape of (s0, 64) and (s0, 128), instead of
  the (32, 64) and (32, 128) shaped tensors that we passed in as example inputs.
  ``s0`` is a symbol representing that this dimension can be a range
  of values.

* ``exported_program.range_constraints`` describes the ranges of each symbol
  appearing in the graph. In this case, we see that ``s0`` has the range
  [2, inf]. For technical reasons that are difficult to explain here, they are
  assumed to be not 0 or 1. This is not a bug, and does not necessarily mean
  that the exported program will not work for dimensions 0 or 1. See
  `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`_
  for an in-depth discussion of this topic.

* ``exported_program.equality_constraints`` describes which dimensions are
  required to be equal. Since we specified in the constraints that the first
  dimension of each argument is equivalent,
  (``dynamic_dim(example_args[0], 0) == dynamic_dim(example_args[1], 0)``),
  we see in the equality constraints the tuple specifying that ``arg5_1``
  dimension 0 and ``arg6_1`` dimension 0 are equal.

(A legacy mechanism for specifying dynamic shapes
involves marking and constraining dynamic dimensions with the
:func:`torch.export.dynamic_dim` API and passing them into :func:`torch.export.export`
through the ``constraints`` argument. That mechanism is now **deprecated** and will
not be supported in the future.)


Serialization
^^^^^^^^^^^^^

To save the ``ExportedProgram``, users can use the :func:`torch.export.save` and
:func:`torch.export.load` APIs. A convention is to save the ``ExportedProgram``
using a ``.pt2`` file extension.

An example:

::

    import torch
    import io

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 10

    exported_program = torch.export.export(MyModule(), torch.randn(5))

    torch.export.save(exported_program, 'exported_program.pt2')
    saved_exported_program = torch.export.load('exported_program.pt2')


Specialization
^^^^^^^^^^^^^^

Input shapes
~~~~~~~~~~~~

As mentioned before, by default, ``torch.export`` will trace the program
specializing on the input tensors' shapes, unless a dimension is specified as
dynamic via the :func:`torch.export.dynamic_dim` API. This means that if there
exists shape-dependent control flow, ``torch.export`` will specialize on the
branch that is being taken with the given sample inputs. For example:

::

    import torch
    from torch.export import export

    def fn(x):
        if x.shape[0] > 5:
            return x + 1
        else:
            return x - 1

    example_inputs = (torch.rand(10, 2),)
    exported_program = export(fn, example_inputs)
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[10, 2]):
                add: f32[10, 2] = torch.ops.aten.add.Tensor(arg0_1, 1);
                return (add,)

The conditional of (``x.shape[0] > 5``) does not appear in the
``ExportedProgram`` because the example inputs have the static
shape of (10, 2). Since ``torch.export`` specializes on the inputs' static
shapes, the else branch (``x - 1``) will never be reached. To preserve the dynamic
branching behavior based on the shape of a tensor in the traced graph,
:func:`torch.export.dynamic_dim` will need to be used to specify the dimension
of the input tensor (``x.shape[0]``) to be dynamic, and the source code will
need to be :ref:`rewritten <Data/Shape-Dependent Control Flow>`.

Non-tensor inputs
~~~~~~~~~~~~~~~~~

``torch.export`` also specializes the traced graph based on the values of inputs
that are not ``torch.Tensor``, such as ``int``, ``float``, ``bool``, and ``str``.
However, we will likely change this in the near future to not specialize on
inputs of primitive types.

For example:

::

    import torch
    from torch.export import export

    def fn(x: torch.Tensor, const: int, times: int):
        for i in range(times):
            x = x + const
        return x

    example_inputs = (torch.rand(2, 2), 1, 3)
    exported_program = export(fn, example_inputs)
    print(exported_program)

.. code-block::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[2, 2], arg1_1, arg2_1):
                add: f32[2, 2] = torch.ops.aten.add.Tensor(arg0_1, 1);
                add_1: f32[2, 2] = torch.ops.aten.add.Tensor(add, 1);
                add_2: f32[2, 2] = torch.ops.aten.add.Tensor(add_1, 1);
                return (add_2,)

Because integers are specialized, the ``torch.ops.aten.add.Tensor`` operations
are all computed with the inlined constant ``1``, rather than ``arg1_1``.
Additionally, the ``times`` iterator used in the ``for`` loop is also "inlined"
in the graph through the 3 repeated ``torch.ops.aten.add.Tensor`` calls, and the
input ``arg2_1`` is never used.


Limitations of torch.export
---------------------------

Graph Breaks
^^^^^^^^^^^^

As ``torch.export`` is a one-shot process for capturing a computation graph from
a PyTorch program, it might ultimately run into untraceable parts of programs as
it is nearly impossible to support tracing all PyTorch and Python features. In
the case of ``torch.compile``, an unsupported operation will cause a "graph
break" and the unsupported operation will be run with default Python evaluation.
In contrast, ``torch.export`` will require users to provide additional
information or rewrite parts of their code to make it traceable. As the
tracing is based on TorchDynamo, which evaluates at the Python
bytecode level, there will be significantly fewer rewrites required compared to
previous tracing frameworks.

When a graph break is encountered, :ref:`ExportDB <torch.export_db>` is a great
resource for learning about the kinds of programs that are supported and
unsupported, along with ways to rewrite programs to make them traceable.

.. _Data/Shape-Dependent Control Flow:

Data/Shape-Dependent Control Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Graph breaks can also be encountered on data-dependent control flow (``if
x.shape[0] > 2``) when shapes are not being specialized, as a tracing compiler cannot
possibly deal with without generating code for a combinatorially exploding
number of paths. In such cases, users will need to rewrite their code using
special control flow operators. Currently, we support :ref:`torch.cond <control_flow_cond>`
to express if-else like control flow (more coming soon!).

Data-Dependent Accesses
^^^^^^^^^^^^^^^^^^^^^^^

Data dependent behavior such as using the value inside of a tensor to construct
another tensor, or using the value of a tensor to slice into another tensor, is
also something the tracer cannot fully determine. Users will need to rewrite
their code using the inline constraint APIs
:func:`torch.export.constrain_as_size` and
:func:`torch.export.constrain_as_value`.

Missing Meta Kernels for Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When tracing, a META implementation (or "meta kernel") is required for all
operators. This is used to reason about the input/output shapes for this
operator.

Note that the official API for registering custom meta kernels for custom ops is
currently undergoing development. While the final API is being refined, you can
refer to the documentation `here <https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0>`_.

In the unfortunate case where your model uses an ATen operator that is does not
have a meta kernel implementation yet, please file an issue.


Read More
---------

.. toctree::
   :caption: Additional Links for Export Users
   :maxdepth: 1

   export.ir_spec
   torch.compiler_transformations
   torch.compiler_ir
   generated/exportdb/index
   control_flow_cond

.. toctree::
   :caption: Deep Dive for PyTorch Developers
   :maxdepth: 1

   torch.compiler_deepdive
   torch.compiler_dynamic_shapes
   torch.compiler_fake_tensor


API Reference
-------------

.. automodule:: torch.export
.. autofunction:: export
.. autofunction:: dynamic_dim
.. autofunction:: constrain_as_size
.. autofunction:: constrain_as_value
.. autofunction:: save
.. autofunction:: load
.. autofunction:: register_dataclass
.. autofunction:: Dim
.. autofunction:: dims
.. autoclass:: Constraint
.. autoclass:: ExportedProgram

    .. automethod:: module
    .. automethod:: buffers
    .. automethod:: named_buffers
    .. automethod:: parameters
    .. automethod:: named_parameters

.. autoclass:: ExportBackwardSignature
.. autoclass:: ExportGraphSignature
.. autoclass:: ModuleCallSignature
.. autoclass:: ModuleCallEntry


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.export.exported_program
