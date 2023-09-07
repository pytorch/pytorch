.. _torch.export:

torch.export
=====================

.. warning::
    This feature is a prototype and may have compatibility breaking changes in the future.


Overview
--------

``torch.export`` is an ahead-of-time (AOT) process built to fully convert
define-by-run PyTorch code into a static and standardized model representation,
with strong soundness guarantees. It is designed to streamline and standardize the
way PyTorch interacts with various systems and platforms.

``torch.export`` creates a full graph representation of a PyTorch model
that can be saved and loaded and run in different environments and
languages. In comparison, ``torch.compile`` is a JIT compiler which provides
the flexibility of creating partial graphs, where untraceable parts of models
result in a "graph break" and will be defaulted back to the eager Python runtime.

``torch.export`` maintains a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found here (coming
soon!).

* **Normalized**: There are no Python semantics within the graph

* **Defined Operator Set**: The graph produced contains a small defined
  :ref:`Core ATen IR <torch.compiler_ir>` opset.

* **Graph properties**: The graph is purely functional, meaning it has no side
  effects.

*  **Metadata**: The graph contains metadata captured during tracing, such as a
   stacktrace from user's code.

Under the hood, ``torch.export`` leverages the following latest technologies:

* **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature
  called the Frame Evaluation API to safely trace PyTorch graphs. This
  provides a massively improved graph capturing experience, with much fewer
  rewrites needed in order to fully trace the PyTorch code.

* **AOT Autograd** provides a functionalized PyTorch graph and ensures the graph
  is lowered to the small defined Core ATen operator set.

* **Torch FX (torch.fx)** is the underlying representation of the graph,
  allowing flexible Python-based transformations.


Exporting a PyTorch Model
-------------------------

The main entrypoint is through :func:`torch.export.export`, which takes a
callable (:class:`torch.nn.Module`, function, or method) and captures the
computation graph into an :class:`torch.export.ExportedProgram`. An example:

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
    """
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[16, 3, 3, 3], arg1_1: f32[16], arg2_1: f32[1, 3, 256, 256], arg3_1: f32[1, 16, 256, 256]):
                # code: a = self.conv(x)
                convolution: f32[1, 16, 256, 256] = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None

                # code: a.add_(constant)
                add: f32[1, 16, 256, 256] = torch.ops.aten.add.Tensor(convolution, arg3_1);  convolution = arg3_1 = None

                # code: return self.maxpool(self.relu(a))
                relu: f32[1, 16, 256, 256] = torch.ops.aten.relu.default(add);  add = None
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [3, 3]);  relu = None
                getitem: f32[1, 16, 85, 85] = max_pool2d_with_indices[0];  max_pool2d_with_indices = None
                return (getitem,)

    Graph signature: ExportGraphSignature(parameters=['L__self___conv.weight', 'L__self___conv.bias'], buffers=[], user_inputs=['arg2_1', 'arg3_1'], user_outputs=['getitem'], inputs_to_parameters={'arg0_1': 'L__self___conv.weight', 'arg1_1': 'L__self___conv.bias'}, inputs_to_buffers={}, buffers_to_mutate={}, backward_signature=None, assertion_dep_token=None)
    Range constraints: {}
    Equality constraints: []
    """

Inspecting the ``ExportedProgram``, we can note the following:

* The :class:`torch.fx.Graph` contains the computation graph of the original
  program, along with records of the original code for easy debugging.

* The graph contains only ``torch.ops.aten`` operators found in the
  :ref:`Core ATen IR <torch.compiler_ir>` opset, and is fully functional,
  without any inplace operators such as ``torch.add_``.

* The parameters (weight and bias to conv) are lifted as inputs to the graph,
  resulting in no ``get_attr`` nodes in the graph, which previously existed in
  the result of :func:`torch.fx.symbolic_trace`.

* The :class:`torch.export.ExportGraphSignature` models the input and output
  signature, along with specifying which inputs are parameters.

* The resulting shape and dtype of tensors produced by each node in the graph is
  noted. For example, the ``convolution`` node will result in a tensor of dtype
  ``torch.float32`` and shape [1, 16, 256, 256].

By default ``torch.export`` will trace the program assuming all input shapes are
**static**, and specializing the exported program to those dimensions. However,
some dimensions, such as a batch dimension, are not expected to not be the same
all the time. Such dimensions must be marked dynamic using the
:func:`torch.export.dynamic_dim` API. An example:

::

    import torch
    from torch.export import export, dynamic_dim

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(
                torch.nn.Linear(64, 32), torch.nn.ReLU()
            )
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1, out2)

    example_args = (torch.randn(32, 64), torch.randn(32, 128))
    constraints = [
        # First dimension of each input is a dynamic batch size
        dynamic_dim(example_args[0], 0),
        dynamic_dim(example_args[1], 0),
        # The dynamic batch size between the inputs are equal
        dynamic_dim(example_args[0], 0) == dynamic_dim(example_args[1], 0),
    ]

    exported_program: torch.export.ExportedProgram = export(
      M(), args=example_args, constraints=constraints
    )
    print(exported_program)
    """
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: f32[32, 64], arg1_1: f32[32], arg2_1: f32[64, 128], arg3_1: f32[64], arg4_1: f32[s0, 64], arg5_1: f32[s1, 128]):
                # code: out1 = self.branch1(x1)
                permute: f32[64, 32] = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
                addmm: f32[s0, 32] = torch.ops.aten.addmm.default(arg1_1, arg4_1, permute);  arg1_1 = arg4_1 = permute = None
                relu: f32[s0, 32] = torch.ops.aten.relu.default(addmm);  addmm = None

                # code: out2 = self.branch2(x2)
                permute_1: f32[128, 64] = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
                addmm_1: f32[s1, 64] = torch.ops.aten.addmm.default(arg3_1, arg5_1, permute_1);  arg3_1 = arg5_1 = permute_1 = None
                relu_1: f32[s1, 64] = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
                return (relu, relu_1)

    Graph signature: ExportGraphSignature(parameters=['L__self___branch1_0.weight', 'L__self___branch1_0.bias', 'L__self___branch2_0.weight', 'L__self___branch2_0.bias'], buffers=[], user_inputs=['arg4_1', 'arg5_1'], user_outputs=['relu', 'relu_1'], inputs_to_parameters={'arg0_1': 'L__self___branch1_0.weight', 'arg1_1': 'L__self___branch1_0.bias', 'arg2_1': 'L__self___branch2_0.weight', 'arg3_1': 'L__self___branch2_0.bias'}, inputs_to_buffers={}, buffers_to_mutate={}, backward_signature=None, assertion_dep_token=None)
    Range constraints: {s0: RangeConstraint(min_val=2, max_val=9223372036854775806), s1: RangeConstraint(min_val=2, max_val=9223372036854775806)}
    Equality constraints: [(InputDim(input_name='arg4_1', dim=0), InputDim(input_name='arg5_1', dim=0))]
    """

Some additional things to note:

* Through the :func:``torch.export.dynamic_dim`` API, we specified the first
  dimension of each input to be dynamic. Lookinga the inputs ``arg4_1`` and
  ``arg5_1``, they have a shape of [s0, 64] and [s1, 128], instead of the [32,
  64] and [32, 128] shaped tensors that we passed in as example input. ``s0``
  and ``s1`` are symbolic shapes which represents a range of values.

* ``exported_program.range_constraints`` describes the ranges of each symbolic
  shape appearing in the graph. In this case, we see that ``s0`` and ``s1`` have
  the range [2, inf]. For technical reasons that are difficult to explain here,
  they are assumed to be not 0 or 1. This is not a bug, and does not necessarily
  mean that the exported program will not work for dimensions 0 or 1. See
  `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`_
  for an in-depth discussion of this topic.

* ``exported_program.equality_constraints`` describes which dimensions are
  equivalent. Since we specified in the constraints that the first dimension of
  each argument is equivalent,
  (``dynamic_dim(example_args[0], 0) == dynamic_dim(example_args[1], 0)``),
  we see in the equality constraints the tuple specifying that ``arg4_1``
  dimension 0 and ``arg5_1`` dimension 0 are equal.


Limitations of torch.export
---------------------------

Graph Breaks
^^^^^^^^^^^^

As ``torch.export`` is a one-shot process for capturing a computation graph from
a PyTorch program, it will ultimately run into untraceable parts of programs as
it is nearly impossible to support tracing all PyTorch and Python features. In
the case of ``torch.compile``, untraceable parts of programs will "graph break"
and run the unsupported operation with default Python evaluation. However,
``torch.export`` will require users to provide additional information or rewrite
parts of their code to make it traceable. As the foundation of the tracing is
based on TorchDynamo, which evaluates at the Python bytecode level, there
will be significantly fewer rewrites required compared to previous tracing
frameworks.

When a graph break is encountered, :ref:`ExportDB <torch.export_db>` is a great
source of supported and unsupported programs, along with ways to rewrite
programs to make them traceable.

Control Flow
^^^^^^^^^^^^

Graph breaks can also be encountered on general control flow
(ex. ``if x[0] > 4``), which a tracing compiler cannot possibly deal with
without generating code for a combinatorially exploding number of paths. In such
cases, users will need to rewrite their code using special
**Control Flow Operators**.

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

Note that the official API for registering custom meta kernels is currently
undergoing development. While the final API is being refined, you can
refer to the documentation `here <https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0>`_.

In the unfortunate case where your model uses an ATen operator that is does not
have a meta kernel implementation yet, please file an issue to the Export team.


Read More
---------

.. toctree::
   :caption: Additional Links for Export Users
   :maxdepth: 1

   torch.compiler_transformations
   torch.compiler_ir
   generated/exportdb/index

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
.. autoclass:: Constraint
.. autoclass:: ExportedProgram

    .. automethod:: module
    .. automethod:: buffers
    .. automethod:: named_buffers
    .. automethod:: parameters
    .. automethod:: named_parameters

.. autoclass:: ExportBackwardSignature
.. autoclass:: ExportGraphSignature
.. autoclass:: ArgumentKind
.. autoclass:: ArgumentSpec
.. autoclass:: ModuleCallSignature
.. autoclass:: ModuleCallEntry
