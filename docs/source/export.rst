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

    class Mod(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a = torch.sin(x)
            b = torch.cos(y)
            return a + b

    example_args = (torch.randn(10, 10), torch.randn(10, 10))

    exported_program: torch.export.ExportedProgram = export(
        Mod(), args=example_args
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

``torch.export`` produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found
:ref:`here <export.ir_spec>`.

* **Soundness**: It is guaranteed to be a sound representation of the original
  program, and maintains the same calling conventions of the original program.

* **Normalized**: There are no Python semantics within the graph. Submodules
  from the original programs are inlined to form one fully flattened
  computational graph.

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
  is decomposed/lowered to the ATen operator set.

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

Inspecting the ``ExportedProgram``, we can note the following:

* The :class:`torch.fx.Graph` contains the computation graph of the original
  program, along with records of the original code for easy debugging.

* The graph contains only ``torch.ops.aten`` operators found `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml>`__
  and custom operators, and is fully functional, without any inplace operators
  such as ``torch.add_``.

* The parameters (weight and bias to conv) are lifted as inputs to the graph,
  resulting in no ``get_attr`` nodes in the graph, which previously existed in
  the result of :func:`torch.fx.symbolic_trace`.

* The :class:`torch.export.ExportGraphSignature` models the input and output
  signature, along with specifying which inputs are parameters.

* The resulting shape and dtype of tensors produced by each node in the graph is
  noted. For example, the ``convolution`` node will result in a tensor of dtype
  ``torch.float32`` and shape (1, 16, 256, 256).


.. _Non-Strict Export:

Non-Strict Export
^^^^^^^^^^^^^^^^^

In PyTorch 2.3, we introduced a new mode of tracing called **non-strict mode**.
It's still going through hardening, so if you run into any issues, please file
them to Github with the "oncall: export" tag.

In *non-strict mode*, we trace through the program using the Python interpreter.
Your code will execute exactly as it would in eager mode; the only difference is
that all Tensor objects will be replaced by ProxyTensors, which will record all
their operations into a graph.

In *strict* mode, which is currently the default, we first trace through the
program using TorchDynamo, a bytecode analysis engine. TorchDynamo does not
actually execute your Python code. Instead, it symbolically analyzes it and
builds a graph based on the results. This analysis allows torch.export to
provide stronger guarantees about safety, but not all Python code is supported.

An example of a case where one might want to use non-strict mode is if you run
into a unsupported TorchDynamo feature that might not be easily solved, and you
know the python code is not exactly needed for computation. For example:

::

    import contextlib
    import torch

    class ContextManager():
        def __init__(self):
            self.count = 0
        def __enter__(self):
            self.count += 1
        def __exit__(self, exc_type, exc_value, traceback):
            self.count -= 1

    class M(torch.nn.Module):
        def forward(self, x):
            with ContextManager():
                return x.sin() + x.cos()

    export(M(), (torch.ones(3, 3),), strict=False)  # Non-strict traces successfully
    export(M(), (torch.ones(3, 3),))  # Strict mode fails with torch._dynamo.exc.Unsupported: ContextManager

In this example, the first call using non-strict mode (through the
``strict=False`` flag) traces successfully whereas the second call using strict
mode (default) results with a failure, where TorchDynamo is unable to support
context managers. One option is to rewrite the code (see :ref:`Limitations of torch.export <Limitations of
torch.export>`), but seeing as the context manager does not affect the tensor
computations in the model, we can go with the non-strict mode's result.


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


We can also specify more expressive relationships between input shapes, such as
where a pair of shapes might differ by one, a shape might be double of
another, or a shape is even. An example:

::

    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y[1:]

    x, y = torch.randn(5), torch.randn(6)
    dimx = torch.export.Dim("dimx", min=3, max=6)
    dimy = dimx + 1

    exported_program = torch.export.export(
        M(), (x, y), dynamic_shapes=({0: dimx}, {0: dimy}),
    )
    print(exported_program)

.. code-block::

    ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: "f32[s0]", arg1_1: "f32[s0 + 1]"):
            # code: return x + y[1:]
            slice_1: "f32[s0]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, 9223372036854775807);  arg1_1 = None
            add: "f32[s0]" = torch.ops.aten.add.Tensor(arg0_1, slice_1);  arg0_1 = slice_1 = None
            return (add,)

    Graph signature: ExportGraphSignature(
        input_specs=[
            InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg0_1'), target=None, persistent=None),
            InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg1_1'), target=None, persistent=None)
        ],
        output_specs=[
            OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add'), target=None)]
    )
    Range constraints: {s0: ValueRanges(lower=3, upper=6, is_bool=False), s0 + 1: ValueRanges(lower=4, upper=7, is_bool=False)}

Some things to note:

* By specifying ``{0: dimx}`` for the first input, we see that the resulting
  shape of the first input is now dynamic, being ``[s0]``. And now by specifying
  ``{0: dimy}`` for the second input, we see that the resulting shape of the
  second input is also dynamic. However, because we expressed ``dimy = dimx + 1``,
  instead of ``arg1_1``'s shape containing a new symbol, we see that it is
  now being represented with the same symbol used in ``arg0_1``, ``s0``. We can
  see that relationship of ``dimy = dimx + 1`` is being shown through ``s0 + 1``.

* Looking at the range constraints, we see that ``s0`` has the range [3, 6],
  which is specified initially, and we can see that ``s0 + 1`` has the solved
  range of [4, 7].


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


Specializations
^^^^^^^^^^^^^^^

A key concept in understanding the behavior of ``torch.export`` is the
difference between *static* and *dynamic* values.

A *dynamic* value is one that can change from run to run. These behave like
normal arguments to a Python function—you can pass different values for an
argument and expect your function to do the right thing. Tensor *data* is
treated as dynamic.


A *static* value is a value that is fixed at export time and cannot change
between executions of the exported program. When the value is encountered during
tracing, the exporter will treat it as a constant and hard-code it into the
graph.

When an operation is performed (e.g. ``x + y``) and all inputs are static, then
the output of the operation will be directly hard-coded into the graph, and the
operation won’t show up (i.e. it will get constant-folded).

When a value has been hard-coded into the graph, we say that the graph has been
*specialized* to that value.

The following values are static:

Input Tensor Shapes
~~~~~~~~~~~~~~~~~~~

By default, ``torch.export`` will trace the program specializing on the input
tensors' shapes, unless a dimension is specified as dynamic via the
``dynamic_shapes`` argument to ``torch.export``. This means that if there exists
shape-dependent control flow, ``torch.export`` will specialize on the branch
that is being taken with the given sample inputs. For example:

::

    import torch
    from torch.export import export

    class Mod(torch.nn.Module):
        def forward(self, x):
            if x.shape[0] > 5:
                return x + 1
            else:
                return x - 1

    example_inputs = (torch.rand(10, 2),)
    exported_program = export(Mod(), example_inputs)
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
:func:`torch.export.Dim` will need to be used to specify the dimension
of the input tensor (``x.shape[0]``) to be dynamic, and the source code will
need to be :ref:`rewritten <Data/Shape-Dependent Control Flow>`.

Note that tensors that are part of the module state (e.g. parameters and
buffers) always have static shapes.

Python Primitives
~~~~~~~~~~~~~~~~~

``torch.export`` also specializes on Python primtivies,
such as ``int``, ``float``, ``bool``, and ``str``. However they do have dynamic
variants such as ``SymInt``, ``SymFloat``, and ``SymBool``.

For example:

::

    import torch
    from torch.export import export

    class Mod(torch.nn.Module):
        def forward(self, x: torch.Tensor, const: int, times: int):
            for i in range(times):
                x = x + const
            return x

    example_inputs = (torch.rand(2, 2), 1, 3)
    exported_program = export(Mod(), example_inputs)
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
are all computed with the hard-coded constant ``1``, rather than ``arg1_1``. If
a user passes a different value for ``arg1_1`` at runtime, like 2, than the one used
during export time, 1, this will result in an error.
Additionally, the ``times`` iterator used in the ``for`` loop is also "inlined"
in the graph through the 3 repeated ``torch.ops.aten.add.Tensor`` calls, and the
input ``arg2_1`` is never used.

Python Containers
~~~~~~~~~~~~~~~~~

Python containers (``List``, ``Dict``, ``NamedTuple``, etc.) are considered to
have static structure.


.. _Limitations of torch.export:

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

An option to get past dealing with this graph breaks is by using
:ref:`non-strict export <Non-Strict Export>`

.. _Data/Shape-Dependent Control Flow:

Data/Shape-Dependent Control Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Graph breaks can also be encountered on data-dependent control flow (``if
x.shape[0] > 2``) when shapes are not being specialized, as a tracing compiler cannot
possibly deal with without generating code for a combinatorially exploding
number of paths. In such cases, users will need to rewrite their code using
special control flow operators. Currently, we support :ref:`torch.cond <cond>`
to express if-else like control flow (more coming soon!).

Missing Fake/Meta/Abstract Kernels for Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When tracing, a FakeTensor kernel (aka meta kernel, abstract impl) is
required for all operators. This is used to reason about the input/output shapes
for this operator.

Please see :func:`torch.library.register_fake` for more details.

In the unfortunate case where your model uses an ATen operator that is does not
have a FakeTensor kernel implementation yet, please file an issue.


Read More
---------

.. toctree::
   :caption: Additional Links for Export Users
   :maxdepth: 1

   export.ir_spec
   torch.compiler_transformations
   torch.compiler_ir
   generated/exportdb/index
   cond

.. toctree::
   :caption: Deep Dive for PyTorch Developers
   :maxdepth: 1

   torch.compiler_dynamo_overview
   torch.compiler_dynamo_deepdive
   torch.compiler_dynamic_shapes
   torch.compiler_fake_tensor


API Reference
-------------

.. automodule:: torch.export
.. autofunction:: export
.. autofunction:: save
.. autofunction:: load
.. autofunction:: register_dataclass
.. autofunction:: torch.export.dynamic_shapes.Dim
.. autofunction:: dims
.. autoclass:: torch.export.dynamic_shapes.ShapesCollection

    .. automethod:: dynamic_shapes

.. autofunction:: torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes
.. autoclass:: Constraint
.. autoclass:: ExportedProgram

    .. automethod:: module
    .. automethod:: buffers
    .. automethod:: named_buffers
    .. automethod:: parameters
    .. automethod:: named_parameters
    .. automethod:: run_decompositions

.. autoclass:: ExportBackwardSignature
.. autoclass:: ExportGraphSignature
.. autoclass:: ModuleCallSignature
.. autoclass:: ModuleCallEntry


.. automodule:: torch.export.exported_program
.. automodule:: torch.export.graph_signature
.. autoclass:: InputKind
.. autoclass:: InputSpec
.. autoclass:: OutputKind
.. autoclass:: OutputSpec
.. autoclass:: ExportGraphSignature

    .. automethod:: replace_all_uses
    .. automethod:: get_replace_hook

.. autoclass:: torch.export.graph_signature.CustomObjArgument

.. py:module:: torch.export.dynamic_shapes

.. automodule:: torch.export.unflatten
    :members:

.. automodule:: torch.export.custom_obj

.. automodule:: torch.export.experimental
.. automodule:: torch.export.passes
.. autofunction:: torch.export.passes.move_to_device_pass
