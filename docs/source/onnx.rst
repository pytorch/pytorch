torch.onnx
============

.. contents:: :local:

.. automodule:: torch.onnx

Example: End-to-end AlexNet from PyTorch to ONNX
------------------------------------------------

Here is a simple script which exports a pretrained AlexNet as defined in
torchvision into ONNX.  It runs a single round of inference and then
saves the resulting traced model to ``alexnet.onnx``::

    import torch
    import torchvision

    dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
    model = torchvision.models.alexnet(pretrained=True).cuda()

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

The resulting ``alexnet.onnx`` is a binary protobuf file which contains both
the network structure and parameters of the model you exported
(in this case, AlexNet).  The keyword argument ``verbose=True`` causes the
exporter to print out a human-readable representation of the network::

    # These are the inputs and parameters to the network, which have taken on
    # the names we specified earlier.
    graph(%actual_input_1 : Float(10, 3, 224, 224)
          %learned_0 : Float(64, 3, 11, 11)
          %learned_1 : Float(64)
          %learned_2 : Float(192, 64, 5, 5)
          %learned_3 : Float(192)
          # ---- omitted for brevity ----
          %learned_14 : Float(1000, 4096)
          %learned_15 : Float(1000)) {
      # Every statement consists of some output tensors (and their types),
      # the operator to be run (with its attributes, e.g., kernels, strides,
      # etc.), its input tensors (%actual_input_1, %learned_0, %learned_1)
      %17 : Float(10, 64, 55, 55) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1), scope: AlexNet/Sequential[features]/Conv2d[0]
      %18 : Float(10, 64, 55, 55) = onnx::Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
      %19 : Float(10, 64, 27, 27) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2]
      # ---- omitted for brevity ----
      %29 : Float(10, 256, 6, 6) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
      # Dynamic means that the shape is not known. This may be because of a
      # limitation of our implementation (which we would like to fix in a
      # future release) or shapes which are truly dynamic.
      %30 : Dynamic = onnx::Shape(%29), scope: AlexNet
      %31 : Dynamic = onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet
      %32 : Long() = onnx::Squeeze[axes=[0]](%31), scope: AlexNet
      %33 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
      # ---- omitted for brevity ----
      %output1 : Float(10, 1000) = onnx::Gemm[alpha=1, beta=1, broadcast=1, transB=1](%45, %learned_14, %learned_15), scope: AlexNet/Sequential[classifier]/Linear[6]
      return (%output1);
    }

You can also verify the protobuf using the `ONNX <https://github.com/onnx/onnx/>`_ library.
You can install ``ONNX`` with conda::

    conda install -c conda-forge onnx

Then, you can run::

    import onnx

    # Load the ONNX model
    model = onnx.load("alexnet.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

To run the exported script with `caffe2 <https://caffe2.ai/>`_, you will need to install `caffe2`: If you don't have one already, Please `follow the install instructions <https://caffe2.ai/docs/getting-started.html>`_.

Once these are installed, you can use the backend for Caffe2::

    # ...continuing from above
    import caffe2.python.onnx.backend as backend
    import numpy as np

    rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # For the Caffe2 backend:
    #     rep.predict_net is the Caffe2 protobuf for the network
    #     rep.workspace is the Caffe2 workspace for the network
    #       (see the class caffe2.python.onnx.backend.Workspace)
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])

You can also run the exported model with `ONNX Runtime <https://github.com/microsoft/onnxruntime>`_,
you will need to install `ONNX Runtime`: please `follow these instructions <https://github.com/microsoft/onnxruntime#installation>`_.

Once these are installed, you can use the backend for ONNX Runtime::

    # ...continuing from above
    import onnxruntime as ort

    ort_session = ort.InferenceSession('alexnet.onnx')

    outputs = ort_session.run(None, {'actual_input_1': np.random.randn(10, 3, 224, 224).astype(np.float32)})

    print(outputs[0])

Here is another `tutorial of exporting the SuperResolution model to ONNX. <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>`_.

In the future, there will be backends for other frameworks as well.

Tracing vs Scripting
--------------------

The ONNX exporter can be both *trace-based* and *script-based* exporter.

* *trace-based* means that it operates by executing your model once, and exporting the operators which
  were actually run during this run.  This means that if your model is
  dynamic, e.g., changes behavior depending on input data, the export
  won't be accurate.  Similarly, a trace is likely to be valid only
  for a specific input size (which is one reason why we require explicit inputs
  on tracing.)  We recommend examining the model trace and making sure
  the traced operators look reasonable.  If your model contains control flows like
  for loops and if conditions, *trace-based* exporter will unroll the loops and if conditions,
  exporting a static graph that is exactly the same as this run.  If you want
  to export your model with dynamic control flows, you will need to use the *script-based* exporter.

* *script-based* means that the model you are trying to export is a `ScriptModule <jit.html>`_.
  `ScriptModule` is the core data structure in `TorchScript`, and `TorchScript` is a subset of Python language,
  that creates serializable and optimizable models from PyTorch code.

We allow mixing tracing and scripting. You can compose tracing and scripting to suit the particular requirements
of a part of a model.  Checkout this example: ::

    import torch

    # Trace-based only

    class LoopModel(torch.nn.Module):
        def forward(self, x, y):
            for i in range(y):
                x = x + i
            return x

    model = LoopModel()
    dummy_input = torch.ones(2, 3, dtype=torch.long)
    loop_count = torch.tensor(5, dtype=torch.long)

    torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True)

With *trace-based* exporter, we get the result ONNX graph which unrolls the for loop: ::

    graph(%0 : Long(2, 3),
          %1 : Long()):
      %2 : Tensor = onnx::Constant[value={1}]()
      %3 : Tensor = onnx::Add(%0, %2)
      %4 : Tensor = onnx::Constant[value={2}]()
      %5 : Tensor = onnx::Add(%3, %4)
      %6 : Tensor = onnx::Constant[value={3}]()
      %7 : Tensor = onnx::Add(%5, %6)
      %8 : Tensor = onnx::Constant[value={4}]()
      %9 : Tensor = onnx::Add(%7, %8)
      return (%9)

To utilize *script-based* exporter for capturing the dynamic loop,
we can write the loop in script, and call it from the regular nn.Module: ::

    # Mixing tracing and scripting

    @torch.jit.script
    def loop(x, y):
        for i in range(int(y)):
            x = x + i
        return x

    class LoopModel2(torch.nn.Module):
        def forward(self, x, y):
            return loop(x, y)

    model = LoopModel2()
    dummy_input = torch.ones(2, 3, dtype=torch.long)
    loop_count = torch.tensor(5, dtype=torch.long)
    torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True,
                      input_names=['input_data', 'loop_range'])

Now the exported ONNX graph becomes: ::

    graph(%input_data : Long(2, 3),
          %loop_range : Long()):
      %2 : Long() = onnx::Constant[value={1}](), scope: LoopModel2/loop
      %3 : Tensor = onnx::Cast[to=9](%2)
      %4 : Long(2, 3) = onnx::Loop(%loop_range, %3, %input_data), scope: LoopModel2/loop # custom_loop.py:240:5
        block0(%i.1 : Long(), %cond : bool, %x.6 : Long(2, 3)):
          %8 : Long(2, 3) = onnx::Add(%x.6, %i.1), scope: LoopModel2/loop # custom_loop.py:241:13
          %9 : Tensor = onnx::Cast[to=9](%2)
          -> (%9, %8)
      return (%4)

The dynamic control flow is captured correctly. We can verify in backends with different loop range. ::

    import caffe2.python.onnx.backend as backend
    import numpy as np
    import onnx
    model = onnx.load('loop.onnx')

    rep = backend.prepare(model)
    outputs = rep.run((dummy_input.numpy(), np.array(9).astype(np.int64)))
    print(outputs[0])
    #[[37 37 37]
    # [37 37 37]]


    import onnxruntime as ort
    ort_sess = ort.InferenceSession('loop.onnx')
    outputs = ort_sess.run(None, {'input_data': dummy_input.numpy(),
                                  'loop_range': np.array(9).astype(np.int64)})
    print(outputs)
    #[array([[37, 37, 37],
    #       [37, 37, 37]], dtype=int64)]


TorchVision support
-------------------

All TorchVision models, except for quantized versions, are exportable to ONNX.
More details can be found in `TorchVision <torchvision/models.html>`_.


Limitations
-----------

* Only tuples, lists and Variables are supported as JIT inputs/outputs. Dictionaries and strings are also accepted
  but their usage is not recommended. Users need to verify their dict inputs carefully, and keep in mind that
  dynamic lookups are not available.

* PyTorch and ONNX backends(Caffe2, ONNX Runtime, etc) often have implementations of operators with some
  numeric differences.  Depending on model structure, these differences
  may be negligible, but they can also cause major divergences in behavior
  (especially on untrained models.)  We allow Caffe2 to call directly to Torch implementations of operators, to
  help you smooth over these differences when precision is important,
  and to also document these differences.

Supported operators
-------------------

The following operators are supported:

* BatchNorm
* ConstantPadNd
* Conv
* Dropout
* Embedding (no optional arguments supported)
* FeatureDropout (training mode not supported)
* Index
* MaxPool1d
* MaxPool2d
* MaxPool3d
* RNN
* abs
* acos
* adaptive_avg_pool1d
* adaptive_avg_pool2d
* adaptive_avg_pool3d
* adaptive_max_pool1d
* adaptive_max_pool2d
* adaptive_max_pool3d
* add (nonzero alpha not supported)
* addmm
* and
* arange
* argmax
* argmin
* asin
* atan
* avg_pool1d
* avg_pool2d
* avg_pool2d
* avg_pool3d
* baddbmm
* bitshift
* cat
* ceil
* clamp
* clamp_max
* clamp_min
* concat
* copy
* cos
* cumsum
* det
* dim_arange
* div
* dropout
* elu
* empty
* empty_like
* eq
* erf
* exp
* expand
* expand_as
* flatten
* floor
* floor_divide
* frobenius_norm
* full
* full_like
* gather
* ge
* gelu
* glu
* group_norm
* gt
* hardtanh
* im2col
* index_copy
* index_fill
* index_put
* index_select
* instance_norm
* interpolate
* isnan
* layer_norm
* le
* leaky_relu
* log
* log1p
* log2
* log_sigmoid
* log_softmax
* logdet
* logsumexp
* lt
* masked_fill
* max
* mean
* min
* mm
* mul
* multinomial
* narrow
* ne
* neg
* nonzero
* norm
* ones
* ones_like
* or
* permute
* pixel_shuffle
* pow
* prelu (single weight shared among input channels not supported)
* prod
* rand
* randn
* randn_like
* reciprocal
* reflection_pad
* relu
* repeat
* replication_pad
* reshape
* reshape_as
* round
* rrelu
* rsqrt
* rsub
* scalar_tensor
* scatter
* scatter_add
* select
* selu
* sigmoid
* sign
* sin
* size
* slice
* softmax
* softplus
* sort
* split
* sqrt
* squeeze
* stack
* std
* sub (nonzero alpha not supported)
* sum
* t
* tan
* tanh
* threshold (non-zero threshold/non-zero value not supported)
* to
* topk
* transpose
* true_divide
* type_as
* unbind
* unfold (experimental support with ATen-Caffe2 integration)
* unique
* unsqueeze
* upsample_nearest1d
* upsample_nearest2d
* upsample_nearest3d
* view
* weight_norm
* where
* zeros
* zeros_like

The operator set above is sufficient to export the following models:

* AlexNet
* DCGAN
* DenseNet
* Inception (warning: this model is highly sensitive to changes in operator
  implementation)
* ResNet
* SuperResolution
* VGG
* `word_language_model <https://github.com/pytorch/examples/tree/master/word_language_model>`_

Adding support for operators
----------------------------

Adding export support for operators is an *advance usage*.

To achieve this, developers need to touch the source code of PyTorch.
Please follow the `instructions <https://github.com/pytorch/pytorch#from-source>`_
for installing PyTorch from source.
If the wanted operator is standardized in ONNX, it should be easy to add
support for exporting such operator (adding a symbolic function for the operator).
To confirm whether the operator is standardized or not, please check the
`ONNX operator list <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_.

ATen operators
~~~~~~~~~~~~~~

If the operator is an ATen operator, which means you can find the declaration
of the function in ``torch/csrc/autograd/generated/VariableType.h``
(available in generated code in PyTorch install dir), you should add the symbolic
function in ``torch/onnx/symbolic_opset<version>.py`` and follow the instructions listed as below:

* Define the symbolic function in ``torch/onnx/symbolic_opset<version>.py``, for example
  `torch/onnx/symbolic_opset9.py <https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py>`_.
  Make sure the function has the same name as the ATen operator/function
  defined in ``VariableType.h``.
* The first parameter is always the exported ONNX graph.
  Parameter names must EXACTLY match the names in ``VariableType.h``,
  because dispatch is done with keyword arguments.
* Parameter ordering does NOT necessarily match what is in ``VariableType.h``,
  tensors (inputs) are always first, then non-tensor arguments.
* In the symbolic function, if the operator is already standardized in ONNX,
  we only need to create a node to represent the ONNX operator in the graph.
* If the input argument is a tensor, but ONNX asks for a scalar, we have to
  explicitly do the conversion. The helper function ``_scalar`` can convert a
  scalar tensor into a python scalar, and ``_if_scalar_type_as`` can turn a
  Python scalar into a PyTorch tensor.

Non-ATen operators
~~~~~~~~~~~~~~~~~~

If the operator is a non-ATen operator, the symbolic function has to be
added in the corresponding PyTorch Function class. Please read the following
instructions:

* Create a symbolic function named ``symbolic`` in the corresponding Function class.
* The first parameter is always the exported ONNX graph.
* Parameter names except the first must EXACTLY match the names in ``forward``.
* The output tuple size must match the outputs of ``forward``.
* In the symbolic function, if the operator is already standardized in ONNX,
  we just need to create a node to represent the ONNX operator in the graph.

Symbolic functions should be implemented in Python. All of these functions interact
with Python methods which are implemented via C++-Python bindings,
but intuitively the interface they provide looks like this::


    def operator/symbolic(g, *inputs):
      """
      Modifies Graph (e.g., using "op"), adding the ONNX operations representing
      this PyTorch function, and returning a Value or tuple of Values specifying the
      ONNX outputs whose values correspond to the original PyTorch return values
      of the autograd Function (or None if an output is not supported by ONNX).

      Arguments:
        g (Graph): graph to write the ONNX representation into
        inputs (Value...): list of values representing the variables which contain
            the inputs for this function
      """

    class Value(object):
      """Represents an intermediate tensor value computed in ONNX."""
      def type(self):
        """Returns the Type of the value."""

    class Type(object):
      def sizes(self):
        """Returns a tuple of ints representing the shape of a tensor this describes."""

    class Graph(object):
      def op(self, opname, *inputs, **attrs):
        """
        Create an ONNX operator 'opname', taking 'args' as inputs
        and attributes 'kwargs' and add it as a node to the current graph,
        returning the value representing the single output of this
        operator (see the `outputs` keyword argument for multi-return
        nodes).

        The set of operators and the inputs/attributes they take
        is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

        Arguments:
            opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
            args (Value...): The inputs to the operator; usually provided
                as arguments to the `symbolic` definition.
            kwargs: The attributes of the ONNX operator, with keys named
                according to the following convention: `alpha_f` indicates
                the `alpha` attribute with type `f`.  The valid type specifiers are
                `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
                specified with type float accepts either a single float, or a
                list of floats (e.g., you would say `dims_i` for a `dims` attribute
                that takes a list of integers).
            outputs (int, optional):  The number of outputs this operator returns;
                by default an operator is assumed to return a single output.
                If `outputs` is greater than one, this functions returns a tuple
                of output `Value`, representing each output of the ONNX operator
                in positional.
        """

The ONNX graph C++ definition is in ``torch/csrc/jit/ir/ir.h``.

Here is an example of handling missing symbolic function for ``elu`` operator.
We try to export the model and see the error message as below::

    UserWarning: ONNX export failed on elu because torch.onnx.symbolic_opset9.elu does not exist
    RuntimeError: ONNX export failed: Couldn't export operator elu

The export fails because PyTorch does not support exporting ``elu`` operator.
We find ``virtual Tensor elu(const Tensor & input, Scalar alpha, bool inplace) const override;``
in ``VariableType.h``. This means ``elu`` is an ATen operator.
We check the `ONNX operator list <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_,
and confirm that ``Elu`` is standardized in ONNX.
We add the following lines to ``symbolic_opset9.py``::

    def elu(g, input, alpha, inplace=False):
        return g.op("Elu", input, alpha_f=_scalar(alpha))

Now PyTorch is able to export ``elu`` operator.

There are more examples in
`symbolic_opset9.py <https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py>`_,
`symbolic_opset10.py <https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset10.py>`_.


The interface for specifying operator definitions is experimental;
adventurous users should note that the APIs will probably
change in a future interface.

Custom operators
~~~~~~~~~~~~~~~~

Following this tutorial `Extending TorchScript with Custom C++ Operators <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_,
you can create and register your own custom ops implementation in PyTorch. Here's how to export such model to ONNX.::

    # Create custom symbolic function
    from torch.onnx.symbolic_helper import parse_args
    @parse_args('v', 'v', 'f', 'i')
    def symbolic_foo_forward(g, input1, input2, attr1, attr2):
        return g.op("Foo", input1, input2, attr1_f=attr1, attr2_i=attr2)

    # Register custom symbolic function
    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('custom_ops::foo_forward', symbolic_foo_forward, 9)

    class FooModel(torch.nn.Module):
        def __init__(self, attr1, attr2):
            super(FooModule, self).__init__()
            self.attr1 = attr1
            self.attr2 = attr2

        def forward(self, input1, input2):
            # Calling custom op
            return torch.ops.custom_ops.foo_forward(input1, input2, self.attr1, self.attr2)

    model = FooModel(attr1, attr2)
    torch.onnx.export(model, (dummy_input1, dummy_input2), 'model.onnx')

Depending on the custom operator, you can export it as one or a combination of existing ONNX ops.
You can also export it as a custom op in ONNX as well. In that case, you will need to extend the backend of your choice
with matching custom ops implementation, e.g. `Caffe2 custom ops <https://caffe2.ai/docs/custom-operators.html>`_,
`ONNX Runtime custom ops <https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md>`_.

Frequently Asked Questions
--------------------------
Q: I have exported my lstm model, but its input size seems to be fixed?

  The tracer records the example inputs shape in the graph. In case the model should accept
  inputs of dynamic shape, you can utilize the parameter `dynamic_axes` in export api. ::

    layer_count = 4

    model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=True)
    model.eval()

    with torch.no_grad():
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(layer_count * 2, 3, 20)
        c0 = torch.randn(layer_count * 2, 3, 20)
        output, (hn, cn) = model(input, (h0, c0))

        # default export
        torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
        onnx_model = onnx.load('lstm.onnx')
        # input shape [5, 3, 10]
        print(onnx_model.graph.input[0])

        # export with `dynamic_axes`
        torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx',
                        input_names=['input', 'h0', 'c0'],
                        output_names=['output', 'hn', 'cn'],
                        dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}})
        onnx_model = onnx.load('lstm.onnx')
        # input shape ['sequence', 3, 10]
        print(onnx_model.graph.input[0])


Q: How to export models with loops in it?

  Please checkout `Tracing vs Scripting`_.

Q: Does ONNX support implicit scalar datatype casting?

  No, but the exporter will try to handle that part.  Scalars are converted to constant tensors in ONNX.
  The exporter will try to figure out the right datatype for scalars.  However for cases that it failed
  to do so, you will need to manually provide the datatype information.  This often happens with scripted models,
  where the datatypes are not recorded.  We are trying to improve the datatype
  propagation in the exporter such that manual changes are not required in the future. ::

    class ImplicitCastType(torch.jit.ScriptModule):
        @torch.jit.script_method
        def forward(self, x):
            # Exporter knows x is float32, will export '2' as float32 as well.
            y = x + 2
            # Without type propagation, exporter doesn't know the datatype of y.
            # Thus '3' is exported as int64 by default.
            return y + 3
            # The following will export correctly.
            # return y + torch.tensor([3], dtype=torch.float32)

    x = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(ImplicitCastType(), x, 'models/implicit_cast.onnx',
                      example_outputs=ImplicitCastType()(x))

Q: Is tensor in-place indexed assignment like `data[index] = new_data` supported?

  Yes, this is supported now for ONNX opset version >= 11. E.g.: ::

    data = torch.zeros(3, 4)
    new_data = torch.arange(4).to(torch.float32)

    # Assigning to left hand side indexing is supported in ONNX opset >= 11.
    class InPlaceIndexedAssignment(torch.nn.Module):
        def forward(self, data, new_data):
            data[1] = new_data
            return data

    out = InPlaceIndexedAssignment()(data, new_data)

    data = torch.zeros(3, 4)
    new_data = torch.arange(4).to(torch.float32)
    torch.onnx.export(InPlaceIndexedAssignment(), (data, new_data), 'inplace_assign.onnx', opset_version=11)

    # onnxruntime
    import onnxruntime
    sess = onnxruntime.InferenceSession('inplace_assign.onnx')
    out_ort = sess.run(None, {
        sess.get_inputs()[0].name: torch.zeros(3, 4).numpy(),
        sess.get_inputs()[1].name: new_data.numpy(),
    })

    assert torch.all(torch.eq(out, torch.tensor(out_ort)))

Q: Is tensor list exportable to ONNX?

  Yes, this is supported now for ONNX opset version >= 11. ONNX introduced the concept of Sequence in opset 11.
  Similar to list, Sequence is a data type that contains arbitrary number of Tensors.
  Associated operators are also introduced in ONNX, such as SequenceInsert, SequenceAt, etc. E.g.: ::

    class ListLoopModel(torch.nn.Module):
        def forward(self, x):
            res = []
            res1 = []
            arr = x.split(2, 0)
            res2 = torch.zeros(3, 4, dtype=torch.long)
            for i in range(len(arr)):
                res = res.append(arr[i].sum(0, False))
                res1 = res1.append(arr[-1 - i].sum(0, False))
                res2 += 1
            return torch.stack(res), torch.stack(res1), res2

    model = torch.jit.script(ListLoopModel())
    inputs = torch.randn(16)

    out = model(inputs)
    torch.onnx.export(model, (inputs, ), 'loop_and_list.onnx', opset_version=11, example_outputs=out)

    # onnxruntime
    import onnxruntime
    sess = onnxruntime.InferenceSession('loop_and_list.onnx')
    out_ort = sess.run(None, {
        sess.get_inputs()[0].name: inputs.numpy(),
    })

    assert [torch.allclose(o, torch.tensor(o_ort)) for o, o_ort in zip(out, out_ort)]

Functions
--------------------------
.. autofunction:: export
.. autofunction:: register_custom_op_symbolic
.. autofunction:: torch.onnx.operators.shape_as_tensor
.. autofunction:: set_training
.. autofunction:: is_in_onnx_export
