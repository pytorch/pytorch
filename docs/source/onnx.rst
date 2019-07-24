torch.onnx
============
.. automodule:: torch.onnx

Example: End-to-end AlexNet from PyTorch to Caffe2
--------------------------------------------------

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

You can also verify the protobuf using the `onnx <https://github.com/onnx/onnx/>`_ library.
You can install ``onnx`` with conda::

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

In the future, there will be backends for other frameworks as well.

Limitations
-----------

* The ONNX exporter is a *trace-based* exporter, which means that it
  operates by executing your model once, and exporting the operators which
  were actually run during this run.  This means that if your model is
  dynamic, e.g., changes behavior depending on input data, the export
  won't be accurate.  Similarly, a trace is likely to be valid only
  for a specific input size (which is one reason why we require explicit inputs
  on tracing.)  We recommend examining the model trace and making sure
  the traced operators look reasonable.

* PyTorch and Caffe2 often have implementations of operators with some
  numeric differences.  Depending on model structure, these differences
  may be negligible, but they can also cause major divergences in behavior
  (especially on untrained models.)  In a future release, we plan to
  allow Caffe2 to call directly to Torch implementations of operators, to
  help you smooth over these differences when precision is important,
  and to also document these differences.

Supported operators
-------------------

The following operators are supported:

* add (nonzero alpha not supported)
* sub (nonzero alpha not supported)
* mul
* div
* cat
* mm
* addmm
* neg
* sqrt
* tanh
* sigmoid
* mean
* sum
* prod
* t
* expand (only when used before a broadcasting ONNX operator; e.g., add)
* transpose
* view
* split
* squeeze
* prelu (single weight shared among input channels not supported)
* threshold (non-zero threshold/non-zero value not supported)
* leaky_relu
* glu
* softmax (only dim=-1 supported)
* avg_pool2d (ceil_mode not supported)
* log_softmax
* unfold (experimental support with ATen-Caffe2 integration)
* elu
* concat
* abs
* index_select
* pow
* clamp
* max
* min
* eq
* gt
* lt
* ge
* le
* exp
* sin
* cos
* tan
* asin
* acos
* atan
* permute
* Conv
* BatchNorm
* MaxPool1d (ceil_mode not supported)
* MaxPool2d (ceil_mode not supported)
* MaxPool3d (ceil_mode not supported)
* Embedding (no optional arguments supported)
* RNN
* ConstantPadNd
* Dropout
* FeatureDropout (training mode not supported)
* Index (constant integer and tuple indices supported)

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

Adding export support for operators is an *advance usage*.
To achieve this, developers need to touch the source code of PyTorch.
Please follow the `instructions <https://github.com/pytorch/pytorch#from-source>`_
for installing PyTorch from source.
If the wanted operator is standardized in ONNX, it should be easy to add
support for exporting such operator (adding a symbolic function for the operator).
To confirm whether the operator is standardized or not, please check the
`ONNX operator list <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_.

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

The ONNX graph C++ definition is in ``torch/csrc/jit/ir.h``.

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
`tensor.py <https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/autograd/_functions/tensor.py#L24>`_,
`padding.py <https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/nn/_functions/padding.py#L8>`_.


The interface for specifying operator definitions is experimental;
adventurous users should note that the APIs will probably
change in a future interface.

Functions
--------------------------
.. autofunction:: export
.. autofunction:: set_training
.. autofunction:: is_in_onnx_export
