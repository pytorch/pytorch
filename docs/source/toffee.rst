torch.toffee
============
.. automodule:: torch.toffee

Example: End-to-end AlexNet from PyTorch to Caffe2
--------------------------------------------------

Here is a simple script which exports a pretrained AlexNet as defined in
torchvision into Toffee IR.  It runs a single round of inference and then
saves the resulting traced model to ``alexnet.proto``::

    from torch.autograd import Variable
    import torch.toffee
    import torchvision

    dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
    model = torchvision.models.alexnet(pretrained=True).cuda()
    torch.toffee.export(model, dummy_input, "alexnet.proto", verbose=True)

The resulting ``alexnet.proto`` is a binary protobuf file which contains both
the network structure and parameters of the model you exported
(in this case, AlexNet).  The keyword argument ``verbose=True`` causes the
exporter to print out a human-readable representation of the network::

    # All parameters are encoded explicitly as inputs.  By convention,
    # learned parameters (ala nn.Module.state_dict) are first, and the
    # actual inputs are last.
    graph(%1 : Float(64, 3, 11, 11)
          %2 : Float(64)
          # The definition sites of all variables are annotated with type
          # information, specifying the type and size of tensors.
          # For example, %3 is a 192 x 64 x 5 x 5 tensor of floats.
          %3 : Float(192, 64, 5, 5)
          %4 : Float(192)
          # ---- omitted for brevity ----
          %15 : Float(1000, 4096)
          %16 : Float(1000)
          %17 : Float(10, 3, 224, 224)) { # the actual input!
      # Every statement consists of some output tensors (and their types),
      # the operator to be run (with its attributes, e.g., kernels, strides,
      # etc.), its input tensors (%17, %1)
      %19 : UNKNOWN_TYPE = Conv[kernels=[11, 11], strides=[4, 4], pads=[2, 2, 2, 2], dilations=[1, 1], group=1](%17, %1), uses = [[%20.i0]];
      # UNKNOWN_TYPE: sometimes type information is not known.  We hope to eliminate
      # all such cases in a later release.
      %20 : Float(10, 64, 55, 55) = Add[broadcast=1, axis=1](%19, %2), uses = [%21.i0];
      %21 : Float(10, 64, 55, 55) = Relu(%20), uses = [%22.i0];
      %22 : Float(10, 64, 27, 27) = MaxPool[kernels=[3, 3], pads=[0, 0, 0, 0], dilations=[1, 1], strides=[2, 2]](%21), uses = [%23.i0];
      # ...
      # Finally, a network returns some tensors
      return (%58);
    }

You can also verify and inspect the actual (substantially more verbose) protobuf
using the `ToffeeIR <https://github.com/ProjectToffee/ToffeeIR/>`_ library::

    import toffee

    graph = toffee.load("alexnet.proto")

    # Check that the IR is well formed
    toffee.checker.check_graph(graph)

    # Print the IR
    print(str(graph))

To run the exported script with Caffe2, you will need to install
`caffe2 <https://caffe2.ai/>`_.  Once these are installed, you can use
the backend for Caffe2::

    # ...continuing from above
    import toffee.backend.c2 as backend
    import numpy as np

    (caffe2_proto, caffe2_workspace) = backend.prepare(graph, device="CUDA:0") # or "CPU"
    outputs = backend.run((caffe2_proto, caffe2_workspace), np.rand(10, 3, 224, 224))
    print(outputs[0])

In the future, there will be backends for other frameworks as well.

Limitations
-----------

* The Toffee exporter is a *trace-based* exporter, which means that it
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

In this tech preview, only the following operators are supported:

* Add (inplace is discarded)
* Mul (inplace is discarded)
* Negate (inplace is discarded)
* Addmm (inplace is discarded, alpha and beta must be 1)
* Tanh (inplace is discarded)
* Sigmoid (inplace is discarded)
* Transpose
* View
* Permute
* Concat
* Squeeze (inplace is discarded)
* BatchNorm
* Convolution
* Dropout (inplace is discarded)
* Relu (inplace is discarded)
* LeakyRelu (inplace is discarded)
* MaxPool1d (ceil_mode must be False)
* MaxPool2d (ceil_mode must be False
* AvgPool2d (ceil_mode must be False)

We plan on expanding support to more operators; RNNs are high on our priority
list.  The operator set above is sufficient to export the following models:

* AlexNet
* DCGAN
* DenseNet
* Inception (warning: this model is highly sensitive to changes in operator
  implementation)
* ResNet
* SqueezeNet
* SuperResolution
* VGG

The interface for specifying operator definitions is highly experimental
and undocumented; adventurous users should note that the APIs will probably
change in a future interface.

Functions
--------------------------
.. autofunction:: export
