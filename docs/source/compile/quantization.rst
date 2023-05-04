(prototype) Quantization with PT2
=================================

This document will discuss the extension of Quantization-Aware Training
(QAT) with PT2. To learn about quantization in general, please visit
this
`link <https://pytorch.org/docs/stable/quantization.html#module-torch.ao.quantization>`__.

Today we have `FX Graph Mode
Quantization <https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization>`__
which uses symbolic_trace to capture the model into a graph, and then
perform quantization transformations on top of the captured model. In a
similar way, we will now use the PT2 Export workflow to capture the
model into a graph, and perform quantizations transformations on top of
the ATen dialect graph. This is expected to have significantly higher
model coverage, better programmability, and a simplified UX.

UX
--

The user flow mirrors the
`PTQ <https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization>`__
UX closely:

.. code:: python

   import torch
   from torch.ao.quantization._quantize_pt2e import convert_pt2e, prepare_pt2e
   import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

   class M(torch.nn.Module):
      def __init__(self):
         super().__init__()
         self.linear = torch.nn.Linear(5, 10)

      def forward(self, x):
         return self.linear(x)


   example_inputs = (torch.randn(1, 5),)
   model = M().eval()

   # Step 1: Trace the model into an FX graph of flattened ATen operators
   exported_graph_module = torch.export(model, example_inputs, ...)

   # Step 2: Insert observers or fake quantize modules
   quantizer = qq.QNNPackQuantizer()
   operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
   quantizer.set_global(operator_config)
   prepared_graph_module = prepare_pt2e(exported_graph_module, quantizer)

   # Step 4: Quantize the model
   convered_graph_module = convert_pt2e(prepared_graph_module)

   # Proceed with rest of PT2 stack
   torch.compile(converted_graph_module, ...)

APIs
----

There are two top level APIs:

prepare_pt2e
~~~~~~~~~~~~

``prepare_pt2e`` prepares the given FX GraphModule for quantization by
inserting observers or fake quantize modules depending on the
specifications of the given quantizer.

Parameters: \* ``model: torch.fx.GraphModule``: an FX GraphModule
captured by PT2 export \* ``quantizer: Quantizer``: A configuration that
specifies what nodes to quantize in the graph. This can either be
provided by backend developers, or users can extend the ``Quantizer``
class to configure their quantization needs

convert_to_pt2e
~~~~~~~~~~~~~~~

``convert_pt2e`` quantizes the FX GraphModule created from the previous
``prepare_pt2e`` call.

Parameters: \* ``model: torch.fx.GraphModule``: an FX GraphModule
produced from a call to ``prepare_pt2e``

Quantizer
~~~~~~~~~

``Quantizer`` is a class that allows users to programmably set up the
observers or fake quant objects for each node in the given GraphModule.
There are two options for users to choose how to get a quantizer.

1. Users can choose a quantizer provided by a backend. Users can find
   out the quantization capabilities of the quantizer by calling
   ``quantizer.get_supported_operators()`` and can configure the
   quantization by calling APIs provided by the backend.

.. code:: python

   from torch.ao.quantization._quantize_pt2e import convert_pt2e, prepare_pt2e
   import torch.ao.quantization._pt2e.quantizer.qnnpack_quantizer as qq

   quantizer = qq.QNNPackQuantizer()
   operator_config = qq.get_symmetric_quantization_config(is_per_channel=True)
   quantizer.set_global(operator_config)
   prepared_graph_module = prepare_pt2e(exported_graph_module, quantizer)

2. Users can programmably set up the configuration by subclassing
   ```Quantizer`` <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_pt2e/quantizer/quantizer.py#L102>`__.
   An example quantizer implementation for the QNNPack Backend is
   `here <https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_pt2e/quantizer/qnnpack_quantizer.py>`__.

Note:

::

   We are still figuring out what exact pattern matching we want to use and
   what pattern matching utilities to offer.

.. code:: python

   from torch.ao.quantization._pte2.quantizer.Quantizer

   class CustomQuantizer(Quantizer):
       def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
           """
           Annotate nodes in the graph with observer or fake quant constructs to
           convey the desired way of quantization.
           """
           ...

       def validate(self, model: torch.fx.GraphModule) -> None:
           """
           Validate the annotated graph is supported by the backend
           """
           ...

       # annotate nodes in the graph with observer or fake quant constructors
       # to convey the desired way of quantization
       def get_supported_operators(cls) -> List[OperatorConfig]:
           """
           Return a list of ATen operators, `torch.nn.Module` types, or
           `torch.nn.functional` operators.
           """
           ...

   quantizer = CustomQuantizer()
   m = prepare_pt2e(m, quantizer)
