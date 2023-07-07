Getting Started
===============

Let’s start with a simple example. Note that you are likely to see more
significant speedups the newer your GPU is.

The below is a tutorial for inference, for a training specific tutorial, make sure to checkout `example on training <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__

.. code:: python

   import torch
   def fn(x, y):
       a = torch.cos(x).cuda()
       b = torch.sin(y).cuda()
       return a + b
   new_fn = torch.compile(fn, backend="inductor")
   input_tensor = torch.randn(10000).to(device="cuda:0")
   a = new_fn(input_tensor, input_tensor)

This example will not actually run faster. Its purpose is to demonstrate
the ``torch.cos()`` and ``torch.sin()`` features which are
examples of pointwise ops as in they operate element by element on a
vector. A more famous pointwise op you might want to use would
be something like ``torch.relu()``. Pointwise ops in eager mode are
suboptimal because each one would need to read a tensor from
memory, make some changes, and then write back those changes. The single
most important optimization that inductor does is fusion. So back to our
example we can turn 2 reads and 2 writes into 1 read and 1 write which
is crucial especially for newer GPUs where the bottleneck is memory
bandwidth (how quickly you can send data to a GPU) rather than compute
(how quickly your GPU can crunch floating point operations).

Another major optimization that inductor makes available is automatic
support for CUDA graphs.
CUDA graphs help eliminate the overhead from launching individual
kernels from a Python program which is especially relevant for newer GPUs.

TorchDynamo supports many different backends but inductor specifically works
by generating `Triton <https://github.com/openai/triton>`__ kernels. Suppose our example above
was called ``trig.py`` we can inspect the code generated triton kernels by
running ``TORCH_COMPILE_DEBUG=1 python trig.py`` with the actual generated kernel being

.. code-block:: python

   @pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
   @triton.jit
   def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
       xnumel = 10000
       xoffset = tl.program_id(0) * XBLOCK
       xindex = xoffset + tl.arange(0, XBLOCK)[:]
       xmask = xindex < xnumel
       x0 = xindex
       tmp0 = tl.load(in_ptr0 + (x0), xmask)
       tmp1 = tl.cos(tmp0)
       tmp2 = tl.sin(tmp0)
       tmp3 = tmp1 + tmp2
       tl.store(out_ptr0 + (x0), tmp3, xmask)

And you can verify that fusing the ``cos`` and ``sin`` did actually occur
because the ``cos`` and ``sin`` operations occur within a single Triton kernel
and the temporary variables are held in registers with very fast access.

You can read up a lot more on Triton’s performance
`here <https://openai.com/blog/triton/>`__ but the key is it’s in Python
so you can easily understand it even if you have not written all that
many CUDA kernels.

Next, let’s try a real model like resnet50 from the PyTorch
hub.

.. code-block:: python

   import torch
   model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
   opt_model = torch.compile(model, backend="inductor")
   model(torch.randn(1,3,64,64))

And that is not the only available backend, you can run in a REPL
``torch._dynamo.list_backends()`` to see all the available backends. Try out the
``cudagraphs`` or ``nvfuser`` next as inspiration.

Let’s do something a bit more interesting now, our community frequently
uses pretrained models from
`transformers <https://github.com/huggingface/transformers>`__ or
`TIMM <https://github.com/rwightman/pytorch-image-models>`__ and one of
our design goals is for Dynamo and inductor to work out of the box with
any model that people would like to author.

So we will directly download a pretrained model from the
HuggingFace hub and optimize it:

.. code-block:: python

   import torch
   from transformers import BertTokenizer, BertModel
   # Copy pasted from here https://huggingface.co/bert-base-uncased
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
   model = torch.compile(model, backend="inductor") # This is the only line of code that we changed
   text = "Replace me by any text you'd like."
   encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
   output = model(**encoded_input)

If you remove the ``to(device="cuda:0")`` from the model and
``encoded_input``, then Triton will generate C++ kernels that will be
optimized for running on your CPU. You can inspect both Triton or C++
kernels for BERT, they’re obviously more complex than the trigonometry
example we had above but you can similarly skim it and understand if you
understand PyTorch.

Similarly let’s try out a TIMM example

.. code-block:: python

   import timm
   import torch._dynamo as dynamo
   import torch
   model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
   opt_model = torch.compile(model, backend="inductor")
   opt_model(torch.randn(64,3,7,7))

Our goal with Dynamo and inductor is to build the highest coverage ML compiler
which should work with any model you throw at it.

Existing Backends
~~~~~~~~~~~~~~~~~

TorchDynamo has a growing list of backends, which can be found in the
`backends <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/>`__ folder
or ``torch._dynamo.list_backends()`` each of which with its optional dependencies.

Some of the most commonly used backends include:

**Training & inference backends**:
  * ``torch.compile(m, backend="inductor")`` - Uses ``TorchInductor`` backend. `Read more <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
  * ``torch.compile(m, backend="aot_ts_nvfuser")`` - nvFuser with AotAutograd/TorchScript. `Read more <https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593>`__
  * ``torch.compile(m, backend="nvprims_nvfuser")`` - nvFuser with PrimTorch. `Read more <https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593>`__
  * ``torch.compile(m, backend="cudagraphs")`` - cudagraphs with AotAutograd. `Read more <https://github.com/pytorch/torchdynamo/pull/757>`__

**Inference-only backends**:
  * ``torch.compile(m, backend="onnxrt")`` - Uses ONNXRT for inference on CPU/GPU. `Read more <https://onnxruntime.ai/>`__
  * ``torch.compile(m, backend="tensorrt")`` - Uses ONNXRT to run TensorRT for inference optimizations. `Read more <https://github.com/onnx/onnx-tensorrt>`__
  * ``torch.compile(m, backend="ipex")`` - Uses IPEX for inference on CPU. `Read more <https://github.com/intel/intel-extension-for-pytorch>`__
  * ``torch.compile(m, backend="tvm")`` - Uses Apache TVM for inference optimizations. `Read more <https://tvm.apache.org/>`__

Why do you need another way of optimizing PyTorch code?
-------------------------------------------------------

While a number of other code optimization tools exist in the PyTorch
ecosystem, each of them has its own flow.
Here is a few examples of existing methods and their limitations:

-  ``torch.jit.trace()`` is silently wrong if it cannot trace, for example:
   during control flow
-  ``torch.jit.script()`` requires modifications to user or library code
   by adding type annotations and removing non PyTorch code
-  ``torch.fx.symbolic_trace()`` either traces correctly or gives a hard
   error but it’s limited to traceable code so still can’t handle
   control flow
-  ``torch._dynamo`` works out of the box and produces partial graphs.
   It still has the option of producing a single graph with
   ``nopython=True`` which are needed for `some
   situations <./documentation/FAQ.md#do-i-still-need-to-export-whole-graphs>`__
   but allows a smoother transition where partial graphs can be
   optimized without code modification
