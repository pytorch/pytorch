.. _dynamo-doc:

Dynamo
============

.. automodule:: torch._dynamo
.. automodule:: torch._inductor

.. warning ::
     TorchDynamo is experimental and under active development. You are welcome to try it out and contribute, but should expect to find bugs and rough edges.

Introduction to TorchDynamo and Inductor
----------------------------------------


   TorchDynamo makes it easy to experiment with different compiler
   backends to make PyTorch code faster with a single line decorator
   ``torch._dynamo.optimize()``

It works either directly over an ``nn.Module`` as a drop-in replacement
for ``torch.jit.script()`` but it can also work over functions and pull
out the relevant PyTorch operations for you. TorchDynamo supports
arbitrary PyTorch code, control flow, mutation and comes with
experimental support for dynamic shapes.

You can follow our nightly benchmarks
`here <https://github.com/pytorch/torchdynamo/issues/681>`__

TorchDynamo is a Python-level JIT compiler designed to make unmodified
PyTorch programs faster. TorchDynamo hooks into the frame evaluation API
in CPython (`PEP 523 <https://peps.python.org/pep-0523/>`__) to
dynamically modify Python bytecode right before it is executed. It
rewrites Python bytecode in order to extract sequences of PyTorch
operations into an `FX
Graph <https://pytorch.org/docs/stable/fx.html>`__ which is then
just-in-time compiled with a customizable backend. It creates this FX
Graph through bytecode analysis and is designed to mix Python execution
with compiled backends to get the best of both worlds: usability and
performance.

|image0|

For more on TorchDynamo you can read our `posts on PyTorch
dev-discuss <https://dev-discuss.pytorch.org/search?q=TorchDynamo%20order%3Alatest>`__
or `watch a deep-dive
video <https://www.youtube.com/watch?v=egZB5Uxki0I>`__.

For more information on TorchInductor, one of the backends supported by TorchDynamo
Graph <https://pytorch.org/docs/stable/fx.html>`__ into
`Triton <https://github.com/openai/triton>`__ for GPUs or
`C++/OpenMP <https://www.openmp.org/>`__ for CPUs. We have a `training
performance
dashboard <https://github.com/pytorch/torchdynamo/issues/681#issuecomment-1233828468>`__
comparing the performance of different training backends. You can read
more in the `TorchInductor post on PyTorch
dev-discuss <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__.



Requirements and Setup
----------------------

Python 3.8 is recommended. Python 3.7 through 3.10 are supported and
tested. Make sure to have a development version of python installed
locally as well.

TorchDynamo is included in the nightly binaries of PyTorch, for
reference, https://pytorch.org/get-started/locally/

Install GPU/CUDA version requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use GPU back ends (and in particular Triton), please make sure that
the cuda that you have installed locally matches the PyTorch version you
are running.

The following command installs GPU PyTorch+TorchDynamo along with GPU
TorchDynamo dependencies (for CUDA 11.7):

``pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117``

CPU requirements
~~~~~~~~~~~~~~~~

There are no additional requirements for CPU TorchDynamo. CPU
TorchDynamo is included in the nightly versions of PyTorch, which, for
reference, can be installed with

``pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu``

Install from local source
~~~~~~~~~~~~~~~~~~~~~~~~~

Build PyTorch from source:
https://github.com/pytorch/pytorch#from-source, which has TorchDynamo
included.

To install GPU TorchDynamo dependencies, run ``make triton`` in the
PyTorch repo root directory.

Verify Installation
~~~~~~~~~~~~~~~~~~~

If you built PyTorch from source, then you can run the following
commands (from the PyTorch repo root directory) that run minimal
examples to check that TorchDynamo is installed correctly:

.. code:: shell

   cd tools/dynamo
   python verify_dynamo.py

If you do not have the PyTorch source locally, you can alternatively
copy the script (``tools/dynamo/verify_dynamo.py``) from the PyTorch
repo and run it locally.

Docker installation
-------------------

We also provide all the required dependencies in the PyTorch nightly
binaries which you can download with

``docker pull ghcr.io/pytorch/pytorch-nightly``

And for ad hoc experiments just make sure that your container has access
to all your GPUs

``docker run --gpus all -it ghcr.io/pytorch/pytorch-nightly:latest /bin/bash``

Getting started
---------------
Let’s start with a simple example and make things more complicated step
by step. Please note that you’re likely to see more significant speedups
the newer your GPU is.

.. code:: python

   from torch._dynamo import optimize
   import torch


   def fn(x, y):
       a = torch.cos(x).cuda()
       b = torch.sin(y).cuda()
       return a + b

   new_fn = optimize("inductor")(fn)
   input_tensor = torch.randn(10000).to(device="cuda:0")
   a = new_fn()

This example won’t actually run faster but it’s a good educational
example that features ``torch.cos()`` and ``torch.sin()`` which are
examples of pointwise ops as in they operate element by element on a
vector. A more famous pointwise op you might actually want to use would
be something like ``torch.relu()``. Pointwise ops in eager mode are
suboptimal because each one would need to need to read a tensor from
memory, make some changes and then write back those changes. The single
most important optimization that inductor does is fusion. So back to our
example we can turn 2 reads and 2 writes into 1 read and 1 write which
is crucial especially for newer GPUs where the bottleneck is memory
bandwidth (how quickly you can send data to a GPU) instead of compute
(how quickly your GPU can crunch floating point operations)

dynamo supports many different backends but inductor specifically works
by generating `Triton <https://github.com/openai/triton>`__ kernels and
we can inspect them by running ``TORCHINDUCTOR_TRACE=1 python trig.py``
with the actual generated kernel being

.. code:: python

   @pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
   @triton.jit
   def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
       xnumel = 10000
       xoffset = tl.program_id(0) * XBLOCK
       xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
       xmask = xindex < xnumel
       x0 = xindex
       tmp0 = tl.load(in_ptr0 + (x0), xmask)
       tmp1 = tl.sin(tmp0)
       tmp2 = tl.sin(tmp1)
       tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)

And you can verify that fusing the two ``sins`` did actually occur
because the two ``sin`` operations occur within a single Triton kernel
and the temporary variables are held in registers with very fast access.

You can read up a lot more on Triton’s performance
`here <https://openai.com/blog/triton/>`__ but the key is it’s in python
so you can easily understand it even if you haven’t written all that
many CUDA kernels.

As a next step let’s try a real model like resnet50 from the PyTorch
hub.

.. code:: python

   import torch
   import torch._dynamo as dynamo
   model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
   opt_model = dynamo.optimize("inductor")(model)
   model(torch.randn(1,3,64,64))

And that’s not the only available backend, you can run in a REPL
``dynamo.list_backends()`` to see all the available ones. Try out the
``aot_cudagraphs`` or ``nvfuser`` next as inspiration.

Let’s do something a bit more interesting now, our community frequently
uses pretrained models from
`transformers <https://github.com/huggingface/transformers>`__ or
`TIMM <https://github.com/rwightman/pytorch-image-models>`__ and one of
our design goals is for dynamo and inductor to work out of the box with
any model that people would like to author.

So we’re going to directly download a pretrained model from the
HuggingFace hub and optimize it

.. code:: python

   import torch
   from transformers import BertTokenizer, BertModel
   import torch._dynamo as dynamo
   # Copy pasted from here https://huggingface.co/bert-base-uncased
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
   model = dynamo.optimize("inductor")(model) # This is the only line of code that we changed
   text = "Replace me by any text you'd like."
   encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
   output = model(**encoded_input)

If you remove the ``to(device="cuda:0")`` from the model and
encoded_input then triton will generate C++ kernels that will be
optimized for running on your CPU. You can inspect both Triton or C++
kernels for BERT, they’re obviously more complex than the trigonometry
example we had above but you can similarly skim it and understand if you
understand PyTorch.

Similarly let’s try out a TIMM example

.. code:: python

   import timm
   import torch._dynamo as dynamo
   import torch
   model = timm.create_model('resnext101_32x8d', pretrained=True, num_classes=2)
   opt_model = dynamo.optimize("inductor")(model)
   opt_model(torch.randn(64,3,7,7))

Our goal with dynamo and inductor was to build the highest coverage ML compiler which should work with any model you throw at it.

Existing Backends
~~~~~~~~~~~~~~~~~

TorchDynamo has a growing list of backends, which can be found in
`backends.py <https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/backends.py>`__
or ``torchdynamo.list_backends()`` each of which with its optional
dependencies.

Some of the most commonly used backends are

**Debugging backends**: \* ``dynamo.optimize("eager")`` - Uses PyTorch
to run the extracted GraphModule. This is quite useful in debugging
TorchDynamo issues. \* ``dynamo.optimize("aot_eager")`` - Uses
AotAutograd with no compiler, i.e, just using PyTorch eager for the
AotAutograd’s extracted forward and backward graphs. This is useful for
debugging, and unlikely to give speedups.

**Training & inference backends**: \* ``dynamo.optimize("inductor")`` -
Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging
codegened Triton kernels `Read
more <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
\* ``dynamo.optimize("nvfuser")`` - nvFuser with TorchScript. `Read
more <https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593>`__
\* ``dynamo.optimize("aot_nvfuser")`` - nvFuser with AotAutograd. `Read
more <https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593>`__
\* ``dynamo.optimize("aot_cudagraphs")`` - cudagraphs with AotAutograd.
`Read more <https://github.com/pytorch/torchdynamo/pull/757>`__

**Inference-only backend**\ s: \* ``dynamo.optimize("ofi")`` - Uses
Torchscript optimize_for_inference. `Read
more <https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html>`__
\* ``dynamo.optimize("fx2trt")`` - Uses Nvidia TensorRT for inference
optimizations. `Read
more <https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst>`__
\* ``dynamo.optimize("onnxrt")`` - Uses ONNXRT for inference on CPU/GPU.
`Read more <https://onnxruntime.ai/>`__ \* ``dynamo.optimize("ipex")`` -
Uses IPEX for inference on CPU. `Read
more <https://github.com/intel/intel-extension-for-pytorch>`__

Why yet another way of optimizing PyTorch code?
-----------------------------------------------

-  ``torch.jit.trace()`` is silently wrong if it cannot trace e.g:
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


.. |image0| image:: ./_static/source/images/TorchDynamo.png

Custom Backends
===============

Debugging Backend
-----------------

Suppose you wanted to better understand what is going on during a
compilation you can create a custom compiler which we’ll refer to as a
backend that will print pretty print the fx ``GraphModule`` extracted
from dynamo’s bytecode analysis and return a ``forward()`` callable.

.. code:: py

   from typing import List
   import torch
   import torch._dynamo as dynamo

   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       print("my_compiler() called with FX graph:")
       gm.graph.print_tabular()
       return gm.forward  # return a python callable

   @dynamo.optimize(my_compiler)
   def fn(x, y):
       a = torch.cos(x)
       b = torch.sin(y)
       return a + b

   fn(torch.randn(10), torch.randn(10))

Running the above example produces this output

::

   my_compiler() called with FX graph:
   opcode         name    target                                                  args        kwargs
   -------------  ------  ------------------------------------------------------  ----------  --------
   placeholder    x       x                                                       ()          {}
   placeholder    y       y                                                       ()          {}
   call_function  cos     <built-in method cos of type object at 0x7f1a894649a8>  (x,)        {}
   call_function  sin     <built-in method sin of type object at 0x7f1a894649a8>  (y,)        {}
   call_function  add     <built-in function add>                                 (cos, sin)  {}
   output         output  output                                                  ((add,),)   {}

This works for ``torch.nn.Module`` as well as shown below

.. code:: py

   import torch
   import torch._dynamo as dynamo

   class MockModule(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.relu = torch.nn.ReLU()

       def forward(self, x):
           return self.relu(torch.cos(x))

   mod = MockModule()
   optimized_mod = dynamo.optimize(my_compiler)(mod)
   optimized_mod(torch.randn(10))

Let’s take a look at one more example with control flow.

.. code:: py

   from typing import List
   import torch
   import torch._dynamo as dynamo

   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       print("my_compiler() called with FX graph:")
       gm.graph.print_tabular()
       return gm.forward  # return a python callable

   @dynamo.optimize(my_compiler)
   def toy_example(a, b):
       x = a / (torch.abs(a) + 1)
       if b.sum() < 0:
           b = b * -1
       return x * b

   for _ in range(100):
       toy_example(torch.randn(10), torch.randn(10))

Running this example produces the following output:

::

   my_compiler() called with FX graph:
   opcode         name     target                                                  args              kwargs
   -------------  -------  ------------------------------------------------------  ----------------  --------
   placeholder    a        a                                                       ()                {}
   placeholder    b        b                                                       ()                {}
   call_function  abs_1    <built-in method abs of type object at 0x7f8d259298a0>  (a,)              {}
   call_function  add      <built-in function add>                                 (abs_1, 1)        {}
   call_function  truediv  <built-in function truediv>                             (a, add)          {}
   call_method    sum_1    sum                                                     (b,)              {}
   call_function  lt       <built-in function lt>                                  (sum_1, 0)        {}
   output         output   output                                                  ((truediv, lt),)  {}

   my_compiler() called with FX graph:
   opcode         name    target                   args         kwargs
   -------------  ------  -----------------------  -----------  --------
   placeholder    b       b                        ()           {}
   placeholder    x       x                        ()           {}
   call_function  mul     <built-in function mul>  (b, -1)      {}
   call_function  mul_1   <built-in function mul>  (x, mul)     {}
   output         output  output                   ((mul_1,),)  {}

   my_compiler() called with FX graph:
   opcode         name    target                   args       kwargs
   -------------  ------  -----------------------  ---------  --------
   placeholder    b       b                        ()         {}
   placeholder    x       x                        ()         {}
   call_function  mul     <built-in function mul>  (x, b)     {}
   output         output  output                   ((mul,),)  {}

Note that the order of the last two graphs is nondeterministic depending
on which one is encountered first by the just-in-time compiler.

Speedy Backend
--------------

Integrating a custom backend that offers superior performance is also
easy and we’ll integrate a real one
with\ `optimize_for_inference <https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html>`__:

.. code:: py

   def optimize_for_inference_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       scripted = torch.jit.trace(gm, example_inputs)
       return torch.jit.optimize_for_inference(scripted)

And then you should be able to optimize any existing code with

.. code:: py

   @dynamo.optimize(optimize_for_inference_compiler)
   def code_to_accelerate():
       ...

Composable Backends
-------------------

TorchDynamo includes many backends, which can be found in
`backends.py <https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/backends.py>`__
or ``torchdynamo.list_backends()``. You can combine these backends
together with code like:

.. code:: py

   from torch._dynamo.optimizations import BACKENDS

   def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
       trt_compiled = BACKENDS["tensorrt"](gm, example_inputs)
       if trt_compiled is not None:
           return trt_compiled
       # first backend failed, try something else...

       cudagraphs_compiled = BACKENDS["cudagraphs"](gm, example_inputs)
       if cudagraphs_compiled is not None:
           return cudagraphs_compiled

       return gm.forward


Frequently Asked Questions
==========================

At a high level, the TorchDynamo stack consists of a graph capture from
Python code using dynamo and a backend compiler. In this example the
backend compiler consists of backward graph tracing using AOTAutograd
and graph lowering using TorchInductor. There are of course many more
compilers available here
https://github.com/pytorch/torchdynamo/blob/0b8aaf340dad4777a080ef24bf09623f1aa6f3dd/README.md#existing-backend
but for this document we will focus on inductor as a motivating example

Torchdynamo supports training, using AotAutograd to capture backwards:
   1. the ``.forward()`` graph and ``optimizer.step()`` is captured by torchdynamo’s python evalframe frontend
   2. for each segment of ``.forward()`` that torchdynamo captures, it uses AotAutograd to generate a backward graph segment
   3. each pair of forward, backward graph are (optionally) min-cut partitioned to save the minimal state between forward/backward 
   4. the forward, backward pairs are wrapped in autograd.function modules 5. usercode calling\ ``.backward()`` still triggers eager’s autograd engine, which runs each ‘compiled backward’ graph as if it were one op, also running any non-compiled eager ops’ .backward() functions

Do you support Distributed code?
--------------------------------

DDP has been tested and works, support for other distributed training
libraries is under discussion.

The main reason why Distributed code is challenging with dynamo is
because AOTAutograd unrolls both the forward and backward pass and
provides 2 graphs for backends to optimize. This is a problem for
distributed code because we’d like to ideally overlap communication
operations with computations. Eager pytorch accomplishes this in
different ways for DDP/FSDP- using autograd hooks, module hooks, and
modifications/mutations of module states. In a naive application of
dynamo, hooks that should run directly after an operation during
backwards may be delayed until after the entire compiled region of
backwards ops, due to how AOTAutograd compiled functions interact with
dispatcher hooks.

The basic strategy for optimizing DDP with Dynamo is outlined in
https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/distributed.py
where the main idea will be to graph break on `DDP bucket
boundaries <https://pytorch.org/docs/stable/notes/ddp.html#internal-design>`__.

When each node in DDP needs to synchronize its weights with the other
nodes it organizes its gradients and parameters into buckets which
reduces communication times and allows a node to broadcast a fraction of
its gradients to other waiting nodes.

Graph breaks in distributed code means you can expect dynamo and its
backends to optimize the compute overhead of a distributed program but
not its communication overhead. Graph-breaks may interfere with
compilation speedups, if the reduced graph-size robs the compiler of
fusion opportunities. However, there are diminishing returns with
increasing graph size since most of the current compute optimizations
are local fusions. So in practice this approach may be sufficient.

Do I still need to export whole graphs?
---------------------------------------

For the vast majority of models you probably don’t and you can use
``torch._dynamo()`` optimize as is but there are a few situations where
full graphs are necessary and you can can ensure a full graph by simply
running ``torch.dynamo(..., nopython=True)`` \* Large scale training
runs, think $250K+ that require pipeline parallelism and other advanced
sharding strategies \* Inference optimizers like
https://github.com/pytorch/TensorRT or
https://github.com/facebookincubator/AITemplate that rely on fusing much
more aggressively than training optimizers \* Mobile training or
inference

Future work will include tracing communication operations into graphs,
coordinating these operations with compute optimizations, and optimizing
the communciation operations.

Why is my code crashing?
------------------------

If your code ran just fine without dynamo and started to crash with it
enabled then the most important first step is figuring out which part of
the stack your failure occurred in so try running things in the below
order and only try the next step if the previous step succeeded. 
    1.``dynamo.optimize("eager")`` which only runs torchdynamo forward graph
    capture and then runs the captured graph with PyTorch. If this fails
    then there’s an issue with dynamo 
    2. ``dynamo.optimize("aot_eager")``
    which runs torchdynamo to capture a forward graph, and then AOTAutograd
    to trace the backward graph without any additional backend compiler
    steps. PyTorch eager will then be used to run the forward and backward
    graphs. If this fails then there’s an issue with AOTAutograd 
    3. ``dynamo.optimize("inductor")`` which runs torchdynamo to capture a
    forward graph, and then AOTAutograd to trace the backward graph with the
    TorchInductor compiler. If this fails then there’s an issue with TorchInductor

TorchDynamo Errors
~~~~~~~~~~~~~~~~~~

If the error that is generated occurs with the ``"eager"`` backend, then
torchdynamo is the most likely source of the error.

To debug these issues we recommend setting
``torch._dynamo.config.verbose=True`` to get a full stack trace to both
the error in torchdynamo and the user code. In addition to this flag,
you can also set the ``log_level`` of torchdynamo through
``torch._dynamo.config.log_level``. The available levels are the
following: - ``logging.DEBUG``: Print every instruction that is
encountered in addition to all below log levels - ``logging.INFO``:
Print each function that is compiled (original and modified bytecode)
and the graph that is captured in addition to all below log levels -
``logging.WARNING`` (default): Print graph breaks in addition to all
below log levels - ``logging.ERROR``: Print errors only

If a model is sufficiently large, the logs can become overwhelming. If
an error occurs deep within a model’s python code, it can be useful to
execute only the frame in which the error occurs to enable easier
debugging. There are 2 tools available to enable this: 1.
``env TORCHDYNAMO_DEBUG_FUNCTION=<desired_function_name>`` will only run
torchdynamo on functions with that name. 2.
``env torch._dynamo.config.replay_record_enabled = True``) which dumps
an execution record when an error is encountered. This record can then
be replayed to run only the frame where an error occurred.

TorchInductor Errors
--------------------

With TorchInductor as the chosen backend, AOTAutograd is used to
generate the backward graph from the forward graph captured by
torchdynamo. It’s important to note that errors can occur during this
tracing and also while TorchInductor lowers the forward and backward
graphs to GPU code or C++.

A model can often consist of hundreds or thousands of FX nodes, so
narrowing the exact nodes where this problem occurred can be very
difficult which is why we highly recommend you use our minifier to
create tiny reproducible examples of failures you’re seeing. We can
minify errors that occur either at the AOTAutograd layer or Inductor
layer which you should try in the following order.
   1. ``env TORCHDYNAMO_REPRO_AFTER="aot" python your_model.py`` 
   2.  ``env TORCHDYNAMO_REPRO_AFTER="dynamo" python your_model.py``

   Minifying your error is the quickest path to getting it fixed

The minifier will actually create a ``repro.py`` for you at the location
set by ``env TORCHDYNAMO_REPRO_DIR`` so make you have right access to
that directory. You can then run ``python repro.py`` and confirm that
you are getting the same error.

Note: for other compilers such as nvfuser, the process is similar but
instead you would leverage
``env TORCHDYNAMO_REPRO_AFTER="dynamo" python your_model.py``

Why is compilation slow?
------------------------

Dynamo Compilation
~~~~~~~~~~~~~~~~~~

TorchDynamo has a builtin stats function for collecting and displaying
the time spent in each compilation phase. These stats can be accessed by
calling ``torch._dynamo.utils.compile_times()`` after executing
``torch._dynamo``. By default, this returns a string representation of
the compile times spent in each TorchDynamo function by name.

Inductor Compilation
~~~~~~~~~~~~~~~~~~~~

TorchInductor has a builtin stats and trace function for displaying time
spent in each compilation phase, output code, output graph visualization
and IR dump. ``env TORCHINDUCTOR_TRACE=1 python repro.py``. This is a
debugging tool designed to make it easier to debug/understand the
internals of TorchInductor with an output that will look something like
`this <https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396>`__

Each file in that debug trace can be enabled/disabled via
``torch._inductor.config.trace.*``. The profile and the diagram are both
disabled by default since they are expensive to generate. See the
`example debug directory
output <https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396>`__
for more examples.

Excessive Recompilation
~~~~~~~~~~~~~~~~~~~~~~~

When TorchDynamo compiles a function (or part of one), it makes certain
assumptions about locals and globals in order to allow compiler
optimizations, and expresses these assumptions as guards that check
particular values at runtime. If any of these guards fail, Dynamo will
recompile that function (or part) up to
``torch._dynamo.config.cache_size_limit`` times. If your program is
hitting the cache limit, you will first need to determine which guard is
failing and what part of your program is triggering it.

The `recompilation profiler <#recompilation-profiler>`__ automates the
process of setting TorchDynamo’s cache limit to 1 and running your
program under an observation-only ‘compiler’ that records the causes of
any guard failures. You should be sure to run your program for at least
as long (as many iterations) as you were running when you ran into
trouble, and the profiler will accumulate statistics over this duration.

.. code:: py

   prof = dynamo.utils.CompilationProfiler()

   @dynamo.optimize(prof)
   def my_model():
       ...

   my_model()
   print(prof.report())

Many of the reasons for graph breaks and excessive recompilation will be
fixed with upcoming support for `tracing dynamic tensor
shapes <https://docs.google.com/document/d/1QJB-GOnbv-9PygGlOMXwiO9K6vVNm8sNg_olixJ9koc/edit?usp=sharing>`__,
more careful choices for guards and better tuned heuristics.

Why are you recompiling in production?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, you may not want unexpected compiles after a program has
warmed up. For example, if you are serving production traffic in a
latency critical application. For this, TorchDynamo provides an
alternate mode where prior compiled graphs are used, but no new ones are
generated:

.. code:: py

   frozen_toy_example = dynamo.run(toy_example)
   frozen_toy_example(torch.randn(10), torch.randn(10))

How are you speeding up my code?
--------------------------------

There are 3 major ways that PyTorch code can be accelerated 

   1. Kernel fusion via vertical fusions which fuse sequential operations to avoid
   excessive read/writes. For example, fuse 2 subsequent cosines means you
   can can do 1 read 1 write instead 2 reads 2 writes 2. Horizontal fusion:
   the simplest example being batching where a single matrix is multiplied
   with a batch of examples but the more general scenario is a grouped GEMM
   where a group of matrix multiplications are scheduled together 

   2. Out of order execution: A general optimization for compilers, by looking ahead
   at the exact data dependencies within a graph we can decide on the most
   opportune time to execute a node and which buffers can be reused 
   
   3. Automatic work placement: Similar of the out of order execution point,
   but by matching nodes of a graph to resources like physical hardware or
   memory we can design an appropriate schedule

The above are general principles for accelerating PyTorch code but
different backends will each make different tradeoffs on what to
optimize. For example Inductor first takes care of fusing whatever it
can and only then generates `Triton <https://openai.com/blog/triton/>`__
kernels. It can also

Triton in addition offers speedups because of automatic memory
coalescing, memory management and scheduling within each Streaming
Multiprocessor and has been designed to handle tiled computations.

However, regardless of the backend you use it’s best to use a benchmark
and see approach so try out the PyTorch profiler, visually inspect the
generated kernels and try to see what’s going on for yourself.

Why am I not seeing speedups?
-----------------------------

Graph Breaks
~~~~~~~~~~~~

The main reason you won’t see the speedups you’d like to by using dynamo
is excessive graph breaks. So what’s a graph break?

Given a program like:

.. code:: py

   @dynamo.optimize(...)
   def some_fun(x):
       ...

   some_fun(x)
   ...

Torchdynamo will attempt to compile all of the torch/tensor operations
within ``some_fun()`` into a single FX graph, but it may fail to capture
everything into one graph.

Some graph break reasons are insurmountable to TorchDynamo like calling
into a C extension other than torch is invisible to torchdynamo, and
could do arbitrary things without TorchDynamo being able to introduce
necessary guards to ensure that the compiled program would be safe to reuse.

   To maximize performance, it’s important to have as few graph breaks
   as possible.

Identifying the cause of a graph break
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To identify all graph breaks in a program and the associated reasons for
the breaks, ``torch._dynamo.explain`` can be used. This tool runs
TorchDynamo on the supplied function and aggregates the graph breaks
that are encountered. Here is an example usage:

.. code:: py

   import torch
   import torch._dynamo as dynamo

   def toy_example(a, b):
       x = a / (torch.abs(a) + 1)
       print("woo")
       if b.sum() < 0:
           b = b * -1
       return x * b


   explanation, out_guards, graphs, ops_per_graph = dynamo.explain(toy_example, torch.randn(10), torch.randn(10))
   print(explanation)

   """
   Dynamo produced 3 graphs, with 2 graph break and 6 ops. 
    Break reasons: 

   1. call_function BuiltinVariable(print) [ConstantVariable(str)] {} 
      File "t2.py", line 16, in toy_example
       print("woo")
    
   2. generic_jump 
      File "t2.py", line 17, in toy_example
       if b.sum() < 0:
    """

To throw an error on the first graph break encountered you can use
disable python fallback by using ``nopython=True``, this should be
familiar if you’ve worked with export based compilers.

.. code:: py

   @dynamo.optimize(<compiler>, nopython=True)
   def toy_example(a, b):
      ...

Why didn’t my code recompile when I changed it?
-----------------------------------------------

If you went ahead and enabled dynamic shapes via
``env TORCHDYNAMO_DYNAMIC_SHAPES=1 python model.py`` then your code
won’t recompile on shape changes. We’ve added support for dynamic shapes
which avoids recompilations in the case when shapes vary by less than a
factor of 2. This is especially useful in scenarios like varying image
sizes in CV or variable sequence length in NLP. In inference scenarios
it’s often not possible to know what a batch size will be beforehand
because you take what you can get from different client apps.

In general torchdynamo tries very hard not to recompile things
unnecessarily so if for example torchdynamo finds 3 graphs and your
change only modified one graph then only that graph will recompile. So
another tip to avoid potentially slow compilation times is to warmup a
model by compiling it once after which subsequent compilations will be
much faster. Cold start compile times is still a metric we track
visibly.

Why am I getting incorrect results?
-----------------------------------

Accuracy issues can also be minified if you set the environment variable
``TORCHDYNAMO_REPRO_LEVEL=4``, it operates with a similar git bisect
model and a full repro might be something like
``TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4`` the reason
we need this is downstream compilers will codegen code whether it’s
Triton code or the C++ backend, the numerics from those downstream
compilers can be different in subtle ways yet have dramatic impact on
your training stability. So the accuracy debugger is very useful for us
to detect bugs in our codegen or with a backend compiler.

Why am I getting OOMs?
----------------------

Dynamo is still an alpha product so there’s a few sources of OOMs and if
you’re seeing an OOM try disabling the following configurations in this
order and then open an issue on Github so we can solve the root problem
1. If you’re using dynamic shapes try disabling them, we’ve disabled
them by default: ``env TORCHDYNAMO_DYNAMIC_SHAPES=0 python model.py`` 2.
CUDA graphs with Triton are enabled by default in inductor but removing
them may alleviate some OOM issues: ``torch._inductor.config = False``
