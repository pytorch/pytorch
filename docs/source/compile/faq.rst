Frequently Asked Questions
==========================
**Author**: `Mark Saroufim <https://github.com/msaroufim>`_

At a high level, the PyTorch 2.0 stack consists of a graph capture from
Python code using dynamo and a backend compiler. In this example the
backend compiler consists of backward graph tracing using AOTAutograd
and graph lowering using TorchInductor. There are of course many more
compilers available `here <https://github.com/pytorch/torchdynamo/blob/0b8aaf340dad4777a080ef24bf09623f1aa6f3dd/README.md#existing-backend>`__
but for this document we will focus on inductor as a motivating example.

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
`distributed.py <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/optimizations/distributed.py>`__
where the main idea will be to graph break on `DDP bucket
boundaries <https://pytorch.org/docs/stable/notes/ddp.html#internal-design>`__.

When each node in DDP needs to synchronize its weights with the other
nodes it organizes its gradients and parameters into buckets which
reduces communication times and allows a node to broadcast a fraction of
its gradients to other waiting nodes.

Graph breaks in distributed code mean you can expect dynamo and its
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
`TensorRT <https://github.com/pytorch/TensorRT>`__ or
`AITemplate <https://github.com/facebookincubator/AITemplate>`__ that rely
on fusing much more aggressively than training optimizers \* Mobile training or
inference.

Future work will include tracing communication operations into graphs,
coordinating these operations with compute optimizations, and optimizing
the communication operations.

Why is my code crashing?
------------------------

If your code ran just fine without dynamo and started to crash with it
enabled then the most important first step is figuring out which part of
the stack your failure occurred in so try running things in the below
order and only try the next step if the previous step succeeded.

1. ``torch.compile(..., backend="eager")`` which only runs torchdynamo forward graph
   capture and then runs the captured graph with PyTorch. If this fails
   then there’s an issue with TorchDynamo.

2. ``torch.compile(..., backend="aot_eager")``
   which runs torchdynamo to capture a forward graph, and then AOTAutograd
   to trace the backward graph without any additional backend compiler
   steps. PyTorch eager will then be used to run the forward and backward
   graphs. If this fails then there’s an issue with AOTAutograd.

3. ``torch.compile(..., backend="inductor")`` which runs torchdynamo to capture a
   forward graph, and then AOTAutograd to trace the backward graph with the
   TorchInductor compiler. If this fails then there’s an issue with TorchInductor

.. :_torchdynamo_errors

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
debugging. There are 2 tools available to enable this:

* ``env TORCHDYNAMO_DEBUG_FUNCTION=<desired_function_name>`` will only run TorchDynamo on functions with that name.

* ``env torch._dynamo.config.replay_record_enabled = True``) which dumps an execution record when an error is encountered. This record can then be replayed to run only the frame where an error occurred.

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

Minifying your error is the quickest path to getting it fixed.

The minifier will actually create a ``repro.py`` for you at the location
set by ``env TORCHDYNAMO_REPRO_DIR`` so make you have right access to
that directory. You can then run ``python repro.py`` and confirm that
you are getting the same error.

.. note::
   For other compilers such as nvfuser, the process is similar but
   instead you would leverage ``env TORCHDYNAMO_REPRO_AFTER="dynamo" python your_model.py``.

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
and IR dump. ``env TORCH_COMPILE_DEBUG=1 python repro.py``. This is a
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

.. code-block:: python

   from torch._dynamo.utils import CompileProfiler

   def my_model():
       ...

   with CompileProfiler() as prof:
       profiler_model = torch.compile(my_model, backend=prof)
       profiler_model()
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

.. code-block:: python

   frozen_toy_example = dynamo.run(toy_example)
   frozen_toy_example(torch.randn(10), torch.randn(10))

How are you speeding up my code?
--------------------------------

There are 3 major ways to accelerate PyTorch code:

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

.. code-block:: python

   def some_fun(x):
       ...

   torch.compile(some_fun)(x)
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

.. code-block:: python

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

.. code-block:: python

   def toy_example(a, b):
      ...

   torch.compile(toy_example, fullgraph=True, backend=<compiler>)

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

In general, TorchDynamo tries very hard not to recompile things
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

If you'd like to ensure that random number generation is the same across both torch
and triton then you can enable ``torch._inductor.config.fallback_random = True``

Why am I getting OOMs?
----------------------

Dynamo is still an alpha product so there’s a few sources of OOMs and if
you’re seeing an OOM try disabling the following configurations in this
order and then open an issue on GitHub so we can solve the root problem
1. If you’re using dynamic shapes try disabling them, we’ve disabled
them by default: ``env TORCHDYNAMO_DYNAMIC_SHAPES=0 python model.py`` 2.
CUDA graphs with Triton are enabled by default in inductor but removing
them may alleviate some OOM issues: ``torch._inductor.config.triton.cudagraphs = False``.
