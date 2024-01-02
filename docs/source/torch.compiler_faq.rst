Frequently Asked Questions
==========================
**Author**: `Mark Saroufim <https://github.com/msaroufim>`_

Does ``torch.compile`` support training?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.compile`` supports training, using AOTAutograd to capture backwards:

1. The ``.forward()`` graph and ``optimizer.step()`` is captured by
   TorchDynamo’s python ``evalframe`` frontend.
2. For each segment of ``.forward()`` that torchdynamo captures, it uses
   AOTAutograd to generate a backward graph segment.
3. Each pair of forward and backward graph are (optionally) min-cut
   partitioned to save the minimal state between forward and backward.
4. The forward and backward pairs are wrapped in ``autograd.function`` modules.
5. Usercode calling\ ``.backward()`` still triggers eager’s autograd engine,
   which runs each *compiled backward* graph as if it were one op, also running
   any non-compiled eager ops’ ``.backward()`` functions.

Do you support Distributed code?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.compile`` supports ``DistributedDataParallel`` (DDP).
Support for other distributed training libraries is being considered.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the vast majority of models you probably don’t and you can use
``torch.compile()`` as is but there are a few situations where
full graphs are necessary and you can can ensure a full graph by simply
running ``torch.compile(..., nopython=True)``. These situations include:

* Large scale training runs, such as $250K+ that require pipeline parallelism
  and other advanced sharding strategies.

* Inference optimizers like `TensorRT <https://github.com/pytorch/TensorRT>`__
  or `AITemplate <https://github.com/facebookincubator/AITemplate>`__ that
  rely on fusing much more aggressively than training optimizers.

* Mobile training or inference.

Future work will include tracing communication operations into graphs,
coordinating these operations with compute optimizations, and optimizing
the communication operations.

Why is my code crashing?
~~~~~~~~~~~~~~~~~~~~~~~~

If your code ran just fine without ``torch.compile`` and started to
crash with it is enabled, then the most important first step is figuring
out which part of the stack your failure occurred. To troubleshoot that,
follow the steps below and only try the next step if the previous one
succeeded.

1. ``torch.compile(..., backend="eager")`` which only runs TorchDynamo
   forward graph capture and then runs the captured graph with PyTorch.
   If this fails then there’s an issue with TorchDynamo.

2. ``torch.compile(..., backend="aot_eager")``
   which runs TorchDynamo to capture a forward graph, and then AOTAutograd
   to trace the backward graph without any additional backend compiler
   steps. PyTorch eager will then be used to run the forward and backward
   graphs. If this fails then there’s an issue with AOTAutograd.

3. ``torch.compile(..., backend="inductor")`` which runs TorchDynamo to capture a
   forward graph, and then AOTAutograd to trace the backward graph with the
   TorchInductor compiler. If this fails then there’s an issue with TorchInductor

Why is compilation slow?
~~~~~~~~~~~~~~~~~~~~~~~~

* **Dynamo Compilation**– TorchDynamo has a builtin stats function for
  collecting and displaying the time spent in each compilation phase.
  These stats can be accessed by calling ``torch._dynamo.utils.compile_times()``
  after executing ``torch._dynamo``. By default, this returns a string
  representation of the compile times spent in each TorchDynamo function by name.

* **Inductor Compilation**– TorchInductor has a builtin stats and trace function
  for displaying time spent in each compilation phase, output code, output
  graph visualization and IR dump. ``env TORCH_COMPILE_DEBUG=1 python repro.py``.
  This is a debugging tool designed to make it easier to debug/understand the
  internals of TorchInductor with an output that will look something like
  `this <https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396>`__
  Each file in that debug trace can be enabled/disabled via
  ``torch._inductor.config.trace.*``. The profile and the diagram are both
  disabled by default since they are expensive to generate. See the
  `example debug directory
  output <https://gist.github.com/jansel/f4af078791ad681a0d4094adeb844396>`__
  for more examples.

* **Excessive Recompilation**
  When TorchDynamo compiles a function (or part of one), it makes certain
  assumptions about locals and globals in order to allow compiler
  optimizations, and expresses these assumptions as guards that check
  particular values at runtime. If any of these guards fail, Dynamo will
  recompile that function (or part) up to
  ``torch._dynamo.config.cache_size_limit`` times. If your program is
  hitting the cache limit, you will first need to determine which guard is
  failing and what part of your program is triggering it. The
  `recompilation profiler <#recompilation-profiler>`__ automates the
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph Breaks
------------

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
into a C extension other than PyTorch is invisible to TorchDynamo, and
could do arbitrary things without TorchDynamo being able to introduce
necessary guards to ensure that the compiled program would be safe to reuse.

   To maximize performance, it’s important to have as few graph breaks
   as possible.

Identifying the cause of a graph break
--------------------------------------

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

If you enabled dynamic shapes by setting
``env TORCHDYNAMO_DYNAMIC_SHAPES=1 python model.py`` then your code
won’t recompile on shape changes. We’ve added support for dynamic shapes
which avoids recompilations in the case when shapes vary by less than a
factor of 2. This is especially useful in scenarios like varying image
sizes in CV or variable sequence length in NLP. In inference scenarios
it’s often not possible to know what a batch size will be beforehand
because you take what you can get from different client apps.

In general, TorchDynamo tries very hard not to recompile things
unnecessarily so if for example TorchDynamo finds 3 graphs and your
change only modified one graph then only that graph will recompile. So
another tip to avoid potentially slow compilation times is to warmup a
model by compiling it once after which subsequent compilations will be
much faster. Cold start compile times is still a metric we track
visibly.

Why am I getting incorrect results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

Dynamo is still an alpha product so there’s a few sources of OOMs and if
you’re seeing an OOM try disabling the following configurations in this
order and then open an issue on GitHub so we can solve the root problem
1. If you’re using dynamic shapes try disabling them, we’ve disabled
them by default: ``env TORCHDYNAMO_DYNAMIC_SHAPES=0 python model.py`` 2.
CUDA graphs with Triton are enabled by default in inductor but removing
them may alleviate some OOM issues: ``torch._inductor.config.triton.cudagraphs = False``.

Does ``torch.func`` work with ``torch.compile`` (for `grad` and `vmap` transforms)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying a ``torch.func`` transform to a function that uses ``torch.compile``
does not work:

.. code-block:: python

    import torch

    @torch.compile
    def f(x):
        return torch.sin(x)

    def g(x):
        return torch.grad(f)(x)

    x = torch.randn(2, 3)
    g(x)

This code will not work. There is an `issue <https://github.com/pytorch/pytorch/issues/100320>`__
that you can track for this.

As a workaround, use ``torch.compile`` outside of the ``torch.func`` function:

.. note::
    This is an experimental feature and can be used by setting `torch._dynamo.config.capture_func_transforms=True`

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    def f(x):
        return torch.sin(x)

    @torch.compile
    def g(x):
        return torch.vmap(f)(x)

    x = torch.randn(2, 3)
    g(x)

Calling ``torch.func`` transform inside of a function handled with ``torch.compile``
------------------------------------------------------------------------------------


Compiling ``torch.func.grad`` with ``torch.compile``
----------------------------------------------------

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    def wrapper_fn(x):
        return torch.func.grad(lambda x: x.sin().sum())(x)

    x = torch.randn(3, 3, 3)
    grad_x = torch.compile(wrapper_fn)(x)

Compiling ``torch.vmap`` with ``torch.compile``
-----------------------------------------------

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    def my_fn(x):
        return torch.vmap(lambda x: x.sum(1))(x)

    x = torch.randn(3, 3, 3)
    output = torch.compile(my_fn)(x)

Limitations
-----------

There are currently a few cases which are not supported and lead to graph breaks
(that is, torch.compile falls back to eager-mode PyTorch on these). We are working
on improving the situation for the next release (PyTorch 2.2)

1. The inputs and outputs of the function being transformed over must be tensors.
We do not yet support things like tuple of Tensors.

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    def fn(x):
        x1, x2 = x
        return x1 + x2

    def my_fn(x):
        return torch.func.vmap(fn)(x)

    x1 = torch.randn(3, 3, 3)
    x2 = torch.randn(3, 3, 3)
    # Unsupported, falls back to eager-mode PyTorch
    output = torch.compile(my_fn)((x1, x2))

2. Keyword arguments are not supported.

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    def fn(x, y):
        return (x + y).sum()

    def my_fn(x, y):
        return torch.func.grad(fn)(x, y=y)

    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    # Unsupported, falls back to eager-mode PyTorch
    output = torch.compile(my_fn)(x, y)

3. Functions with observable side effects. For example, it is OK to mutate a list created in the function,
but not OK to mutate a list created outside of the function.

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    some_list = []

    def f(x, y):
        some_list.append(1)
        return x + y

    def my_fn(x, y):
        return torch.func.vmap(f)(x, y)

    x = torch.ones(2, 3)
    y = torch.randn(2, 3)
    # Unsupported, falls back to eager-mode PyTorch
    output = torch.compile(my_fn)(x, y)

4. ``torch.vmap`` over a function that calls one or more operators in the following list.

.. note::
    'stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex',
    'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point',
    'is_inference', 'is_ipu', 'is_leaf', 'is_meta', 'is_mkldnn', 'is_mps', 'is_neg', 'is_nested',
    'is_nonzero', 'is_ort', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared',
    'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu'

.. code-block:: python

    import torch

    torch._dynamo.config.capture_func_transforms=True

    def bad_fn(x):
        x.stride()
        return x

    def my_fn(x):
        return torch.func.vmap(bad_fn)(x)

    x = torch.randn(3, 3, 3)
    # Unsupported, falls back to eager-mode PyTorch
    output = torch.compile(my_fn)(x)

Compiling functions besides the ones which are supported (escape hatch)
-----------------------------------------------------------------------

For other transforms, as a workaround, use ``torch._dynamo.allow_in_graph``

``allow_in_graph`` is an escape hatch. If your code does not work with
``torch.compile``, which introspects Python bytecode, but you believe it
will work via a symbolic tracing approach (like ``jax.jit``), then use
``allow_in_graph``.

By using ``allow_in_graph`` to annotate a function, you must make sure
your code meets the following requirements:

- All outputs in your function only depend on the inputs and
  do not depend on any captured Tensors.
- Your function is functional. That is, it does not mutate any state. This may
  be relaxed; we actually support functions that appear to be functional from
  the outside: they may have in-place PyTorch operations, but may not mutate
  global state or inputs to the function.
- Your function does not raise data-dependent errors.

.. code-block:: python

    import torch

    @torch.compile
    def f(x):
        return torch._dynamo.allow_in_graph(torch.vmap(torch.sum))(x)

    x = torch.randn(2, 3)
    f(x)

A common pitfall is using ``allow_in_graph`` to annotate a function that
invokes an ``nn.Module``. This is because the outputs now depend on the
parameters of the ``nn.Module``. To get this to work, use
``torch.func.functional_call`` to extract the module state.

Does NumPy work with ``torch.compile``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting in 2.1, ``torch.compile`` understands native NumPy programs that
work on NumPy arrays, and mixed PyTorch-NumPy programs that convert from PyTorch
to NumPy and back via ``x.numpy()``, ``torch.from_numpy``, and related functions.

.. _nonsupported-numpy-feats:

Which NumPy features does ``torch.compile`` support?
----------------------------------------------------

NumPy within ``torch.compile`` follows NumPy 2.0 pre-release.

Generally, ``torch.compile`` is able to trace through most NumPy constructions,
and when it cannot, it falls back to eager and lets NumPy execute that piece of
code. Even then, there are a few features where ``torch.compile`` semantics
slightly deviate from those of NumPy:

- NumPy scalars: We model them as 0-D arrays. That is, ``np.float32(3)`` returns
  a 0-D array under ``torch.compile``. To avoid a graph break, it is best to use this 0-D
  array. If this breaks your code, you can workaround this by casting the NumPy scalar
  to the relevant Python scalar type ``bool/int/float``.

- Negative strides: ``np.flip`` and slicing with a negative step return a copy.

- Type promotion: NumPy's type promotion will change in NumPy 2.0. The new rules
  are described in `NEP 50 <https://numpy.org/neps/nep-0050-scalar-promotion.html)>`__.
  ``torch.compile`` implements NEP 50 rather than the current soon-to-be deprecated rules.

- ``{tril,triu}_indices_from/{tril,triu}_indices`` return arrays rather than a tuple of arrays.

There are other features for which we do not support tracing and we gracefully
fallback to NumPy for their execution:

- Non-numeric dtypes like datetimes, strings, chars, void, structured dtypes and recarrays.

- Long dtypes ``np.float128/np.complex256`` and some unsigned dtypes ``np.uint16/np.uint32/np.uint64``.

- ``ndarray`` subclasses.

- Masked arrays.

- Esoteric ufunc machinery like ``axes=[(n,k),(k,m)->(n,m)]`` and ufunc methods (e.g., ``np.add.reduce``).

- Sorting / ordering ``complex64/complex128`` arrays.

- NumPy ``np.poly1d`` and ``np.polynomial``.

- Positional ``out1, out2`` args in functions with 2 or more returns (``out=tuple`` does work).

- ``__array_function__``, ``__array_interface__`` and ``__array_wrap__``.

- ``ndarray.ctypes`` attribute.

Can I execute NumPy code on CUDA via ``torch.compile``?
-------------------------------------------------------

Yes you can! To do so, you may simply execute your code within a ``torch.device("cuda")``
context. Consider the example

.. code-block:: python

   import torch
   import numpy as np

   @torch.compile
   def numpy_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
       return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))

   X = np.random.randn(1024, 64)
   Y = np.random.randn(1024, 64)
   with torch.device("cuda"):
       Z = numpy_fn(X, Y)


In this example, ``numpy_fn`` will be executed in CUDA. For this to be
possible, ``torch.compile`` automatically moves ``X`` and ``Y`` from CPU
to CUDA, and then it moves the result ``Z`` from CUDA to CPU. If we are
executing this function several times in the same program run, we may want
to avoid all these rather expensive memory copies. To do so, we just need
to tweak our ``numpy_fn`` so that it accepts cuda Tensors and returns tensors:

.. code-block:: python

   @torch.compile
   def numpy_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
       X, Y = X.numpy(), Y.numpy()
       Z = np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))
       return torch.from_numpy(Z)

   X = torch.randn(1024, 64, device="cuda")
   Y = torch.randn(1024, 64, device="cuda")
   with torch.device("cuda"):
       Z = numpy_fn(X, Y)

By doing this, we explicitly create the tensors in CUDA memory, and we keep
them there. In this case ``X.numpy()`` and ``from_numpy()`` are hints to the compiler
but no real data movement happens. Note that the original program would not run
on eager mode now. If you want to run it in eager mode, you would need to call
``.numpy(force=True)`` doing ``Z = Z.cuda()`` before returning
``Z``. Of course, doing this would execute the program on eager mode NumPy, and
on CPU.


How do I debug NumPy code under ``torch.compile``?
--------------------------------------------------

Debugging JIT compiled code is challenging, given the complexity of modern
compilers and the daunting errors that they raise.
`The tutorial on how to diagnose runtime errors within torch.compile <https://pytorch.org/docs/main/torch.compiler_troubleshooting.html#diagnosing-runtime-errors>`__
contains a few tips and tricks on how to tackle this task.

If the above is not enough to pinpoint the origin of the issue, there are still
a few other NumPy-specific tools we can use. We can discern whether the bug
is entirely in the PyTorch code by disabling tracing through NumPy functions:


.. code-block:: python

   from torch._dynamo import config
   config.trace_numpy = False

If the bug lies in the traced NumPy code, we can execute the NumPy code eagerly (without ``torch.compile``)
using PyTorch as a backend by importing ``import torch._numpy as np``.
This should just be used for **debugging purposes** and is in no way a
replacement for the PyTorch API, as it is **much less performant** and, as a
private API, **may change without notice**. At any rate, ``torch._numpy`` is a
Python implementation of NumPy in terms of PyTorch and it is used internally by ``torch.compile`` to
transform NumPy code into Pytorch code. It is rather easy to read and modify,
so if you find any bug in it feel free to submit a PR fixing it or simply open
an issue.

If the program does work when importing ``torch._numpy as np``, chances are
that the bug is in TorchDynamo. If this is the case, please feel open an issue
with a `minimal reproducer <https://pytorch.org/docs/2.1/torch.compiler_troubleshooting.html>`__.

I ``torch.compile`` some NumPy code and I did not see any speed-up.
-------------------------------------------------------------------

The best place to start is the
`tutorial with general advice for how to debug these sort of torch.compile issues <https://pytorch.org/docs/main/torch.compiler_faq.html#why-am-i-not-seeing-speedups>`__.

Some graph breaks may happen because of the use of unsupported features. See
:ref:`nonsupported-numpy-feats`. More generally, it is useful to keep in mind
that some widely used NumPy features do not play well with compilers. For
example, in-place modifications make reasoning difficult within the compiler and
often yield worse performance than their out-of-place counterparts.As such, it is best to avoid
them. Same goes for the use of the ``out=`` parameter. Instead, prefer
out-of-place ops and let ``torch.compile`` optimize the memory use. Same goes
for data-dependent ops like masked indexing through boolean masks, or
data-dependent control flow like ``if`` or ``while`` constructions.


Which API to use for fine grain tracing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, you might need to exclude small parts of your code from the
torch.compile compilations. This section provides some of the answers and
you can find more information in :ref:`torchdynamo_fine_grain_tracing`.

How do I graph break on a function?
-----------------------------------

Graph break on a function is not enough to sufficiently express what you  want
PyTorch to do. You need to be more specific about your use case. Some of the
most common use cases you might want to consider:

* If you want to disable compilation on this function frame and the recursively
  invoked frames, use ``torch._dynamo.disable``.

* If you want a particular operator, such as ``fbgemm`` to use the  eager mode,
  use ``torch._dynamo.disallow_in_graph``.

Some of the uncommon use cases include:

* If you want to disable TorchDynamo on the function frame but enable it back
  on the recursively invoked frames – use ``torch._dynamo.disable(recursive=False)``.

* If you want to prevent inlining of a function frame – use ``torch._dynamo.graph_break``
  at the beginning of the function you want to prevent inlining.

What's the difference between ``torch._dynamo.disable`` and ``torch._dynamo.disallow_in_graph``
-----------------------------------------------------------------------------------------------

Disallow-in-graph works at the level of operators, or more specifically,
the operators that you see in the TorchDynamo extracted graphs.

Disable works at the function frame level and decides if TorchDynamo
should look into the function frame or not.

What's the difference between ``torch._dynamo.disable`` and ``torch._dynamo_skip``
----------------------------------------------------------------------------------

.. note::
   ``torch._dynamo_skip`` is deprecated.

You most likely need ``torch._dynamo.disable``. But in an unlikely scenario, you
might need even finer control. Suppose you want to disable the tracing on just
the ``a_fn`` function, but want to continue the tracing back in ``aa_fn`` and
``ab_fn``. The image below demonstrates this use case:


.. figure:: _static/img/fine_grained_apis/call_stack_diagram.png
   :alt: diagram of torch.compile + disable(a_fn, recursive=False)

In this case, you can use ``torch._dynamo.disable(recursive=False)``.
In previous versions, this functionality was provided by ``torch._dynamo.skip``.
This is now supported by the ``recursive`` flag inside ``torch._dynamo.disable``.
