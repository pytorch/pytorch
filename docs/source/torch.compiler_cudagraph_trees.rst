CUDAGraph Trees
================

**Background**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDAGraph
--------------------

For a longer background on CUDAGraphs, read `accelerating pytorch with CUDAGraphs <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>`_.

`CUDA Graphs <https://developer.nvidia.com/blog/cuda-10-features-revealed/>`_, which made its debut in CUDA 10, let a series of CUDA kernels to be defined and encapsulated as a single unit, i.e., a graph of operations, rather than a sequence of individually-launched operations. It provides a mechanism to launch multiple GPU operations through a single CPU operation, and hence reduces the launching overheads.

CUDA Graphs can give large speedups, especially for models with high CPU overhead or small compute. There are a number of limitations from requiring the same kernels to be run with the same arguments and dependencies, and memory addresses.

- Control Flow is not possible
- Kernels which trigger host to device syncs (such as .item()) errors
- All input arguments to kernels are fixed to what they were recorded
- CUDA Memory addresses are fixed, however the values of the memory at those addresses can change
- No Essential CPU ops or CPU side effects

PyTorch CUDAGraph Integration
-----------------------------

PyTorch provides a `convenience wrapper <https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html>`_ around CUDAGraphs that handles a couple of tricky interactions with PyTorch’s caching allocator.

The CachingAllocator uses a separate memory pool for all the new allocations. During CUDAGraph recording, memory is accounted for, allocated, and freed exactly as during eager run. On replay, just the kernels are invoked, and there are no changes to the allocator. Subsequent to initial recording, the allocator does not know which memory is actively being used in user programs.

Using a separate memory pool between eager allocations and cudagraph allocations may increase the memory of your program if there is substantial memory allocated to both.

Make Graphed Callables
----------------------

`Make Graphed Callables <https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html>`_ is a PyTorch Abstraction to share a single memory pool over a series of callables. Graphed Callables takes advantage of the fact that on CUDA Graph recording, memory is exactly accounted for by the caching allocator to safely share memory between separate CUDA Graph recordings. In each invocation, outputs are preserved as live memory, preventing one callable from overwriting the live memory of another. Graphed Callables can only be invoked in a single order; memory addresses from the first run are burned into the second, and so forth.

TorchDynamo Previous CUDA Graphs Integration
--------------------------------------------

Running with ``cudagraph_trees=False`` does not reuse memory across separate graph captures, which can lead to large memory regressions. Even for a model that has no graph breaks, this has issues. The forward and backward are separate graph captures, so the memory pools for forward and backward are not shared. In particular, memory for activations that are saved in the forward cannot be reclaimed in the backward.

**CUDAGraph Trees Integration**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like Graph Callables, CUDA Graph Trees use a single memory pool across all graph captures. However, instead of requiring a single sequence of invocations, CUDA Graph Trees create separate trees of CUDA Graph captures. Let’s take a look at an illustrative example:

.. code-block:: python

    @torch.compile(mode="reduce-overhead")
    def foo(x):
        # GRAPH 1
        y = x * x * x
        # graph break triggered here
        if y.sum() > 0:
            # GRAPH 2
            z = y ** y
        else:
            # GRAPH 3
            z = (y.abs() ** y.abs())
        torch._dynamo.graph_break()
        # GRAPH 4
        return z * torch.rand_like(z)

    # the first run warms up each graph, which does things like CuBlas or Triton benchmarking
    foo(torch.arange(0, 10, device="cuda"))
    # The second run does a CUDA Graph recording, and replays it
    foo(torch.arange(0, 10, device="cuda"))
    # Finally we hit the optimized, CUDA Graph replay path
    foo(torch.arange(0, 10, device="cuda"))


In this example, there are two separate paths that we make through the function: 1 -> 2 -> 4, or 1 -> 3 -> 4.

We share all of the memory in a single memory pool between separate recordings by building up a tape of CUDA Graph recordings, in this instance, 1 -> 2 -> 4. We add invariants to ensure that memory is always in the same location as it were recorded, and no live tensors exist in user programs that might be overwritten.

- Same constraints from CUDA Graphs apply: same kernels must be invoked with the same arguments (static sizes, addresses, etc)
- The same pattern of memory must be observed between recording and replay: if a tensor output of one graph dies subsequent to another graph during recording, it must also do so during replay.
- Live memory in the CUDA pool forces a dependence between two recordings
- These recordings can only be invoked in a single order 1 - > 2 -> 4

All of the memory is shared in a single memory pool, so there is no additional memory overhead compared to eager. Now, what happens if we were to hit a new path and run Graph 3?

Graph 1 gets replayed, and then we hit Graph 3, which we have not yet recorded. On graph replays, the private memory pool is not updated, so y is not reflected in the allocator. Without care, we would overwrite it. To support reusing the same memory pool after replaying other graphs, we checkpoint the memory pool back to its state at the end of graph 1. Now that our live tensors are reflected in the caching allocator, we are safe to run a new graph.

First, we would hit the optimized, CUDAGraph.replay() path that we have already recorded in graph 1. Then we would hit Graph 3. Just as before, we will need to warm up the graph once before recording. On the warmup run, the memory addresses are not fixed, so graph 4 will also fallback to the inductor, non-cudagraph invocation.

The second time we hit graph 3 we are warmed up and ready to record. We record graph 3 and then record graph 4 again since the input memory addresses have changed. This creates a tree of CUDA Graph recordings. A CUDA Graph Tree!

::

    1
   / \\
  2   3
   \\   \\
    4   4


Input Mutation Support
----------------------

Input mutation function refers to a function conducting in-place writes to an input tensor,
as illustrated below:

.. code-block:: python

    def foo(x, y):
        # mutates input x
        x.add_(1)
        return x + y

Input mutation functions generally lead to challenges for CUDAGraph Trees. Due to the static
CUDA memory address requirement from CUDAGraph, for each input tensor x, CUDAGraph Trees may
allocate a static memory address x'. During execution, CUDAGraph Trees first copy the input
tensor x to the static memory address x', and then replay the recorded CUDAGraph. For input
mutation function, x' is in-place updated, which is not reflected on the input tensor x since
x and x' reside on different CUDA memory addresses.

A closer look at input mutation functions reveals that there are three types of inputs:

* **inputs from eager**: These tensors we assume will vary input tensor addresses from
  execution to execution. Because cudagraphs freeze memory   addresses, we need to copy these
  inputs to a static address tensor prior to graph recording and execution.
* **Parameters and buffers**: These tensors we assume (and runtime-check) have the same tensor
  addresses on every execution. We do not need to copy over their contents because the recorded
  memory address will be the same as the executed memory address.
* **Tensors which are prior outputs from CUDAGraph Trees**: Because the output tensor addresses
  of a cudagraph are fixed, if we run CUDAGraph1, then run CUDAGraph2, the inputs which came from
  CUDAGraph1 into CUDAGraph2 will have a fixed memory address. These inputs, like parameters and
  buffers, do not require copying over to a static address tensor. We check to make sure that
  these inputs are stable at runtime, and if they're not we will re-record.

CUDAGraph Trees support input mutation on parameters and buffers, and tensors which are prior
outputs from CUDAGraph Trees. For mutation on inputs from eager, CUDAGraph Trees will run the
function without CUDAGraph and emit *skipping due to mutated inputs* log. The following example
shows CUDAGraph Trees' support for tensors which are prior outputs from CUDAGraph Trees.


.. code-block:: python

    import torch

    @torch.compile(mode="reduce-overhead")
    def foo(x):
        return x + 1

    @torch.compile(mode="reduce-overhead")
    def mut(x):
        return x.add_(2)

    # Enable input mutation support
    torch._inductor.config.triton.cudagraph_support_input_mutation = True

    for i in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        inp = torch.rand([4], device="cuda")

        # CUDAGraph is applied since `foo` does not mutate `inp`
        tmp = foo(inp)
        # Although `mut` mutates `tmp`, which is an output of a CUDAGraph
        # managed function. So CUDAGraph is still applied.
        mut(tmp)


    torch.compiler.cudagraph_mark_step_begin()
    inp = torch.rand([4], device="cuda")

    tmp = foo(inp)
    # While `tmp` is a CUDAGraph Tree managed function's output, `tmp.clone()`
    # is not. So CUDAGraph is not applied to `mut` and there is a log
    # `skipping cudagraphs due to mutated inputs`
    mut(tmp.clone())


To enable CUDAGraph Trees for a function mutating inputs from eager, please re-write
the function to avoid input mutation.

.. note:: Enable input mutation support by setting
  `torch._inductor.config.cudagraph_support_input_mutation = True <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L662>`_
  for "reduce-overhead" mode.


Dynamic Shape Support
---------------------

`Dynamic shape <https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html>`_
means that an input tensor has different shapes across function calls. Since CUDAGraph
requires fixed tensor addresses, CUDAGraph Trees re-record CUDAGraph for every unique
shape of an input tensor. This leads to multiple CUDAGraphs for a single inductor graph.
When there are limited shapes (e.g., batch sizes in inference), it is profitable to
re-record CUDAGraphs. However, if input tensor shapes change frequently or even on
every invocation, re-recording CUDAGraph may not be profitable. Nvidia uses 64 KB of
device memory per kernel launch in CUDAGraph, up until CUDA 12.4 and Driver Version 550+.
This memory cost can be significant with many CUDAGraph re-recordings.

For functions with frequently changing input tensor shapes, we suggest padding input
tensors to a few fixed tensor shapes to still enjoy benefits from CUDAGraph. In addition,
setting  `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True <https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L653>`_
allows to skip cudagraphing functions with dynamic shape inputs and only cudagraphing
functions with static input tensor shapes.


NCCL Support
------------

CUDAGraph Trees support functions with nccl operators. While CUDAGraph Trees perform per-device
record for CUDAGraph, NCCL support allows cross-device communication.

.. code-block:: python

    @torch.compile(mode="reduce-overhead")
    def func(x):
        y = x * x
        y = torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
        x = torch.nn.functional.silu(x)
        return x * y


Reasons for Skipping CUDAGraph
------------------------------

Since CUDAGraph has requirements such as static input tensor addresses and not supporting
CPU operators, CUDAGraph Trees check whether a function satisfies these requirements and
may skip CUDAGraph when necessary. Here, we list common reasons for skipping CUDAGraph.

* **Input mutation**: CUDAGraph Trees skip functions that in-place mutates eager input.
  In-place mutating parameters and buffers, or output tensors from CUDAGraph Tree managed
  functions are still supported. Please see *Input Mutation Support* section for more details.
* **CPU operators**: Functions containing CPU operator are skipped. Please split the
  function into multiple functions and apply CUDAGraph Trees on functions with only GPU operators.
* **Multi-device operators**: A function is skipped if it contains operators on multiple
  devices. Currently, CUDAGraph is applied on a per-device basis. Please use supported
  libraries such as NCCL for cross-device communication. Please see *NCCL Support*
  section for more details.
* **Free unbacked symbols**: Free unbacked symbols usually happen during
  `dynamic shapes <https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html>`_.
  CUDAGraph Trees currently record a CUDAGraph for every unique input tensor shapes.
  Please see *Dynamic Shape Support* for more details.
* **Incompatible operators**: CUDAGraph Trees skip a function if it contain incompatible
  operators. Please replace these operators in a function with supported operators. We
  show an exhaustive list of incompatible operators:


.. code-block:: python

    aten._fused_moving_avg_obs_fq_helper.default
    aten._fused_moving_avg_obs_fq_helper_functional.default
    aten.multinomial.default
    fbgemm.dense_to_jagged.default
    fbgemm.jagged_to_padded_dense.default
    run_and_save_rng_state
    run_with_rng_state
    aten._local_scalar_dense
    aten._assert_scalar


The following operators are incompatible when `torch.are_deterministic_algorithms_enabled() <https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled.html>`_.


.. code-block:: python

    aten._fused_moving_avg_obs_fq_helper.default
    aten._fused_moving_avg_obs_fq_helper_functional.default
    aten.multinomial.default
    fbgemm.dense_to_jagged.default
    fbgemm.jagged_to_padded_dense.default
    run_and_save_rng_state
    run_with_rng_state
    aten._local_scalar_dense
    aten._assert_scalar


Limitations
-----------

Because CUDA Graph fixes memory addresses, CUDA Graphs do not have a great way of handling live tensors from a previous invocation.

Let’s say we are benchmarking running inference with the following code:

.. code-block:: python

    import torch

    @torch.compile(mode="reduce-overhead")
    def my_model(x):
        y = torch.matmul(x, x)
        return y

    x = torch.randn(10, 10)
    y1 = my_model(x)
    y2 = my_model(x)
    print(y1)
    # RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.

In the Separate CUDA Graph implementation, the output from the first invocation will be overwritten by the second invocation. In CUDAGraph
Trees, we don’t want to add unintended dependencies between iterations that would cause us to not hit the hot path, nor do we want we want
to prematurely free memory from a prior invocation. Our heuristics are in inference we start a new iteration on each invocation for
torch.compile, and in training we do the same so long as there is not a pending backward that has not been invoked. If those heuristics
are wrong, you can mark the start of a new iteration with
`torch.compiler.mark_step_begin() <https://pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html>`_, or clone
tensors of a prior iteration (outside of torch.compile) before you begin the next run.


Comparisons
-----------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Footguns
     - Separate CudaGraph
     - CUDAGraph Trees
   * - Memory Can Increase
     - On each graph compilation (new sizes, etc.)
     - If you are also running non-cudagraph memory
   * - Recordings
     - On any new invocation of a graph
     - Will re-record on any new, unique path you take through your program
   * - Footguns
     - Invocation of one graph will overwrite prior invocation
     - Cannot persist memory between separate runs through your model - one training loop training, or one run of inference
