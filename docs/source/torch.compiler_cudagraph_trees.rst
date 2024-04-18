CUDAGraph Trees
================

CUDAGraph Background
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

CUDAGraph Trees Integration
---------------------------

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


In the Separate CUDA Graph implementation, the output from the first invocation will be overwritten by the second invocation. In CUDA Graph Trees, we don’t want to add unintended dependencies between iterations that would cause us to not hit the hot path, nor do we want we want to prematurely free memory from a prior invocation. Our heuristics are in inference we start a new iteration on each invocation for torch.compile, and in training we do the same so long as there is not a pending backward that has not been invoked. If those heuristics are wrong, you can mark the start of a new iteration with torch.compiler.mark_step_begin(), or clone tensors of a prior iteration (outside of torch.compile) before you begin the next run.

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
