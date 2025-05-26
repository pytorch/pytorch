Fake tensor
===========

Code: `fake_tensor.py <https://github.com/pytorch/pytorch/blob/db4572dbf18f1cf50cf662547e272d3117063747/torch/_subclasses/fake_tensor.py>`_

Motivation
----------

When doing Dynamo symbolic evaluation and compiler passes, we often want to be able to run tensor operations to understand what output sizes/dtypes/devices are, without actually running those operations (or trashing preexisting tensors), which would be slower (if you're doing a lot of compute) and take a lot of memory (it's bad if your compiler needs to use GPU memory while you are compiling the program). A fake tensor is like a real tensor in all respects, except that it doesn't actually have any data. For example, when we do Dynamo tracing, we need to trace through user Tensor code and answer questions about intermediates (e.g., if a user does a conditional on an intermediate tensor). Without fake tensor, we would not have accurate information for these queries.

Similarly, suppose you want to store metadata for a tensor, e.g., on an FX IR node (meta['val']). You can instead store a fake tensor directly on the node, which will give you all the metadata you need for the tensor, including subtle stuff that you probably wouldn't have handled (e.g., aliasing relationships).

Related work
------------

- A meta tensor is a tensor with device='meta'. This is actually a lot of what you want for fake tensor, but meta tensors don't model devices, and sometimes stride behavior varies depending on your device, so fake tensors really can get a lot more accurate info this way. Also, meta tensors are "global" (they exist on their own, similar to how a CPU/CUDA tensor exist on their own), whereas fake tensors are scoped to a FakeTensorMode.

- A tensor subclass lets you subclass torch.Tensor and customize their behavior. Fake tensors are implemented as a tensor subclass; that means almost all of its implementation lives in Python! For more simple examples of tensor subclasses check out `subclass_zoo <https://github.com/albanD/subclass_zoo/>`_.

- Dynamic shapes allow you to create tensors with symbolic sizes rather than only concrete sizes, and propagate these sizes symbolically through operations. Dynamic shapes maintain state in a ShapeEnv, which is always associated with a FakeTensorMode (so fake tensors also are responsible for managing symbolic sizes.) In general, whenever we compile a subgraph with PT2, there is a tracing context associated with this compilation, which contains, among other things, a FakeTensorMode and (possibly) a ShapeEnv.

Overall architecture
--------------------

All fake tensors are associated with a FakeTensorMode. Because fake tensor's primary use case is to do analysis on real tensors, the general workflow is you have a bunch of real tensors, you allocate a FakeTensorMode, and then you use from_real_tensor to convert all those real tensors into fake tensors, and then you do things to the fake tensors. In particular, the FakeTensorMode maintains a memo table persistently mapping tensors (and storages) to the same storages. If you fakeify the same tensor multiple times, you will get the same fake tensor; if you fakeify two tensors which alias each other, you will get two fake tensors which alias the same fake storage. FakeTensors are tensor subclasses, so if you do operations on them, you'll automatically get a fake tensor, but in general you will want to do operations on fake tensors (e.g., if you're running an FX pass) with the FakeTensorMode active; what a tensor operation will do is automatically turn on the fake tensor mode and try again.

A fake tensor is represented as a __torch_dispatch__ tensor subclass of a meta tensor. This means under the hood, fake tensors are meta device tensors; they then use extra extensibility hooks, specifically dispatch_device, to lie about what the actual device of the tensor is. This was one of the more error-prone parts of fake tensors in the early days: sometimes, fake tensors were too good at lying about being CPU/CUDA whatever, and you'd end up with a CPU kernel getting called with a fake tensor trying to dereference the data pointer, which obviously won't work. If you are segfaulting in fake tensor code, this is the first thing you should check: is the C++ backtrace in a CPU kernel (unexpected!) or a meta kernel (expected!) A meta kernel is like a real kernel, but all it does is allocate the outputs, it doesn't do any data compute.

A tensor subclass has to define how to implement various operations. Here is the general fake tensor recipe:

- Run the meta kernel on the input fake tensors, reinterpreting them as meta tensors. This is done via a magic context manager in_kernel_invocation_manager which instructs all of PyTorch to view fake tensors as their underlying meta tensors, rather than "unwrapping" fake tensors into meta tensors (a fake tensor is a meta tensor). Fake tensors are represented this way to avoid having to keep two sets of metadata in sync (the meta tensor's metadata, and the fake tensor's metadata); the "is a" relationship ensures there is only one canonical copy of metadata.

- If you're a factory function, you'll instead call the underlying factory function with device='meta'.

- Convert the resulting meta tensor into a fake tensor, computing what the output device of the tensor should be (this is usually trivial, but sometimes it is not, e.g., cpu scalar promotion, or device-converting operations.)

API: the important bits
-----------------------

Non-PT2 usage (check out test/test_fake_tensor.py for more examples):

.. code:: python

    # Create a fake mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    fake_mode = FakeTensorMode()
    converter = fake_mode.fake_tensor_converter
    # Fakeify some real tensors
    fake_x = converter.from_real_tensor(fake_mode, x)
    with fake_mode:
        # Do some operations on the fake tensors
        fake_y = fake_x * 2
        # Factory operations automatically get fakeified in the context manager
        fake_z = torch.empty(20)

Q: Why do you have real tensors as inputs?

A: In a PT2 context, this is because you typically are compiling just-in-time, so for all the inputs to a graph you're compiling, you already have the "real" inputs, because you're compiling while you're executing the program.

PT2 pre-AOTAutograd usage (this is unusual, you probably don't want to do this):

.. code:: python


    # Fake mode is not enabled!
    from torch._guards import detect_fake_mode
    fake_mode = detect_fake_mode(args)
    # if fake_mode isn't None
    converter = fake_mode.fake_tensor_converter
    fake_args = [converter.from_real_tensor(fake_mode, arg) for arg in args]
    with fake_mode:
        ... # do stuff with the fake args, if needed ...

detect_fake_mode will search a number of locations to try to find "the" fake tensor mode associated with the lifecycle. Typically it will be pulled off of the tracing context.

PT2 post-AOTAutograd usage:

.. code:: python


    # Fake mode is enabled! example_inputs is typically fake already
    # TODO: we probably want to change this
    # Still do this to access fake mode
    fake_mode = detect_fake_mode(example_inputs)
    # But in general you don't have to turn it on

Other useful stuff:

.. code:: python

    from torch._subclasses.fake_tensor import unset_fake_temporarily
    with unset_fake_temporarily():
        ... # fake mode is disabled here, you can do real tensor compute

When might you want to disable fake tensor mode? Usually you don't want to do this. One niche case where we've found it useful is to implement constant propagation on fake tensors: in this case, we need to do some actual tensor computation even though we're in a fake tensor mode.

.. code:: python

    import FakeTensorProp from torch.fx.passes.fake_tensor_prop
    gm: GraphModule
    real_inputs: List[Tensor]
    FakeTensorProp(gm).propagate(*real_inputs)
    # This will populate meta['val'] on all the FX nodes with a fake tensor
    # or if you have a preexisting fake mode, you should use it
    FakeTensorProp(gm, mode=fake_mode).propagate(*real_inputs)
    # There is also propagate_dont_convert_inputs if your inputs are already fake
    fake_inputs: List[FakeTensor]
    FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)

Details
-------

Auto-convert or not?
Originally, FakeTensorMode would not automatically fakeify real tensors if you tried to do compute on them inside a FakeTensorMode region. The motivation behind this was to prevent the following footgun:

.. code:: python

    with FakeTensorMode():
        real_tensor.t_()

What should this code do? It would be surprising if we actually modified the metadata on the real tensor. But at the same time, there isn't any obvious opportunity to create a FakeTensor. So we conservatively decided to make this raise an error: "Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. Please convert all Tensors to FakeTensors first."

This error is pretty annoying in practice. For example, suppose you have a real nn.Module and you want to feed fake tensors through it. You need to somehow fakeify the nn.Module. This motivated FakeCopyMode.

Eventually, we gave up and added automatic fakeification. However, this is still not yet enabled by default in many uses of FakeTensorMode.

Metadata mutation on fake tensor
If you have a fake tensor, and you t_() it, the metadata on the fake tensor changes. This is reasonable on its face, but sometimes you want to also store fake tensors as metadata on FX nodes; mutating a fake tensor is bad because this will invalidate old metadata!

In fact, there is a fundamental tension here, which is that fake tensors maintain extremely accurate metadata about tensors, up to and including object identity. If object metadata changes over time in an FX graph, there is not actually any way to represent this change over time. Most of the time, our serious FX analyses are done on functionalized graphs, which don't have this, but occasionally you need to do an analysis on a non-functionalized graph. Maybe it was a mistake to put fake tensor in meta['val']

About the tensor subclass
-------------------------

Fake tensor uses both a subclass and a mode tensor subclass pattern, where FakeTensor.__torch_dispatch__ enables the FakeTensorMode associated with the fake tensor, and then redispatches (relying on FakeTensorMode to do the heavy lifting). If fake tensor operations get a subclass argument it doesn't recognize, it will return NotImplemented, giving the other subclass a chance to run first (hopefully desugaring into plain tensor operations), before it tries again. This can cause infinite loops.

How is each individual operator implemented?
--------------------------------------------

Unfortunately, there is a pretty complicated set of places where any given operator may be implemented. Some important cases to know about:

- Tensor subclasses support limited constant propagation if the number of elements is very small (this helps deal with some cases where we immediately call item() on such tensors.)
- We have some fastpath implementations for certain operators, which are done entirely in fake tensor, for performance reasons.
- If you use @custom_op to generate a custom tensor, these will register impl_abstract directly to fake tensor.
- Fake tensor itself has some hardcoded special cases for device-converting operations.
- If there is no meta implementation nor any decomposition, we will generate real zero-filled tensors and attempt to run the operator directly to find out what the results will be. This can cause segfaults if the operator attempts to do indexing with data, so we don't turn this on by default for custom ops.

How does the converter work?
----------------------------

Because fake tensors are used in situations that are very sensitive to the exact properties of a tensor, fake tensors do conversion very carefully, preserving leaf-ness, requires_grad'ness, aliasing, and a whole host of other properties. The bulk of the heavy lifting is in MetaConverter.

Performance characteristics
---------------------------

You would think fake tensors are fast because they don't do any tensor compute. But at small tensor sizes we are actually entirely overhead bound, and, well, fake tensor is in Python, and we often do a LOT of work to do a single tensor operation (because they are implemented as decompositions). So fake tensors are actually pretty slow in practice, especially when symbolic shapes are involved. There are two important fastpaths we currently have in fake tensor that make a big difference in practice:

- Pointwise ops don't go through PrimTorch decomps, instead we've hand-coded their propagation rule.
- If possible, we should.

Fake tensor of fake tensor?
----------------------------

There is interest in sending fake tensors as user inputs into the PT2 stack, which would imply we would need to be able to create a fake tensor of a fake tensor. This isn't really supported right now, but maybe it would not be too difficult to do.

Interaction with dynamic shapes
-------------------------------

Every FakeTensorMode contains a ShapeEnv, which tracks all symbolic shapes information. Their lifetimes are typically tied: they live and die together.

Because FakeTensorMode has a ShapeEnv (but meta implementations do not), meta functions that are data-dependent and require allocating an unbacked SymInt live in fake tensor. Fake tensor also takes care of memoizing unbacked SymInts, so that, e.g., if you call nonzero() on the same fake tensor twice, you get the same symbolic size.

Other resources
---------------

`Colab Tutorial On Using FakeTensor To Determine Max Batch Size <https://colab.research.google.com/drive/1zjAisRrc8R6uixKsrs1DRm3lwz5MWN68>`_
