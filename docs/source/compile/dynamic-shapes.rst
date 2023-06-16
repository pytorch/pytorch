Dynamic shapes
==============

Code: `symbolic_shapes.py <https://github.com/pytorch/pytorch/blob/db4572dbf18f1cf50cf662547e272d3117063747/torch/fx/experimental/symbolic_shapes.py>`_

See also: `The dynamic shapes manual <https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng>`_

Motivation
----------

Deep learning compilers commonly only work for static shapes, that is to say, they produced compiled programs which only work for a single specific configuration of input shapes, and must recompile if any input shape changes. This assumption works great for the majority of commonly run deep learning models today, but there are a few situations where it is insufficient:

- Some dimensions, such as batch size or sequence length, may vary. For example, an inference service performing adaptive batching will execute inference requests with varying batch sizes depending on how many requests it received within its batching window. We may also want to consider padding out variable size sequences only to the maximum sequence length within a batch, which may vary from batch-to-batch.
- Some models exhibit data-dependent output shapes, that is to say, the size of their outputs and intermediates may depend on the actual input data which may vary across runs. For example, detection models may first generate a variable number of potential bounding boxes before running a more expensive image recognition model to identify if the subject is in a bounding box. The number of bounding boxes is data dependent.
- One particularly important case of data-dependent shapes occurs when dealing with sparse representations, such as sparse tensors, jagged tensors, and graph neural networks. In all of these cases, the amount of data to be processed depends on the sparse structure of the problem, which will typically vary in a data-dependent way.

In supporting dynamic shapes, we chose not to support dynamic rank programs, e.g., programs whose inputs tensors change in dimensionality, as this pattern rarely occurs in real-world deep learning programs, and it avoids the need to reason inductively over symbolic lists of shapes.

Abridged public API
-------------------

The eventual plan:

- PT2 assumes everything is static by default
- If we recompile because a size changed, we will instead attempt to recompile that size as being dynamic (so we will never recompile because of that size again)
- If you know ahead of time something will be dynamic, you can skip the first recompile with ``torch._dynamo.mark_dynamic(tensor, dim)``
- If you say ``torch.compile(dynamic=True)`` we will attempt to make as much dynamic as possible

Unbacked integers for eager mode:

What we have currently:

- You must explicitly opt into dynamic shapes with ``torch._dynamo.config.automatic_dynamic_shapes = True`` or ``torch.compile(dynamic=True)``
- ``torch.compile(dynamic=True)`` proactively attempts to make everything dynamic
- ``torch._dynamo.config.automatic_dynamic_shapes`` will assume everything is
  static, but if we recompile because a size varied, the next time we will try
  to compile it dynamically
- ``torch._dynamo.mark_dynamic`` works

Use ``TORCH_LOGS=dynamic`` to view more information about what is going on with dynamic shapes.

The Guard Model
---------------

When considering how to add support for dynamic shapes to TorchDynamo and TorchInductor, we made a major design decision: in order to reuse decompositions and other preexisting code written in Python/C++ targeting the PyTorch API, we must be able to trace through dynamic shapes. Unlike a fully symbolic system which might capture both branches of a conditional, we always pick one branch and specialize our trace under the assumption that we only use this trace when we would have made the same choice for that branch in the future. To do this, we maintain a "hint" for every symbolic size saying what its concrete value is at compile time (as TorchDynamo is a just-in-time compiler, it always knows what the actual input sizes are.) When we perform a condition on a tensor, we simply consult the hint to find out which branch to take.

This greatly simplifies the symbolic shape formulas we produce, but means we have a much more involved system for managing guards. Consider, for example, the following program:

.. code-block:: python

    def f(x, y):
        z = torch.cat([x, y])
        if z.size(0) > 2:
            return z.mul(2)
        else:
            return z.add(2)

The final IR we will compile with TorchInductor will either be ``torch.cat([x, y]).add(2)`` or ``torch.cat([x, y]).mul(2)`` (with the condition flattened away), but to determine which branch we are in, we would need to know the size of ``z``, an intermediate. Because TorchDynamo must know upfront if a compiled trace is valid (we do not support bailouts, like some JIT compilers), we must be able to reduce ``z.size(0)`` as an expression in terms of the inputs, ``x.size(0) + y.size(0)``. This is done by writing meta functions for all operators in PyTorch which can propagate size information to the output of a tensor without actually performing computation on the node.

Overall architecture
--------------------

Symbolic shapes workflow:

1. When we start compiling a frame in Dynamo, we allocate a ShapeEnv (attached to FakeTensorMode) which keeps track of symbolic shapes state.
2. We allocate symbolic sizes for tensors on entry (what is static or dynamic is a policy decision, with some knobs).
3. We propagate the symbolic sizes through operators, maintaining both (1) FX IR so that we can faithfully export symbolic compute, and (2) Sympy expressions representing the size vars, so we can reason about them.
4. When we condition on symbolic sizes, either in Dynamo tracing or in Inductor optimization, we add guards based on the conditional. These can be induced from both Python and C++.
5. These guards can induce further simplifications on symbolic variables. For example, if you assert ``s0 == 4``, we can now replace all occurrences of ``s0`` with ``4``.
6. When we're done tracing and optimizing, we install all of these guards with the compiled code; the compiled code is only reusable if all the guards evaluate true.

Important files:

- C++ SymInt API: ``c10/core/SymInt.h``, ``SymFloat.h``, ``SymBool.h``
- Python SymInt API: ``torch/__init__.py`` (look for ``SymInt/SymFloat/SymBool``)
- C++ plumbing: ``c10/core/SymNodeImpl.h``, ``torch/csrc/utils/python_symnode.h``, ``torch/csrc/jit/python/init.cpp``
- Python infrastructure: ``torch/fx/experimental/symbolic_shapes.py``
- Other important files: ``torch/_subclasses/fake_tensor.py``, ``torch/_meta_registrations.py``, decomps, PrimTorch refs

Abridged internal API
---------------------

Understanding the Python class hierarchy:

- SymInt/SymFloat/SymBool: these are user-visible classes that simulate their int/float/bool counterparts. If you add two SymInts, we give you a new SymInt that symbolically tracks that the integer addition had occurred.
- SymNode: this is the internal structure (accessible via e.g., ``symint.node``) which holds the actual symbolic tracking info. SymNode is type erased; this makes it more convenient to represent mixed-type operations. Note that technically you don't have to call into Python SymNode from SymInt; for example, XLA's C++ ``SymNodeImpl`` would take the place of SymNode.
- ShapeEnv: per-compile context state which keeps track of all the free symbols and guards we have accumulated so far. Every SymNode records its ShapeEnv (but not vice versa; SymNodes only get used if they participate in a guard).

C++ is fairly similar:

- c10::SymInt/SymFloat/SymBool: user-visible classes that simulate int/float/bool.
- c10::SymNode/SymNodeImpl: analogous to SymNode
- There is no ShapeEnv in C++; for ease of debugging, the entire symbolic reasoning apparatus is in Python.

When you write code that is traceable with ``make_fx``, it must be able to deal with SymInt/SymFloat/SymBool flowing through it. `The dynamic shapes manual <https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng>`_ gives some guidance for how to do this.

DimDynamic policy
-----------------

Symbolic reasoning:

- Value ranges
- Sympy usage notes
- Constraints
- DimDynamic/Constraint

Unbacked SymInts
----------------

To resolve control flow, we check the hint, aka actual value, of a symbolic integer to determine which branch to go. However, in some cases, we may not have a hint: so-called unbacked symbolic integers arise when a size variable emerges from a data-dependent operation like ``.nonzero()`` or ``.item()``. It is illegal to perform control flow on these symbolic integers, so we must graph break on these operations.

Naively implemented, this is too restrictive: most PyTorch programs will immediately fail if you try to do anything with unbacked symbolic integers. Here are the most important enhancements to make this actually work:

- On tensor creation, PyTorch precomputes a lot of data about a tensor; for example, if you use ``empty_strided`` to create a tensor, we will eagerly sort the strides and determine if the tensor is non-overlapping and dense. Sorts produce a lot of guards. However, it is more common to produce a tensor directly with a higher-level API like ``empty``, which is guaranteed to produce a non-overlapping and dense tensor. We modified PyTorch to avoid needlessly recomputing these properties.
- Even if nontrivial compute is needed, sometimes a property is never actually queried at all. Making these precomputed properties lazy allows us to avoid guarding on an unbacked symbolic integer unless it is actually needed.
- The data in an integer tensor is generally not known to be non-negative. However, we provide an API ``constrain_range`` whereby a user can specify that a size is bounded above and below by known limits.
