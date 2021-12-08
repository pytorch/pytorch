.. _meta_init-doc:

.. currentmodule:: torch.nn.utils.meta_init

Large Module Materialization
============================
Large Model Materialization API enables constructing ``torch.nn.Module``
instances on the meta device for inspection purposes. It is meant to be used if
the size of a model is too big to fit on a single machine and you want to
inspect it without instantiating.

.. warning::
    This is an experimental feature and is subject to change. If you experience
    any issues, let us know by opening a GitHub issue.


Problem
-------
With ever increasing model sizes, it is becoming increasingly common for models
to exceed the memory capacity of a single machine. This means training such
models requires some partitioning strategy to distribute parts of the model onto
different machines. However the techniques (e.g. 3D parallelism, FSDP) used to
apply these strategies often need access to the model architecture to decide on
the optimal strategy and this represents a chicken-egg problem. Partitioning the
model requires first instantiating the model for inspection purposes and this
contradicts the whole purpose of partitioning.

Meta Initialization
-------------------
The :func:`meta_init` context manager is a non-intrusive API that enables users
to construct a ``torch.nn.Module`` on the meta device. Internally it forces all
parameters, buffers, and other tensors within its scope to use the meta device
regardless of their real device. The returned module(s) can be used for
inspection purposes and afterwards, if desired, be materialized in a
fine-grained way via :func:`materialize`.

Note though that the context manager uses a best effort algorithm and is not
guaranteed to succeed if a module's implementation does not support the meta
device. If you are the module author, you can use the :func:`is_meta_init()`
function to find out whether your module is being used within the scope of a
meta-init context and have an alternate logic if necessary. See
:ref:`common-failures` below for more information.

Code Examples
-------------
The most straightforward use case is to construct a module within the scope of
a :func:`meta_init` context and then later :func:`materialize` it after some
form of inspection.

::

    >>> import torch
    >>>
    >>> with torch.nn.utils.meta_init():
    ...     m = torch.nn.Linear(5, 1)
    >>> m.weight
    Parameter containing:
    tensor(..., device='meta', requires_grad=True)
    >>>
    >>> # Do some form of inspection (e.g. apply a sharding algorithm).
    >>>
    >>> torch.nn.utils.materialize(m)
    >>> m.weight
    Parameter containing:
    tensor([[-1.4677e+24, 4.5915e-41, 1.4013e-45, 0.0000e+00,
             -1.4677e+24, 4.5915e-41]], requires_grad=True)


You can also materialize just one or more submodules of a large model.

::

    >>> import torch
    >>>
    >>> with torch.nn.utils.meta_init():
    ...     m = MyVeryLargeModel()
    >>>
    >>> # Do some form of inspection (e.g. apply a sharding algorithm).
    >>>
    >>> # Only materialize `sublayer1` and `sublayer2`.
    >>> torch.nn.utils.materialize(m.sublayer1)
    >>> torch.nn.utils.materialize(m.sublayer2)


Note that :func:`meta_init` overrides even explicitly passed ``device``
arguments and forces the use of the meta device.

::

    >>> class MyModule(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.param = torch.nn.Parameter(torch.ones([3], device="cpu"))
    ...
    >>> with torch.nn.utils.meta_init():
    ...     m = MyModule()
    >>> m.param
    Parameter containing:
    tensor(..., device='meta', size=(10, 10), requires_grad=True)
    >>>
    >>> torch.nn.utils.materialize(m)
    >>> m.param
    Parameter containing:
    tensor([1., 1., 1.], requires_grad=True)


Although the main use case of :func:`meta_init` is large module materialization,
its underlying machinery is actually more flexible. You can also construct and
later materialize individual tensors.

::

    >>> with torch.nn.utils.meta_init():
    ...     t1 = torch.ones([3])
    ...     t2 = t1 + 2
    >>> t1
    tensor(..., device='meta', size=(3,))
    >>> t2
    tensor(..., device='meta', size=(3,))
    >>> # By default, `materialize()` clears the meta-init cache. If you want
    ... # to materialize more than one object, set the `keep_cache` parameter
    ... # to `True`.
    >>> torch.nn.utils.materialize(t1, keep_cache=True)
    >>> t1
    tensor([1., 1., 1.])
    >>> torch.nn.utils.materialize(t2)
    >>> t2
    tensor([3., 3., 3.])


API
---
.. autosummary::
    :toctree: generated
    :nosignatures:

    meta_init
    is_meta_init
    materialize
    clear_meta_init_cache

.. autofunction:: meta_init
.. autofunction:: is_meta_init
.. autofunction:: materialize
.. autofunction:: clear_meta_init_cache

.. _common-failures:

Common Failure Cases
--------------------
Since meta initialization relies on constructing modules and their tensors on
the meta device, there are some certain usage patterns that will fail when used
with :func:`meta_init`.

**A module that uses an operator not supported by the meta backend:**
If the module uses an operator that is not yet supported by the meta backend,
the operator call will fail. If you experience such failure, please open a
GitHub issue and let us know which operator you are using.

**A module that has device-specific initialization logic:**
A module that expects to be initialized using one or more specific device types
and includes initialization logic such as:

::

  def MyModule(Module):
      def __init__(self):
          x = torch.ones([10, 10])
          if x.is_cpu():
              y = x + 1
          elif x.is_cuda():
              y = x + 2


will silently fail, potentially with some partial initialization.

**A module that uses auxiliary state beyond its parameters and buffers:**
The :func:`materialize` function simply traverses through the parameters and
buffers of a module and materializes them. This means if the module has some
other auxiliary state those tensors won't be materialized.

**A module that directly accesses to raw tensor storage:**
If the module attempts to directly read from or write to the raw storage of a
tensor using a naked pointer without checking its validity, it will cause a
segmentation fault since meta tensors don't have any storage allocated to them.

**A tensor loaded from external data:**
Although not necessarily a failure, if a tensor is constructed from external
data (e.g. numpy), :func:`meta_init` won't intercept it and will allow the
tensor to be constructed with the actual data. Since the data is externally
owned and already loaded into memory, this is beyond :func:`meta_init`'s
control.

Mitigation Strategy
^^^^^^^^^^^^^^^^^^^
If you are the module author, you can use the :func:`is_meta_init()`
function to find out whether your module is being used within the scope of a
:func:`meta_init` context and have an alternate logic if necessary:

::

    class MyModule(Module):
        def __init__(self):
            super().__init__()

            if torch.nn.utils.is_meta_init():
                self.my_buffer = torch.empty([10,10], device="meta")
            else:
                self.my_buffer = load_my_buffer()

        # A function that does not support the meta backend.
        def load_my_buffer(self) -> Tensor:
            ...


Alternative Approach
--------------------
If you are a module author, an alternative to :func:`meta_init()` is to have an
explicit ``device`` parameter as part of your constructor (i.e. ``MyModule(device="meta")``).
However this approach is quite intrusive and requires all your submodules to
have a ``device`` parameter as well. In particular for existing large models
introducing a new constructor parameter can be prohibitively expensive.

:func:`meta_init` on the other hand does not require any code changes, but can
potentially fail to construct the module if the module initializes its state in a
way that conflicts with the meta device (e.g. if it uses an operator that does
not support the meta backend).

Relationship to ``torch.nn.utils.skip_init()``
--------------------------------------------------
Although they sound similar :func:`meta_init()` and :func:`torch.nn.utils.skip_init()`
serve different purposes. ``meta_init()`` is meant for large module
materialization and targets model users, while ``skip_init()`` is meant for
optimized module initialization and targets model authors.

Technically ``meta_init()`` constructs a module where all its parameters and
buffers reside on the meta device, while ``skip_init()`` returns a module where
some or all its parameters or buffers are empty tensors allocated on a real
device.
