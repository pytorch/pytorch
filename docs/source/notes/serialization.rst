
Serialization semantics
=======================

This note describes how you can save and load PyTorch tensors and module states
in Python, and how to serialize Python modules so they can be loaded in C++.

.. contents:: Table of Contents

.. _saving-loading-tensors:

Saving and loading tensors
--------------------------

:func:`torch.save` and :func:`torch.load` let you easily save and load tensors:

::

    >>> t = torch.tensor([1., 2.])
    >>> torch.save(t, 'tensor.pt')
    >>> torch.load('tensor.pt')
    tensor([1., 2.])

By convention, PyTorch files are typically written with a ‘.pt’ or ‘.pth’ extension.

:func:`torch.save` and :func:`torch.load` use Python’s pickle by default,
so you can also save multiple tensors as part of Python objects like tuples,
lists, and dicts:

::

    >>> d = {'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])}
    >>> torch.save(d, 'tensor_dict.pt')
    >>> torch.load('tensor_dict.pt')
    {'a': tensor([1., 2.]), 'b': tensor([3., 4.])}

Custom data structures that include PyTorch tensors can also be saved if the
data structure is pickle-able.

.. _preserve-storage-sharing:

Saving and loading tensors preserves views
---------------------------------------------

Saving tensors preserves their view relationships:

::

    >>> numbers = torch.arange(1, 10)
    >>> evens = numbers[1::2]
    >>> torch.save([numbers, evens], 'tensors.pt')
    >>> loaded_numbers, loaded_evens = torch.load('tensors.pt')
    >>> loaded_evens *= 2
    >>> loaded_numbers
    tensor([ 1,  4,  3,  8,  5, 12,  7, 16,  9])

Behind the scenes, these tensors share the same "storage." See
`Tensor Views <https://pytorch.org/docs/main/tensor_view.html>`_ for more
on views and storage.

When PyTorch saves tensors it saves their storage objects and tensor
metadata separately. This is an implementation detail that may change in the
future, but it typically saves space and lets PyTorch easily
reconstruct the view relationships between the loaded tensors. In the above
snippet, for example, only a single storage is written to 'tensors.pt'.

In some cases, however, saving the current storage objects may be unnecessary
and create prohibitively large files. In the following snippet a storage much
larger than the saved tensor is written to a file:

::

    >>> large = torch.arange(1, 1000)
    >>> small = large[0:5]
    >>> torch.save(small, 'small.pt')
    >>> loaded_small = torch.load('small.pt')
    >>> loaded_small.storage().size()
    999

Instead of saving only the five values in the `small` tensor to 'small.pt,'
the 999 values in the storage it shares with `large` were saved and loaded.

When saving tensors with fewer elements than their storage objects, the size of
the saved file can be reduced by first cloning the tensors. Cloning a tensor
produces a new tensor with a new storage object containing only the values
in the tensor:

::

    >>> large = torch.arange(1, 1000)
    >>> small = large[0:5]
    >>> torch.save(small.clone(), 'small.pt')  # saves a clone of small
    >>> loaded_small = torch.load('small.pt')
    >>> loaded_small.storage().size()
    5

Since the cloned tensors are independent of each other, however, they have
none of the view relationships the original tensors did. If both file size and
view relationships are important when saving tensors smaller than their
storage objects, then care must be taken to construct new tensors that minimize
the size of their storage objects but still have the desired view relationships
before saving.

.. _saving-loading-python-modules:

Saving and loading torch.nn.Modules
-----------------------------------

See also: `Tutorial: Saving and loading modules <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_

In PyTorch, a module’s state is frequently serialized using a ‘state dict.’
A module’s state dict contains all of its parameters and persistent buffers:

::

    >>> bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    >>> list(bn.named_parameters())
    [('weight', Parameter containing: tensor([1., 1., 1.], requires_grad=True)),
     ('bias', Parameter containing: tensor([0., 0., 0.], requires_grad=True))]

    >>> list(bn.named_buffers())
    [('running_mean', tensor([0., 0., 0.])),
     ('running_var', tensor([1., 1., 1.])),
     ('num_batches_tracked', tensor(0))]

    >>> bn.state_dict()
    OrderedDict([('weight', tensor([1., 1., 1.])),
                 ('bias', tensor([0., 0., 0.])),
                 ('running_mean', tensor([0., 0., 0.])),
                 ('running_var', tensor([1., 1., 1.])),
                 ('num_batches_tracked', tensor(0))])

Instead of saving a module directly, for compatibility reasons it is recommended
to instead save only its state dict. Python modules even have a function,
:meth:`~torch.nn.Module.load_state_dict`, to restore their states from a state dict:

::

    >>> torch.save(bn.state_dict(), 'bn.pt')
    >>> bn_state_dict = torch.load('bn.pt')
    >>> new_bn = torch.nn.BatchNorm1d(3, track_running_stats=True)
    >>> new_bn.load_state_dict(bn_state_dict)
    <All keys matched successfully>

Note that the state dict is first loaded from its file with :func:`torch.load`
and the state then restored with :meth:`~torch.nn.Module.load_state_dict`.

Even custom modules and modules containing other modules have state dicts and
can use this pattern:

::

    # A module with two linear layers
    >>> class MyModule(torch.nn.Module):
          def __init__(self):
            super().__init__()
            self.l0 = torch.nn.Linear(4, 2)
            self.l1 = torch.nn.Linear(2, 1)

          def forward(self, input):
            out0 = self.l0(input)
            out0_relu = torch.nn.functional.relu(out0)
            return self.l1(out0_relu)

    >>> m = MyModule()
    >>> m.state_dict()
    OrderedDict([('l0.weight', tensor([[ 0.1400, 0.4563, -0.0271, -0.4406],
                                       [-0.3289, 0.2827, 0.4588, 0.2031]])),
                 ('l0.bias', tensor([ 0.0300, -0.1316])),
                 ('l1.weight', tensor([[0.6533, 0.3413]])),
                 ('l1.bias', tensor([-0.1112]))])

    >>> torch.save(m.state_dict(), 'mymodule.pt')
    >>> m_state_dict = torch.load('mymodule.pt')
    >>> new_m = MyModule()
    >>> new_m.load_state_dict(m_state_dict)
    <All keys matched successfully>


.. _serialized-file-format:

Serialized file format for ``torch.save``
-----------------------------------------

Since PyTorch 1.6.0, ``torch.save`` defaults to returning an uncompressed ZIP64
archive unless the user sets ``_use_new_zipfile_serialization=False``.

In this archive, the files are ordered as such

.. code-block:: text

    checkpoint.pth
    ├── data.pkl
    ├── byteorder  # added in PyTorch 2.1.0
    ├── data/
    │   ├── 0
    │   ├── 1
    │   ├── 2
    │   └── …
    └── version

The entries are as follows:
  * ``data.pkl`` is the result of pickling the object passed to ``torch.save``
    excluding ``torch.Storage`` objects that it contains
  * ``byteorder`` contains a string with the ``sys.byteorder`` when saving (“little” or “big”)
  * ``data/`` contains all the storages in the object, where each storage is a separate file
  * ``version`` contains a version number at save time that can be used at load time

When saving, PyTorch will ensure that the local file header of each file is padded
to an offset that is a multiple of 64 bytes, ensuring that the offset of each file
is 64-byte aligned.

.. note::
    Tensors on certain devices such as XLA are serialized as pickled numpy arrays. As
    such, their storages are not serialized. In these cases ``data/`` might not exist
    in the checkpoint.

.. _layout-control:

Layout Control
--------------

The ``mmap`` argument in :func:`torch.load` allows for lazy loading of tensor storages.

In addition, there are some advanced features that allow for more fine-grained
control and manipulation of a ``torch.save`` checkpoint.

The :class:`torch.serialization.skip_data` context manager enables
  * Saving a checkpoint with ``torch.save`` that includes empty space for data bytes
    to be written later.
  * Loading a checkpoint with ``torch.load`` and filling in the data bytes of tensors later.

To inspect tensor metadata in a ``torch.save`` checkpoint without allocating memory for storage
data, use ``torch.load`` within the ``FakeTensorMode`` context manager. On top of skipping loading
storage data similar to ``skip_data`` above, it additionally tags storages with their offset within
the checkpoint, enabling direct checkpoint manipulation.

.. code-block:: python

  import torch.nn as nn
  from torch._subclasses.fake_tensor import FakeTensorMode

  m = nn.Linear(10, 10)
  torch.save(m.state_dict(), "checkpoint.pt")

  with FakeTensorMode() as mode:
      fake_sd = torch.load("checkpoint.pt")

  for k, v in fake_sd.items():
      print(f"key={k}, dtype={v.dtype}, shape={v.shape}, stride={v.stride()}, storage_offset={v.storage_offset()}")
      # offset of the storage in the checkpoint
      print(f"key={k}, checkpoint_offset={v.untyped_storage()._checkpoint_offset}")

For more information, `this tutorial <https://docs.pytorch.org/tutorials/prototype/gpu_direct_storage.html>`_
offers a comprehensive example of using these features to manipulate a checkpoint.


.. _weights-only:

``torch.load`` with ``weights_only=True``
-----------------------------------------

Starting in version 2.6, ``torch.load`` will use ``weights_only=True`` if the ``pickle_module``
argument is not passed.

.. _weights-only-security:

weights_only security
^^^^^^^^^^^^^^^^^^^^^

As discussed in the documentation for :func:`torch.load`, ``weights_only=True`` restricts
the unpickler used in ``torch.load`` to only executing functions/building classes required for
``state_dicts`` of plain ``torch.Tensors`` as well as some other primitive types. Further,
unlike the default ``Unpickler`` provided by the ``pickle`` module, the ``weights_only`` Unpickler
is not allowed to dynamically import anything during unpickling.

``weights_only=True`` narrows the surface of remote code execution attacks but has the following limitations:

1. ``weights_only=True`` does not guard against denial of service attacks.
2. We try to prevent memory corruptions during ``torch.load(weights_only=True)`` but they might still be possible.

Note that even if memory corruption does not occur during ``torch.load`` itself, loading CAN create
unexpected objects for the downstream code that can also lead to memory corruption (e.g. a Tensor of
indices and values made to a sparse Tensor in user code might write/read out of bounds).

.. _weights-only-allowlist:

weights_only allowlist
^^^^^^^^^^^^^^^^^^^^^^

As mentioned above, saving a module's ``state_dict`` is a best practice when using ``torch.save``. If loading an old
checkpoint that contains an ``nn.Module``, we recommend ``weights_only=False``. When loading a checkpoint that contains
tensor subclasses, there will likely be functions/classes that need to be allowlisted, see below for further details.

If the ``weights_only`` Unpickler encounters a function or class that is not allowlisted
by default within the pickle file, you should see an actionable error like such

.. code::

    _pickle.UnpicklingError: Weights only load failed. This file can still be loaded,
    to do so you have two options, do those steps only if you trust the source of the checkpoint.
        1. Re-running `torch.load` with `weights_only` set to `False` will likely succeed,
            but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        2. Alternatively, to load with `weights_only=True` please check the recommended
           steps in the following error message.
           WeightsUnpickler error: Unsupported global: GLOBAL {__module__}.{__name__} was not an allowed global by
           default. Please use `torch.serialization.add_safe_globals([{__name__}])` or the
           `torch.serialization.safe_globals([{__name__}])` context manager to allowlist this global
           if you trust this class/function.

Please follow the steps in the error message and allowlist the functions or classes only if you trust them.

To get all GLOBALs (functions/classes) in the checkpoint that are not yet allowlisted you can use
:func:`torch.serialization.get_unsafe_globals_in_checkpoint` which will return a list of strings of the form
``{__module__}.{__name__}``. If you trust these functions/classes, you can import them and allowlist them per
the error message either via :func:`torch.serialization.add_safe_globals` or the context manager
:class:`torch.serialization.safe_globals`.

To access the list of user-allowlisted functions/classes you can use :func:`torch.serialization.get_safe_globals` and
to clear the current list see :func:`torch.serialization.clear_safe_globals`.

Troubleshooting ``weights_only``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Getting unsafe globals
""""""""""""""""""""""

A caveat is that :func:`torch.serialization.get_unsafe_globals_in_checkpoint` analyzes the checkpoint statically,
some types might be built dynamically during the unpickling process and hence will not be reported by
:func:`torch.serialization.get_unsafe_globals_in_checkpoint`. One such example is ``dtypes`` in numpy. In
``numpy < 1.25`` after allowlisting all the functions/classes reported by
:func:`torch.serialization.get_unsafe_globals_in_checkpoint` you might see an error like

.. code::

    WeightsUnpickler error: Can only build Tensor, Parameter, OrderedDict or types allowlisted via `add_safe_globals`,
    but got <class 'numpy.dtype[float32]'>

This can be allowlisted via ``{add_}safe_globals([type(np.dtype(np.float32))])``.

In ``numpy >=1.25`` you would see

.. code::

    WeightsUnpickler error: Can only build Tensor, Parameter, OrderedDict or types allowlisted via `add_safe_globals`,
    but got <class 'numpy.dtypes.Float32DType'>

This can be allowlisted via ``{add_}safe_globals([np.dtypes.Float32DType])``.

Environment Variables
"""""""""""""""""""""

There are two environment variables that will influence the behavior of ``torch.load``. These can be helpful
if one does not have access to the ``torch.load`` callsites.

* ``TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`` will override all ``torch.load`` callsites to use ``weights_only=True``.
* ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` will make ``torch.load`` callsites use ``weights_only=False`` **only**
  if ``weights_only`` was not passed as an argument.


.. _utility functions:

Utility functions
-----------------

The following utility functions are related to serialization:

.. currentmodule:: torch.serialization

.. autofunction:: register_package
.. autofunction:: get_crc32_options
.. autofunction:: set_crc32_options
.. autofunction:: get_default_load_endianness
.. autofunction:: set_default_load_endianness
.. autofunction:: get_default_mmap_options
.. autofunction:: set_default_mmap_options
.. autofunction:: add_safe_globals
.. autofunction:: clear_safe_globals
.. autofunction:: get_safe_globals
.. autofunction:: get_unsafe_globals_in_checkpoint
.. autoclass:: safe_globals
.. autoclass:: skip_data

.. _serialization config:

Config
------
.. py:module:: torch.utils.serialization
.. py:module:: torch.utils.serialization.config

``torch.utils.serialization.config`` provides a global config that can control the behavior of
``torch.save`` and ``torch.load``.


``torch.utils.serialization.config.save`` contains options that control the behavior of ``torch.save``.

  * ``compute_crc32``: whether to compute and write the zip file checksum (Default : ``True``).
    See :func:`~torch.serialization.set_crc32_options`.
  * ``use_pinned_memory_for_d2h``: for storages that are on an accelerator when passed to ``torch.save``, whether to
    move storage to pinned memory or pageable memory on CPU within ``torch.save``. (Default: ``False`` (i.e. pageable))
  * ``storage_alignment``: alignment of storages in the checkpoint during ``torch.save`` in bytes. (Default ``64``)

``torch.utils.serialization.config.load`` contains options that control the behavior of ``torch.load``.

  * ``mmap``: See the documentation for ``mmap`` argument in :func:`torch.load`.
    This config will set the behavior of ``mmap`` for ``torch.load`` if it is not
    already explicitly passed to the ``torch.load`` call (Default : ``False``).
  * ``endianness``: See :func:`~torch.serialization.set_default_load_endianness`.
    (Default : ``torch.serialization.LoadEndianness.NATIVE``)
  * ``mmap_flags``: See :class:`~torch.serialization.set_default_mmap_options`.
    (Default : ``MAP_PRIVATE``)
  * ``calculate_storage_offsets``: If this config is set to ``True``, offsets for storages will be
    calculated rather than read via random reads when using ``torch.load(mmap=True)``. This minimizes
    random reads, which can be helpful when the file is being loaded over a network. (Default : ``False``)
