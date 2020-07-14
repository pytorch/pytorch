
Serialization semantics
=======================

Storage sharing is preserved in serialization
---------------------------------------------

.. _preserve-storage-sharing:

PyTorch saves the underlying storages so that tensors sharing the same storage before :func:`torch.save`
will still share storage after :func:`torch.load`.

::

    >>> tensor = torch.zeros(1000000)
    >>> slice1 = tensor[:1000]
    >>> slice2 = tensor[:10] # slice1 and slice2 share the same storage
    >>> torch.save([slice1, slice2], 'share.pt')
    >>> loaded_1, loaded_2 = torch.load('share.pt')
    >>> loaded_1[0]
    tensor(0.)
    >>> loaded_2[0]
    tensor(0.)
    >>> loaded_2[0] = 1
    >>> loaded_1[0] # loaded tensors still share storage
    tensor(1.)

Note that saving storage instead of tensor itself means the serialized file size might not match tensor size.
In the example above the whole `tensor`'s storage (of size 1000000) is serialized instead of only slices.
When tensor is expanded from a smaller storage, serialized file size might be smaller than tensor size as well.

::

    >>> a = torch.zeros(4).expand(4, 4)
    >>> a.size()
    torch.Size([4, 4])
    >>> a.storage() # All columns of `a` share the same storage
     0.0
     0.0
     0.0
     0.0
    [torch.FloatStorage of size 4]
    >>> torch.save(a, 'a.pt')  # Only 4 float numbers are serialized
    >>> loaded = torch.load('a.pt')
    >>> loaded.storage()  # All colums of `loaded` share the same storage
     0.0
     0.0
     0.0
     0.0
    [torch.FloatStorage of size 4]

If saving storages causes issues like saved file contains a lot of unwanted data,
you can break the storage sharing before saving using :meth:`~torch.Tensor.clone`. But it might
produce different results compared to the original storage sharing version.

Best practices
--------------

.. _recommend-saving-models:

Recommended approach for saving a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two main approaches for serializing and restoring a model.

The first (recommended) saves and loads only the model parameters::

    torch.save(the_model.state_dict(), PATH)

Then later::

    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))

The second saves and loads the entire model::

    torch.save(the_model, PATH)

Then later::

    the_model = torch.load(PATH)

However in this case, the serialized data is bound to the specific classes
and the exact directory structure used, so it can break in various ways when
used in other projects, or after some serious refactors.

.. note::
    The 1.6 release of PyTorch switched ``torch.save`` to use a new
    zipfile-based file format. ``torch.load`` still retains the ability to
    load files in the old format. If for any reason you want ``torch.save``
    to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.
