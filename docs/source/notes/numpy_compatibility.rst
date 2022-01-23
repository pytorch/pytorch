.. _numpy_compatibility:

NumPy Compatibility
=====================

.. note::

   **TL;DR:**

   - Be explicit about obtaining and using NumPy arrays, or
   - Avoid operators (``-``, ``+``, ``*``, ``/``) in favor of methods
     (``np.subtract``, ``np.add``, ``np.multiply``, ``np.divide``)
   - `This tutorial`_ shows a more concrete example of interoperability between
     NumPy and PyTorch

At the onset it should be stressed that PyTorch tensors and NumPy arrays are
fundamentally different types. However, in the interests of being accessible to
the wider scientific python ecosystem, there are implicit and explicit
conversion methods. The major considerations to keep in mind are:

- **Where** is the structure allocated?
    * This may be on a compute device (e.g. a GPU) or the CPU
- **What** is the underlying data?
    * In particular, several functions will either return a copy or a view
- **How** are modifications to the data handled?
    * The data returned may be read-only, or may be writable

From the end-user perspective, it is most desirable to have zero copy methods,
however, for exploratory analyses and their visualizations in particular, it is
often more important to have guaranteed conversions, even if copies are created.

Basic interoperability
------------------------

NumPy arrays have no conception of the GPU, and so all conversions require a
``cpu`` tensor. Similarly, the **computational graph** cannot be represented in
```numpy``, and so the pre-requisite to have an operation or object without one,
i.e. only leaf nodes are supported.

.. code-block:: python

   import torch
   b_torch = torch.ones(3)
   assert(b_torch.requires_grad == False)
   assert(b_torch.is_cuda == False)
   b_torch.numpy() # Fine, will work

The interoperability guidelines places a strong emphasis on correctness over
convenience. As the conversion to a NumPy array may potentially be a lossy
operation (the computational graph), there are no inbuilt operations to do
something which unconditionally returns a NumPy array.

.. code-block:: python

   # Never do this
   def uncond_numpy(torch_tensor):
        if torch_tensor.is_cuda:
           warnings.warn("Moving back to CPU")
           torch_tensor = torch_tensor.cpu()
        if torch_tensor.requires_grad == False:
            return torch_tensor.numpy()
        else:
            warnings.warn("Calling detach, probably lossy!")
            return torch_tensor.detach().numpy()

Operations
------------

All ``torch`` operators will helpfully fail with a ``TypeError`` if called with
``numpy`` arrays. However, for **numpy methods**, using a ``torch.Tensor`` with
an ``np.ndarray`` will return a ``torch.Tensor`` without creating any additional
copies.

 - Due to the :meth:`torch.Tensor.__array__()` implementation, a
   ``np.ndarray`` which shares memory with the ``torch.Tensor`` is used for the
   operation.
 - The return type functionality is defined by
   :meth:`torch.Tensor.__array_wrap__()`, and calls ``torch.from_numpy()``
   internally.

.. code-block:: python

   import torch
   import numpy as np
   a_np = np.ones(3) # dtype=float64
   b_torch = torch.ones(3) # dtype=float32
   torch.add(a_np, b_torch) # TypeError
   b_torch + a_np # TypeError
   a_np + b_torch # OK but awkward
   c_tmp = np.add(a_np, b_torch) # OK
   # c_tmp.dtype is torch.float64

Since the ``torch.Tensor`` methods will not work with ``np.ndarray`` objects, to
prevent the apparent asymmetry of ``a_np * b_torch`` passing while ``b_torch *
a_np`` fails, the user would **do best** to use NumPy functions instead:

.. csv-table::
   :header: Operator, NumPy Function, Description

   "``+``", "``np.add()``", "Addition"
   "``-``", "``np.subtract()``", "Subtraction"
   "``*``", "``np.multiply()``", "Multiplication"
   "``/``", "``np.divide()``", "Division"

Conversions
-------------

Only a small subset of data type (``dtype``) objects defined in NumPy have an
equivalent in PyTorch, namely:

.. csv-table:: $ indicates the sizes supported, e.g. ``uint8``
   :header: ``np.dtype``, ``torch.dtype``, sizes

    "``bool_``", "``bool``", "N/A"
    "``uint$``", "``uint$``", ":math:`8`"
    "``int$``", "``int$``", ":math:`8, 16, 32, 64`"
    "``float$``", "``float$``", ":math:`16, 32, 64`"
    "``complex$``", "``complex$``", ":math:`64, 128`"

.. warning::

   The promotion and casting rules `of NumPy`_ differ from those `of PyTorch`_.

To ``numpy``
^^^^^^^^^^^^^

The :meth:`torch.numpy()` method  and the :doc:`np.asarray()
<numpy:reference/generated/numpy.asarray>` function returns a **view** of the
underlying tensor as a ``np.ndarray`` object.

.. code-block:: python

    b_torch = torch.ones(3)
    b_torch.numpy()[2] = 32
    # b_torch is tensor([ 1.,  1., 32.])
    a_np = np.array([1, 1, 32], dtype = np.float32)
    np.array_equal(b_torch.numpy() == a_np) # True
    c_tmp = np.asarray(b_torch, dtype = np.float32) # No copy if same dtype
    # c_tmp is array([1., 1., 32.], dtype=float32)
    c_tmp[2] = 1.
    # b_torch is tensor([1., 1., 1.])

.. note::

   Since ``np.asarray()`` depends on the implementation of
   ``torch.Tensor.__array__()`` which calls ``torch.numpy()``, the **leaf node**
   requirement still needs to be satisfied by the user, i.e., ``requires_grad ==
   False``

From ``numpy``
^^^^^^^^^^^^^^^^^

To obtain a **view** of the data, :meth:`torch.from_numpy()` can be used.

.. code-block:: python

   import torch
   import numpy as np
   a_np = np.array([1, 2, 3], dtype = np.float64)
   b_torch = torch.from_numpy(a_np)
   # b_torch = torch.as_tensor(a_np) # see note
   b_torch[2] = 23
   assert(b_torch.numpy() == a_np) # True

- :meth:`torch.from_numpy()` is guaranteed to share memory with NumPy.
- :meth:`torch.as_tensor()` will try to stay away from copy operations, it
  also has the effect of sharing memory. However, ``torch.as_tensor()`` has
  slightly higher overhead as it checks and accepts other iteratable objects as
  well, e.g. ``list`` objects.
- :meth:`torch.from_dlpack()` called with a NumPy array as its argument will also generate a ``torch.Tensor`` view.

To obtain a **copy** of the ``ndarray`` object, and not share memory, the
:meth:`torch.tensor()` constructor accepts :meth:`np.ndarray` objects as a data
source to construct and return a ``torch.Tensor``.

.. note::

   Recall that, if ``x`` is a tensor, ``torch.tensor(x)`` is equivalent to
    ``x.clone().detach()``.

.. code-block:: python

   import torch
   import numpy as np
   a_np = np.array([1, 2, 3], dtype = np.float64)
   b_torch = torch.tensor(a_np)
   # b_torch.dtype is torch.float64
   b_torch[2] = 23
   assert(b_torch.numpy() == a_np) # Will fail

Special considerations apply when calling NumPy functions on PyTorch tensors.
Where possible, the equivalent PyTorch function is called.

.. note::

   This is due to the fact that ``torch.Tensor.__array_priority__`` is higher
   than the NumPy default of ``0``.

Indexing
^^^^^^^^^^^^

For the most part, both simple and "fancy" indexing work as expected. Edge cases
are enumerated here.

Negative strides
~~~~~~~~~~~~~~~~~

NumPy arrays may have negative strides, which is not true for PyTorch tensors.

.. code-block:: python

   import torch
   import numpy as np
   a_np = np.array([1, 2, 3])
   b_torch = torch.from_numpy(a_np[::-1]) # Error
   b_torch = torch.from_numpy(a_np[::-1].copy()) # Works
   b_torch = torch.from_numpy(np.ascontiguousarray(a_np[::-1])) # Works

.. _`of Numpy`: https://numpy.org/devdocs/reference/generated/numpy.promote_types.html#numpy.promote_types
.. _`of PyTorch`: https://pytorch.org/docs/stable/tensor_attributes.html
.. _`This tutorial`: https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
