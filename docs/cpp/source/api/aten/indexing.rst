Tensor Indexing
===============

The PyTorch C++ API provides tensor indexing similar to Python. Use
``torch::indexing`` namespace for index types:

.. code-block:: cpp

   using namespace torch::indexing;

Getter Operations
-----------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Python
     - C++
   * - ``tensor[None]``
     - ``tensor.index({None})``
   * - ``tensor[...]``
     - ``tensor.index({Ellipsis})`` or ``tensor.index({"..."})``
   * - ``tensor[1, 2]``
     - ``tensor.index({1, 2})``
   * - ``tensor[1::2]``
     - ``tensor.index({Slice(1, None, 2)})``
   * - ``tensor[torch.tensor([1, 2])]``
     - ``tensor.index({torch::tensor({1, 2})})``

Setter Operations
-----------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Python
     - C++
   * - ``tensor[1, 2] = 1``
     - ``tensor.index_put_({1, 2}, 1)``
   * - ``tensor[1::2] = 1``
     - ``tensor.index_put_({Slice(1, None, 2)}, 1)``

Slice Syntax
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Python
     - C++
   * - ``:`` or ``::``
     - ``Slice()`` or ``Slice(None, None)``
   * - ``1:``
     - ``Slice(1, None)``
   * - ``:3``
     - ``Slice(None, 3)``
   * - ``1:3``
     - ``Slice(1, 3)``
   * - ``1:3:2``
     - ``Slice(1, 3, 2)``
   * - ``::2``
     - ``Slice(None, None, 2)``
