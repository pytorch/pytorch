Library Registration Macros
===========================

These macros provide stable ABI equivalents of the standard PyTorch operator
registration macros (``TORCH_LIBRARY``, ``TORCH_LIBRARY_IMPL``, etc.).
Use these when building custom operators that need to maintain binary
compatibility across PyTorch versions.

STABLE_TORCH_LIBRARY
--------------------

.. c:macro:: STABLE_TORCH_LIBRARY(ns, m)

   Defines a library of operators in a namespace using the stable ABI.

   This is the stable ABI equivalent of :c:macro:`TORCH_LIBRARY`.
   Use this macro to define operator schemas that will maintain
   binary compatibility across PyTorch versions. Only one ``STABLE_TORCH_LIBRARY``
   block can exist per namespace; use ``STABLE_TORCH_LIBRARY_FRAGMENT`` for
   additional definitions in the same namespace from different translation units.

   :param ns: The namespace in which to define operators (e.g., ``mylib``).
   :param m: The name of the StableLibrary variable available in the block.

   **Example:**

   .. code-block:: cpp

      STABLE_TORCH_LIBRARY(mylib, m) {
          m.def("my_op(Tensor input, int size) -> Tensor");
          m.def("another_op(Tensor a, Tensor b) -> Tensor");
      }

   Minimum compatible version: PyTorch 2.9.

STABLE_TORCH_LIBRARY_IMPL
-------------------------

.. c:macro:: STABLE_TORCH_LIBRARY_IMPL(ns, k, m)

   Registers operator implementations for a specific dispatch key using the stable ABI.

   This is the stable ABI equivalent of ``TORCH_LIBRARY_IMPL``. Use this macro
   to provide implementations of operators for a specific dispatch key (e.g.,
   CPU, CUDA) while maintaining binary compatibility across PyTorch versions.

   .. note::

      All kernel functions registered with this macro must be boxed using
      the ``TORCH_BOX`` macro.

   :param ns: The namespace in which the operators are defined.
   :param k: The dispatch key (e.g., ``CPU``, ``CUDA``).
   :param m: The name of the StableLibrary variable available in the block.

   **Example:**

   .. code-block:: cpp

      STABLE_TORCH_LIBRARY_IMPL(mylib, CPU, m) {
          m.impl("my_op", TORCH_BOX(&my_cpu_kernel));
      }

      STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
          m.impl("my_op", TORCH_BOX(&my_cuda_kernel));
      }

   Minimum compatible version: PyTorch 2.9.

STABLE_TORCH_LIBRARY_FRAGMENT
-----------------------------

.. c:macro:: STABLE_TORCH_LIBRARY_FRAGMENT(ns, m)

   Extends operator definitions in an existing namespace using the stable ABI.

   This is the stable ABI equivalent of ``TORCH_LIBRARY_FRAGMENT``. Use this macro
   to add additional operator definitions to a namespace that was already
   created with ``STABLE_TORCH_LIBRARY``.

   :param ns: The namespace to extend.
   :param m: The name of the StableLibrary variable available in the block.

   Minimum compatible version: PyTorch 2.9.

TORCH_BOX
---------

.. c:macro:: TORCH_BOX(func)

   Wraps a function to conform to the stable boxed kernel calling convention.

   This macro takes an unboxed kernel function pointer and generates a boxed wrapper
   that can be registered with the stable library API.

   :param func: The unboxed kernel function to wrap.

   **Example:**

   .. code-block:: cpp

      Tensor my_kernel(const Tensor& input, int64_t size) {
          return input.reshape({size});
      }

      STABLE_TORCH_LIBRARY_IMPL(my_namespace, CPU, m) {
          m.impl("my_op", TORCH_BOX(&my_kernel));
      }

   Minimum compatible version: PyTorch 2.9.
