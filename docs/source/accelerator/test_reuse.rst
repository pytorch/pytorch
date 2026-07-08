.. _accelerator-test-reuse:

Reusing PyTorch Test Cases for New Accelerator Backends
========================================================

PyTorch provides a large suite of built-in test cases that validate operator
correctness, numerical precision, and device behavior. When integrating a new
out-of-tree accelerator backend (e.g., a ``PrivateUse1``-based device), reusing
these existing test cases is strongly preferred over writing tests from scratch.

This guide explains the mechanisms available to new backend developers to
selectively run, skip, or modify PyTorch's built-in tests without modifying any
upstream PyTorch files.

.. note::

   These mechanisms are pragmatic short-term solutions intended to help new
   backends get test coverage quickly. The long-term direction is to incrementally
   migrate all relevant accelerator tests to be device-generic and remove
   device-specific decorators like ``@onlyCUDA`` from tests that should apply
   broadly.

Overview
--------

Three complementary mechanisms are available:

1. **bypass_device_restrictions** — Run tests gated by ``@onlyCUDA`` or other
   ``@onlyOn``-based decorators on your backend.
2. **op_skips / op_decorators** — Skip or modify operator-level tests generated
   via ``@ops`` on a per-backend basis.
3. **skipped_testcases** — Skip entire test classes or individual test methods
   during device-type test instantiation.

These mechanisms are all configured declaratively on your backend's
``DeviceTypeTestBase`` subclass, keeping all backend-specific logic in your own
code.

Bypassing ``@onlyOn`` Decorators
---------------------------------

Some PyTorch tests are decorated with ``@onlyCUDA`` or other ``@onlyOn``-based
decorators that restrict execution to specific device types. These tests are
often meaningful for new backends to run as well.

To bypass these restrictions, set ``bypass_device_restrictions = True`` on your
backend's test base class:

.. code-block:: python

   from torch.testing._internal.common_device_type import DeviceTypeTestBase

   class MyAcceleratorTestBase(DeviceTypeTestBase):
       device_type = "mydevice"
       bypass_device_restrictions = True

With this flag set, tests decorated with ``@onlyCUDA`` or similar decorators
will run on your backend instead of being skipped. No upstream PyTorch files
need to be modified.

This flag is defined on ``PrivateUse1TestBase`` and defaults to ``False``,
preserving existing behavior for all standard device types.

.. seealso::

   For a working example, see the ``openreg`` test backend in
   ``test/cpp_extensions/open_registration_extension/torch_openreg/tests/``.

Skipping Operator-Level Tests
------------------------------

PyTorch uses ``op_db`` and ``instantiate_device_type_tests`` to dynamically
generate large numbers of operator test cases via the ``@ops`` decorator. New
backends may not yet support every operator, or may have different numerical
precision characteristics.

Two class-level fields on ``DeviceTypeTestBase`` allow you to handle this:

- ``op_skips`` — skip specific operators entirely for your backend.
- ``op_decorators`` — attach additional decorators (e.g., ``skipIf``,
  ``toleranceOverride``) to specific operator tests for your backend.

.. code-block:: python

   from torch.testing._internal.common_device_type import (
       DeviceTypeTestBase,
       DecorateInfo,
       skipIf,
   )

   class MyAcceleratorTestBase(DeviceTypeTestBase):
       device_type = "mydevice"

       # Skip specific operators your backend does not yet support
       op_skips = {
           "torch.ops.aten.svd": "SVD not yet implemented on mydevice",
           "torch.ops.aten.fft_fft": "FFT not supported on mydevice",
       }

       # Apply additional decorators to specific operator tests
       op_decorators = {
           "torch.ops.aten.add": [
               DecorateInfo(
                   toleranceOverride({torch.float32: tol(atol=1e-3, rtol=1e-3)}),
                   "TestCommon",
                   "test_compare_cpu",
               )
           ],
       }

The keys in both dictionaries are the full operator names (``op.full_name``).
PyTorch merges these into the operator's existing decorator list during test
instantiation via ``DeviceTypeTestBase.update_op_list``.

Skipping Test Classes and Methods
-----------------------------------

Beyond operator-level skipping, you may need to skip entire test classes or
individual test methods that are not relevant or not yet supported for your
backend.

Use the ``skipped_testcases`` field on your test base class:

.. code-block:: python

   from torch.testing._internal.common_device_type import DeviceTypeTestBase

   class MyAcceleratorTestBase(DeviceTypeTestBase):
       device_type = "mydevice"

       skipped_testcases = {
           # Skip an entire test class
           "TestLinearAlgebra": "Linear algebra ops not yet supported",

           # Skip a specific test method within a class
           "TestCommon.test_noncontiguous_samples": "Known failure on mydevice",
       }

The keys follow the format ``"ClassName"`` to skip an entire class, or
``"ClassName.method_name"`` to skip a specific method. Skipping is applied
during ``instantiate_device_type_tests`` before tests are collected, so skipped
tests do not appear as failures in CI.

Putting It All Together
------------------------

A typical out-of-tree backend test setup using all three mechanisms looks like
this:

.. code-block:: python

   from torch.testing._internal.common_device_type import (
       DeviceTypeTestBase,
       DecorateInfo,
       instantiate_device_type_tests,
   )

   class MyAcceleratorTestBase(DeviceTypeTestBase):
       device_type = "mydevice"

       # Allow tests gated by @onlyCUDA to run on this backend
       bypass_device_restrictions = True

       # Skip unsupported operators
       op_skips = {
           "torch.ops.aten.svd": "Not yet implemented",
       }

       # Apply precision overrides for specific operators
       op_decorators = {
           "torch.ops.aten.add": [
               DecorateInfo(toleranceOverride(...), "TestCommon", "test_compare_cpu")
           ],
       }

       # Skip test classes or methods not applicable to this backend
       skipped_testcases = {
           "TestLinearAlgebra": "Not supported",
           "TestCommon.test_noncontiguous_samples": "Known issue",
       }

   # Instantiate PyTorch's built-in tests for your device
   from torch.testing._internal.common_methods_invocations import op_db
   from test_ops import TestCommon

   instantiate_device_type_tests(TestCommon, globals(), only_for="mydevice")

.. note::

   This page is updated continuously as the PyTorch testing infrastructure
   evolves. See the `PyTorch Test Refactoring project
   <https://github.com/orgs/pytorch/projects/154>`_ for ongoing work.

.. seealso::

   - :ref:`cpp-extension`
   - `Contributing to PyTorch <https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md>`_
