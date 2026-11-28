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
2. **op_overrides** — Skip or modify operator-level tests generated via ``@ops``
   on a per-backend basis.
3. **test_exclusions** — Skip entire test classes or individual test methods
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

This flag defaults to ``False`` on ``DeviceTypeTestBase``, preserving existing
behavior for all standard device types.

.. seealso::

   For a working example, see the ``openreg`` test backend in
   ``test/cpp_extensions/open_registration_extension/torch_openreg/tests/``.

Overriding Operator-Level Tests
---------------------------------

PyTorch uses ``op_db`` and ``instantiate_device_type_tests`` to dynamically
generate large numbers of operator test cases via the ``@ops`` decorator. New
backends may not yet support every operator, or may have different numerical
precision characteristics.

The ``op_overrides`` class-level field on ``DeviceTypeTestBase`` allows you to
attach additional decorators (e.g., ``unittest.skip``, ``unittest.expectedFailure``,
``precisionOverride``) to specific operator tests for your backend:

.. code-block:: python

   import unittest
   from torch.testing._internal.common_device_type import (
       DeviceTypeTestBase,
       DecorateInfo,
       precisionOverride,
   )

   class MyAcceleratorTestBase(DeviceTypeTestBase):
       device_type = "mydevice"

       op_overrides = {
           # Skip an operator entirely
           "op_skip": [DecorateInfo(unittest.skip("not supported on mydevice"))],

           # Skip for a specific dtype only
           "op_skip_f32": [
               DecorateInfo(unittest.skip("f32 not supported"), dtypes=(torch.float32,))
           ],

           # Mark as expected failure
           "op_xfail": [DecorateInfo(unittest.expectedFailure)],

           # Override numerical precision
           "op_precision": [
               DecorateInfo(precisionOverride({torch.float32: 1e-2}))
           ],
       }

The keys in ``op_overrides`` are operator names as they appear in ``op_db``.
PyTorch merges these into the operator's existing decorator list during test
instantiation via ``DeviceTypeTestBase._apply_op_overrides``.

Skipping Test Classes and Methods
-----------------------------------

Beyond operator-level overrides, you may need to skip entire test classes or
individual test methods that are not relevant or not yet supported for your
backend.

Use the ``test_exclusions`` field on your backend's test base class:

.. code-block:: python

   from torch.testing._internal.common_device_type import (
       DeviceTypeTestBase,
       instantiate_device_type_tests,
   )

   # Define exclusions directly on the base class
   PrivateUse1TestBase.test_exclusions = {
       # Skip specific methods within a class
       "TestSkippedSpecificTestCases": ["test_skipped", "test_other"],

       # Skip an entire test class using the wildcard "*"
       "TestSkippedWholeTestClass": "*",
   }

The format is a dictionary where:

- Keys are test class names (as strings).
- Values are either a list of method names to skip, or ``"*"`` to skip the
  entire class.

Skipping is applied during ``instantiate_device_type_tests`` before tests are
collected, so skipped tests do not appear as failures in CI.

Putting It All Together
------------------------

A typical out-of-tree backend test setup using all three mechanisms looks like
this:

.. code-block:: python

   import unittest
   from torch.testing._internal.common_device_type import (
       DeviceTypeTestBase,
       DecorateInfo,
       instantiate_device_type_tests,
       precisionOverride,
   )

   class MyAcceleratorTestBase(DeviceTypeTestBase):
       device_type = "mydevice"

       # Allow tests gated by @onlyCUDA to run on this backend
       bypass_device_restrictions = True

       # Override specific operator tests
       op_overrides = {
           "op_skip": [DecorateInfo(unittest.skip("not yet implemented"))],
           "op_xfail": [DecorateInfo(unittest.expectedFailure)],
       }

   # Skip specific test classes or methods for this backend
   MyAcceleratorTestBase.test_exclusions = {
       "TestLinearAlgebra": "*",
       "TestCommon": ["test_noncontiguous_samples"],
   }

   # Instantiate PyTorch's built-in tests for your device
   from test_ops import TestCommon

   instantiate_device_type_tests(TestCommon, globals(), only_for="mydevice")

.. note::

   This page is updated continuously as the PyTorch testing infrastructure
   evolves. See the `PyTorch Test Refactoring project
   <https://github.com/orgs/pytorch/projects/154>`_ for ongoing work.

.. seealso::

   - :ref:`cpp-extension`
   - `Contributing to PyTorch <https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md>`_
