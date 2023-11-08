.. _torch.compiler_ir:

IRs
===============

PyTorch 2.0 offers two set of IRs for backends to interface with: Core Aten IR and Prims IR.

Core Aten IR
--------------------

Core aten ops is the core subset of aten operators that can be used to compose other operators.
Core aten IR is fully functional, and there is no `inplace` or `_out` variants in this opset.
In contrast to Prims IR, core aten ops reuses the existing aten ops in "native_functions.yaml",
and it doesn't further decompose ops into explicit type promotion and broadcasting ops.
This opset is designed to serve as the functional IR to interface with backends.

.. warning::
  This opset is still under active development, more ops will be added in the future.

.. csv-table::
   :file: ../build/ir/aten_ops.csv
   :widths: auto
   :header-rows: 1

Prims IR
-----------

Prims IR is a set of primitive operators that can be used to compose other operators.
Prims IR is a lower level opset than core aten IR, and it further decomposes ops into explicit
type promotion and broadcasting ops: prims.convert_element_type and prims.broadcast_in_dim.
This opset is designed to interface with compiler backends.

.. warning::
  This opset is still under active development, more ops will be added in the future.

.. csv-table::
   :file: ../build/ir/prims_ops.csv
   :widths: auto
   :header-rows: 1
