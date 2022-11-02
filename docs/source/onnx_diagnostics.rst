torch.onnx diagnostics
======================

.. contents:: :local:
.. automodule:: torch.onnx._internal.diagnostics
.. currentmodule:: torch.onnx._internal.diagnostics

Overview
--------

NOTE: This feature is underdevelopment and is subject to change.

The goal is to improve the diagnostics to help users debug and improve their model export to ONNX.

- The diagnostics are emitted in machine parsable `Static Analysis Results Interchange Format (SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__.
- A new clearer, structured way to add new and keep track of diagnostic rules.
- Serve as foundation for more future improvements consuming the diagnostics.


Diagnostic Rules
----------------

.. toctree::
    :glob:

    generated/onnx_diagnostics_rules/*

API Reference
-------------

.. autofunction:: torch.onnx._internal.diagnostics.diagnose

.. autoclass:: torch.onnx._internal.diagnostics.ExportDiagnostic
    :members:

.. autoclass:: torch.onnx._internal.diagnostics.infra.DiagnosticEngine
    :members:
