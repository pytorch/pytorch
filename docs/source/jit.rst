TorchScript
===========

.. toctree::
    :maxdepth: 1
    :hidden:

    jit_language_reference
    jit_language_reference_v2
    jit_python_reference
    jit_unsupported
    torch.jit.supported_ops <jit_builtin_functions>

.. warning::
    TorchScript is deprecated, please use
    `torch.export <https://docs.pytorch.org/docs/stable/export.html>`__ instead.

.. automodule:: torch.jit

Creating TorchScript Code
--------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    script
    trace
    script_if_tracing
    trace_module
    fork
    wait
    ScriptModule
    ScriptFunction
    freeze
    optimize_for_inference
    enable_onednn_fusion
    onednn_fusion_enabled
    set_fusion_strategy
    strict_fusion
    save
    load
    ignore
    unused
    interface
    isinstance
    Attribute
    annotate


.. This package is missing doc. Adding it here for coverage
.. This does not add anything to the rendered page.
.. py:module:: torch.jit.supported_ops
.. py:module:: torch.jit.unsupported_tensor_ops
.. py:module:: torch.jit.mobile
.. py:module:: torch.jit.annotations
.. py:module:: torch.jit.frontend
.. py:module:: torch.jit.generate_bytecode
.. py:module:: torch.jit.quantized
