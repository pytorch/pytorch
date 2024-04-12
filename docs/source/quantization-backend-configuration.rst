Quantization Backend Configuration
----------------------------------

FX Graph Mode Quantization allows the user to configure various
quantization behaviors of an op in order to match the expectation
of their backend.

In the future, this document will contain a detailed spec of
these configurations.


Default values for native configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is the output of the configuration for quantization of ops
in x86 and qnnpack (PyTorch's default quantized backends).

Results:

.. literalinclude:: scripts/quantization_backend_configs/default_backend_config.txt
