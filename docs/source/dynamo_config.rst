torch._dynamo.config
===================

.. currentmodule:: torch._dynamo.config

.. automodule:: torch._dynamo.config
   :members:
   :undoc-members:
   :show-inheritance:
   
  Configuration Variables
------------------------

.. autodata:: torch._dynamo.config.log_file_name
   :annotation: = None

.. autodata:: torch._dynamo.config.verbose
   :annotation: = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"

.. autodata:: torch._dynamo.config.verify_correctness
   :annotation: = False

.. autodata:: torch._dynamo.config.recompile_limit
   :annotation: = 8

.. autodata:: torch._dynamo.config.accumulated_recompile_limit
   :annotation: = 256

.. autodata:: torch._dynamo.config.suppress_errors
   :annotation: = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))

.. autodata:: torch._dynamo.config.dynamic_shapes
   :annotation: = True

.. autodata:: torch._dynamo.config.assume_static_by_default
   :annotation: = True

.. autodata:: torch._dynamo.config.automatic_dynamic_shapes
   :annotation: = True

.. autodata:: torch._dynamo.config.capture_scalar_outputs
   :annotation: = os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "1"

.. autodata:: torch._dynamo.config.capture_dynamic_output_shape_ops
   :annotation: = os.environ.get("TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS", "0") == "1"
