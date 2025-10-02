```{eval-rst}
.. currentmodule:: torch._dynamo.config
```

# PyTorch Compiler Configuration

```{eval-rst}
.. automodule:: torch._dynamo.config
   :members:
   :undoc-members:
   :show-inheritance:
```

This documentation covers PyTorch compiler configuration options, organized according to their scope and intended usage. Based on the principle that the `torch.compiler` namespace is intended for cross-cutting configurations that affect the entire compilation stack, we distinguish between:

- **Cross-Cutting Configurations**: User-facing settings that affect multiple components (TorchDynamo, TorchInductor, AOTAutograd)
- **TorchDynamo-Specific Configurations**: Internal settings that only affect TorchDynamo's graph tracing and capture behavior

## Cross-Cutting Configuration Options

These configurations affect the entire PyTorch compilation pipeline and are intended for end-user control:

### Dynamic Shape Settings

```{eval-rst}
.. autodata:: torch._dynamo.config.dynamic_shapes
   :annotation: = True
```

```{eval-rst}
.. autodata:: torch._dynamo.config.assume_static_by_default
   :annotation: = True
```

```{eval-rst}
.. autodata:: torch._dynamo.config.automatic_dynamic_shapes
   :annotation: = True
```

### Compilation Performance Controls

```{eval-rst}
.. autodata:: torch._dynamo.config.recompile_limit
   :annotation: = 8
```

```{eval-rst}
.. autodata:: torch._dynamo.config.accumulated_recompile_limit
   :annotation: = 256
```

### User-Facing Debugging

```{eval-rst}
.. autodata:: torch._dynamo.config.verbose
   :annotation: = os.environ.get("TORCHDYNAMO_VERBOSE", "0") == "1"
```

## TorchDynamo-Specific Configuration Options

These configurations are specific to TorchDynamo's internal tracing and graph capture behavior:

### Tracing and Capture Behavior

```{eval-rst}
.. autodata:: torch._dynamo.config.capture_scalar_outputs
   :annotation: = os.environ.get("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS") == "1"
```

```{eval-rst}
.. autodata:: torch._dynamo.config.capture_dynamic_output_shape_ops
   :annotation: = os.environ.get("TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS", "0") == "1"
```

### Error Handling and Validation

```{eval-rst}
.. autodata:: torch._dynamo.config.suppress_errors
   :annotation: = bool(os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", False))
```

```{eval-rst}
.. autodata:: torch._dynamo.config.verify_correctness
   :annotation: = False
```

### Debugging and Logging

```{eval-rst}
.. autodata:: torch._dynamo.config.log_file_name
   :annotation: = None
   :noindex:
```

## Configuration Organization

This documentation follows the architectural principle that cross-cutting configurations (those affecting the entire compilation stack) should be distinguished from component-specific configurations (those affecting only TorchDynamo's internal behavior).

**Cross-cutting configurations** are suitable for:
- End-user configuration and tuning
- Public API documentation
- General compilation behavior control

**TorchDynamo-specific configurations** are primarily for:
- Advanced users and developers
- Internal debugging and validation
- Fine-tuning TorchDynamo's tracing behavior
