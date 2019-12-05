This directory contains the C++ backend for the autocasting component of automatic mixed precision.

This README is for developers interested in the implementation details of autocasting.  It assumes familiarity with the
internal [op registration API](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/core/op_registration) and
[dispatch](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/core/dispatch).
If you're looking for guidance on what autocasting is, how it helps, or how to use it in your network,
this is _not_ the right place.  Please consult the official [Python API documentation]() or [C++ API documentation]().

Autocasting uses a dedicated `TensorTypeId::AutocastTensorId`.
Several dozen operations receive wrapping functions explicitly registered for `AutocastTensorId`.
Conceptually, explicitly-registered wrappers _should_ (see **Known Issues**) fall into 3 categories, corresponding to entries in
`register_autocast_ops.cpp`'s `CastPolicy` enum:

* **CastPolicy::fp16**: The wrapper casts incoming floating-point Tensors to `torch.float16`.
    Examples:  GEMMs, Convolutions.
* **CastPolicy::fp32**: The wrapper casts incoming floating-point Tensors to `torch.float32`.
    Examples:  Reductions, Exponentiation.
* **CastPolicy::promote**: The wrapper determines the widest type among several floating-point Tensor arguments, and casts all
    incoming floating-point Tensors to that type.  This is required for operations with multiple Tensor arguments that do not
    implement their own type promotion logic internally.

Explicitly-registered operations for each category are chosen based on Tensor Core support, experience, and reasoning about
their numerical properties.  The set of operations in each category may change over time.

Any op that doesn't require casts _should_ (see **Known Issues**) route through `autocast_fallback`,
which runs the op with unaltered arguments.  `autocast_fallback` may become unnecessary in the long term,
if dispatch eventually implements [fallthrough](https://github.com/pytorch/pytorch/issues/29548) for ops that aren't
explicitly registered for a given `TensorTypeId`.

`AutocastTensorId` takes precedence over `VariableTensorId`.  If the dispatcher detects that `AutocastTensorId` is
enabled (either because the `AutocastTensorId` bit is set in `c10::impl::local_tensor_type_set().included_`,
or because at least one Tensor argument has the `AutocastTensorId` bit set in its `TensorTypeSet`), the dispatcher will
route through the registered autocast operation (or `autocast_fallback`) before routing through the VariableType operation.
This ensures casting occurs before autograd history recording.  Operations that run in `torch.float16` (e.g. matrix multiplies
and convolutions) therefore save arguments for backward in `torch.float16`, which reduces their memory footprint.

`register_autocast_ops.cpp` maintains a cache of all leaf tensors (model weights) it has casted from
`torch.float32` to `torch.float16`.  If a given `float32` leaf tensor is reused across several `CastPolicy::fp16` ops
during a single Python-side invocation of [`with torch.cuda.amp.autocast()`](), all `fp16` ops after the first will pull the
cached cast, rather than creating another temporary `torch.float16` Tensor.  This is an important optimization for RNNs.
The cache is cleared by the `torch.cuda.amp.autocast` context manager on exit.

#### Files:

* autocast_mode.h/cpp:  Minimal interface used by external code.
* register_autocast_ops.cpp:  Backend functionality (wrapper templates and wrapper registration).

#### Known Issues:

Currently, the autocast wrappers require a decent amount of special casing, including an additional
**CastPolicy::passthrough** to service some ops and handwritten wrappers for a few others.
See the commented categories ("Templates for well-behaved ops" and "Special treatment for ops that aren't well-behaved")
in `register_autocast_ops.cpp` for more detail.
