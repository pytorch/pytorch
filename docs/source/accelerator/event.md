# Event

## Background

Events are lightweight synchronization primitives used to coordinate work across streams on the same device. In PyTorch, an event API should:
- Record a point on a stream’s work queue
- Make another stream wait until that point completes
- Allow non-blocking completion queries and host-side synchronization
- Optionally measure elapsed time between two recorded events (when timing is enabled)

Goal: deliver a minimal, production-ready Event implementation aligned with the OpenReg reference, with correct device routing and full `torch.Event` interop.

## Design

The accelerator needs to support the following basic functionalities.

| Functionality | Description | When to use |
| :-- | :-- | :-- |
| Record | Record the event on a given stream (or the current stream). | Mark a synchronization point in a stream. |
| Wait/Block | Make a stream wait until the event completes. | Coordinate work across streams. |
| Query | Non-blocking completion check. | Poll for completion. |
| Synchronize | Host-side wait until completion. | Ensure completion before host continues. |
| Timing (optional) | Measure elapsed time between two completed, timing-enabled events. | Lightweight timing when the backend supports it. |

## Implementation

Below we use a single representative function, record, to walk through how an Event is implemented across layers.

### Python Side

The Python wrapper `torch.Event.record(stream=None)` either records on an explicit stream or, if none is provided, fetches the current stream from the backend via `VirtualGuardImpl` and records there.

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT RECORD
    :end-before: LITERALINCLUDE END: PYTORCH EVENT RECORD
    :linenos:
```

### C++ Side

On the backend, OpenReg provides two entry points: a convenience `record()` that uses the current stream, and the core `record(stream)` that validates device consistency, lazily creates the event if needed, then enqueues the record operation.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT RECORD DEFAULT
    :end-before: LITERALINCLUDE END: OPENREG EVENT RECORD DEFAULT
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT RECORD
    :end-before: LITERALINCLUDE END: OPENREG EVENT RECORD
    :linenos:
```

### Integration

#### Guard

When no stream is provided from Python, the record path uses `VirtualGuardImpl(DeviceType)` to retrieve the current stream for the event’s device type and device index. This dispatch relies on the backend registering a `DeviceGuardImplInterface` implementation. OpenReg wires this up as follows:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD REGISTRATION
    :end-before: LITERALINCLUDE END: OPENREG GUARD REGISTRATION
    :linenos:
```

With this registration in place, `VirtualGuardImpl` can obtain the backend’s current stream, ensuring `torch.Event.record()` without an explicit stream is correctly routed.

## Summary

This minimal design implements Event by routing Python calls through `VirtualGuardImpl` and delegating backend specifics to OpenReg. Using the record path as an example, we showed how Python resolves the stream (explicit or current), how the backend validates devices and records, and how Guard wiring makes the integration work uniformly across backends.
