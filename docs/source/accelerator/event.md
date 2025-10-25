# Background

Events are lightweight synchronization primitives used to coordinate work across streams on the same device:
- Record: mark a point on a stream's work queue
- Wait/Block: make another stream wait until the recorded point completes
- Query: non-blocking completion check
- Synchronize: host-side wait until completion
- Timing: measure elapsed time between two recorded events (when enabled)

Goal: provide a minimal, production-lean Event design that works across backends via `c10::Event` and `VirtualGuardImpl`, with OpenReg as a clear, small reference.

# Design and API (OpenRegEvent.h)

Interface and semantics at a glance

| API | Explanation |
| :--- | :--- |
| `OpenRegEvent(bool enable_timing)` | Construct an event wrapper; timing enabled controls whether `elapsed_time` is allowed. |
| `record(stream)` / `recordOnce(stream)` | Record the event on a given stream; `recordOnce` is idempotent. Requires event/stream on same device. |
| `block(stream)` | Insert a wait on the stream until the event completes. |
| `query()` | Non-blocking completion check; true if never recorded or already completed. |
| `synchronize()` | Block the host until the event completes. |
| `elapsed_time(other)` | Milliseconds between two completed, timing-enabled events. |
| `device()`/`device_index()` | Inspect device ownership if created; otherwise empty/default. |
| `operator orEvent_t()` / `event()` | Unwrap to backend handle; ownership not transferred. |

# Implementation

This section shows the key C++ and Python pieces and how they fit together.

## C++ implementation (OpenReg)

1) Destructor safety: never let exceptions escape from destructors.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT DTOR
    :end-before: LITERALINCLUDE END: OPENREG EVENT DTOR
    :linenos:
```

2) Create event lazily on first record; remember device ownership.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT CREATE
    :end-before: LITERALINCLUDE END: OPENREG EVENT CREATE
    :linenos:
```

3) Record variants: default to current stream or explicitly provided stream; enforce same-device.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT RECORD DEFAULT
    :end-before: LITERALINCLUDE END: OPENREG EVENT RECORD DEFAULT
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT RECORD ONCE
    :end-before: LITERALINCLUDE END: OPENREG EVENT RECORD ONCE
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT RECORD
    :end-before: LITERALINCLUDE END: OPENREG EVENT RECORD
    :linenos:
```

4) Wait/block and host synchronize.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT BLOCK
    :end-before: LITERALINCLUDE END: OPENREG EVENT BLOCK
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT SYNC
    :end-before: LITERALINCLUDE END: OPENREG EVENT SYNC
    :linenos:
```

5) Query and elapsed time (requires timing-enabled pair and completion).

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT QUERY
    :end-before: LITERALINCLUDE END: OPENREG EVENT QUERY
    :linenos:

.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG EVENT ELAPSED
    :end-before: LITERALINCLUDE END: OPENREG EVENT ELAPSED
    :linenos:
```

Note: When using PyTorch's `c10::Event` facade, device switching is handled by the backend `DeviceGuardImpl` (see Python section). If you directly use `OpenRegEvent` without the guard path, ensure calls that depend on the current device are issued on the correct device.

## Python implementation (C extension)

1) Device-agnostic constructor: parse flags and build a `c10::Event` bound to a device type.

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT CTOR
    :end-before: LITERALINCLUDE END: PYTORCH EVENT CTOR
    :linenos:
```

2) Record: use an explicit stream if provided; otherwise take the current stream from the backend via `VirtualGuardImpl`.

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT RECORD
    :end-before: LITERALINCLUDE END: PYTORCH EVENT RECORD
    :linenos:
```

3) Wait (block) on a stream.

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT WAIT
    :end-before: LITERALINCLUDE END: PYTORCH EVENT WAIT
    :linenos:
```

4) Query, elapsed time, and synchronize.

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT QUERY
    :end-before: LITERALINCLUDE END: PYTORCH EVENT QUERY
    :linenos:

.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT ELAPSED
    :end-before: LITERALINCLUDE END: PYTORCH EVENT ELAPSED
    :linenos:

.. literalinclude:: ../../../torch/csrc/Event.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH EVENT SYNC
    :end-before: LITERALINCLUDE END: PYTORCH EVENT SYNC
    :linenos:
```

Technical note: `VirtualGuardImpl(device_type)` routes `record/block/...` to the backend's `DeviceGuardImplInterface` implementation (OpenReg registers one for `PrivateUse1`), which handles any required device switching and interop with native stream/event handles.
