# Background

Streams are the asynchronous work queues of an accelerator runtime:
- Enable multiple independent queues to make progress on the same device
- Provide a per-device default stream and a reusable pool of non-default streams
- Support events to record/wait/synchronize cross-stream dependencies
- Expose a device-agnostic Python entry via `torch.Stream`

Goal: implement the minimal, production-worthy Stream support by following the OpenReg reference, with correct behavior, context restoration, and full interop with `torch.Stream`.

# Design and API (OpenRegStream.cpp)

Interface and semantics at a glance

| API | Explanation |
| :--- | :--- |
| getStreamFromPool(priority, device_index) | Round-robin a non-default stream from the per-device pool. Clamp out-of-range priority; device_index = -1 means “current device” (may trigger lazy init). The returned stream’s device_type/device_index must match the input. |
| getStreamFromPool(isHighPriority, device_index) | Global pool alias (in the example, equivalent to a fixed priority). device_index semantics same as above. Returns a non-default stream. |
| getDefaultOpenRegStream(device_index) | Return the device’s default stream. device_index = -1 means current device. Default stream has a fixed id; the underlying orStream_t is typically nullptr; it must not be destroyed. |
| getCurrentOpenRegStream(device_index) | Return the “current (TLS) stream” on the device for the current thread. If not explicitly set, it equals the default stream. device_index = -1 means current device. |
| setCurrentOpenRegStream(stream) | Set the TLS current stream for the device in the current thread; does not switch devices; requires the stream to belong to the same device; no return value. |
| getStreamFromExternal(ext_stream, device_index) | Wrap a foreign orStream_t as an OpenRegStream (StreamId type EXT). Ownership is not transferred; device_index must match the handle’s real device. |
| OpenRegStream::stream() | Unwrap StreamId to the underlying orStream_t. External streams unwrap to the original pointer. The default stream maps to the backend’s default lane (often the priority-0 lane); whether the underlying handle is null depends on the driver. |


# Implementation

This section shows the key C++ and Python pieces and how they fit together.

## C++ implementation

1) Stream pool and creation: lazily initialize per device; batch-create orStream_t by device and priority; initialize counters; set up per-device TLS default/current streams.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegStream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG INIT SINGLE DEVICE STREAM
    :end-before: LITERALINCLUDE END: OPENREG INIT SINGLE DEVICE STREAM
    :linenos:
```

2) Guard hooks: `torch.Stream(device=...)` calls into the backend via `VirtualGuardImpl.getNewStream` to get a pooled stream; context managers switch TLS current stream via `exchangeStream` and return the previous stream for restoration.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD GET NEW STREAM
    :end-before: LITERALINCLUDE END: OPENREG GUARD GET NEW STREAM
    :linenos:
```

Switch TLS current stream (with semantics) and return the previous stream:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD EXCHANGE STREAM
    :end-before: LITERALINCLUDE END: OPENREG GUARD EXCHANGE STREAM
    :linenos:
```

Guard registration: register for DeviceType PrivateUse1 so `VirtualGuardImpl(PrivateUse1)` routes to your implementation.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD REGISTRATION
    :end-before: LITERALINCLUDE END: OPENREG GUARD REGISTRATION
    :linenos:
```

3) TLS current stream and external stream: set the per-thread (TLS) current stream.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegStream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG SET CURRENT STREAM
    :end-before: LITERALINCLUDE END: OPENREG SET CURRENT STREAM
    :linenos:
```

Wrap a foreign orStream_t as an OpenRegStream for unified PyTorch interop:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegStream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GET EXTERNAL STREAM
    :end-before: LITERALINCLUDE END: OPENREG GET EXTERNAL STREAM
    :linenos:
```

## Python implementation (C extension)

1) Device-agnostic constructor: parse `device/priority` and delegate creation via `VirtualGuardImpl(device_type).getNewStream(...)`; or unpack an existing stream via `(stream_id, device_index, device_type)`.

Delegate creation to the backend Guard’s getNewStream:

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Stream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH STREAM CTOR GETNEW
    :end-before: LITERALINCLUDE END: PYTORCH STREAM CTOR GETNEW
    :linenos:
```

Optional: unpack an existing stream (restore the underlying stream from the triplet).

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Stream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH STREAM CTOR UNPACK
    :end-before: LITERALINCLUDE END: PYTORCH STREAM CTOR UNPACK
    :linenos:
```

2) Context manager: on enter, switch device if needed and replace the TLS current stream; on exit, restore the previous device and stream.

```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Stream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH STREAM ENTER SET CURRENT
    :end-before: LITERALINCLUDE END: PYTORCH STREAM ENTER SET CURRENT
    :linenos:
```


```{eval-rst}
.. literalinclude:: ../../../torch/csrc/Stream.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: PYTORCH STREAM EXIT RESTORE
    :end-before: LITERALINCLUDE END: PYTORCH STREAM EXIT RESTORE
    :linenos:
```
