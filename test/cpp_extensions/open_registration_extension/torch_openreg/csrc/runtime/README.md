# OpenReg Runtime — Overview

This directory implements a minimal runtime integration for the OpenReg backend (PrivateUse1). It provides device management, allocators, RNG generator hooks, serialization support, and device guard integration used by the rest of the codebase.

## Key Modules

- **OpenRegFunctions**
High-level device APIs exposed to the rest of the codebase and other modules:
  - `device_count()`: query number of devices.
  - `current_device()`, `set_device()`, `ExchangeDevice()`: get/set/atomically exchange current device.

- **OpenRegGuard**
`OpenRegGuardImpl` implements c10 device guard interface where PyTorch High-level API will use to control the device:
  - Provide device management: `setDevice()`/`getDevice()`/`exchangeDevice()` etc.
  - Provides stream/event management: `getStream()`/`queryStream()`/`synchronizeEvent()` etc.

- **OpenRegDeviceAllocator**
`OpenRegDeviceAllocator` implements the `Allocator` interface to allocate, free and copy device memory for the PrivateUse1 (OpenReg) device type:
  - `allocate()`, `raw_deleter()`, `copy_data()`: allocate/free/copy device memory.

- **OpenRegHostAllocator**
`OpenRegHostAllocator` implements the `HostAllocator` interface for host memory management:
  - `allocate()`, `raw_deleter()`, `copy_data()`: allocate/free/copy host memory.

- **OpenRegGenerator**
`OpenRegGeneratorImpl` implements the `CPUGeneratorImpl` for random number generator:
  - `OpenRegGeneratorImpl` derives `CPUGeneratorImpl` and sets the generator's device and dispatch key for PrivateUse1.
  - `getDefaultOpenRegGenerator()` returns a per-device default generator and lazily initializes vector of generators.

- **OpenRegHooks**
`OpenRegHooksInterface` implement `PrivateUse1HooksInterface` for PyTorch to access to backend's capability:
  - `getPinnedMemoryAllocator()`: get device pinned buffers.
  - `isPinnedPtr()`: check whether a pointer is pointing to a pinned buffer.
  - `getDefaultGenerator()` / `getNewGenerator()`: get a generator by device index.
  - `hasPrimaryContext()`: check the existence of the primary context.

- **OpenRegSerialization**
Registers serialization callbacks with the PrivateUse1 backend:
  - Provides utility functions to attach backend metadata to tensors during (de)serialization.
