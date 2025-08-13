# OpenReg: An Accelerator Backend that Simulates CUDA Behavior on a CPU

## Introduction

OpenReg is a C++ backend library that simulates the behavior of a CUDA-like device on a CPU. Its core objective is **not to accelerate computation or improve performance**, but rather to **simulate modern CUDA programming, enabling developers to prototype and test in an environment without actual GPU hardware**. The current design principles are as follows:

* **API Consistency**: Provide an interface consistent with the CUDA Runtime API, allowing upper-level applications (like PyTorch's `PrivateUse1` backend) to switch and test seamlessly.
* **Functional Consistency**: Provide behavior consistent with the CUDA Runtime, such as memory isolation, device context management, etc.
* **Completeness**: Aim to support `PrivateUse1` device integration and safeguard the third-party device integration mechanism, without striving to cover all capabilities of the CUDA Runtime.

## Directory Structure

The project's code is organized with a clear structure and separation of responsibilities:

```text
openreg/
├── CMakeLists.txt      # Top-level CMake build script, used to compile and generate libopenreg.so
├── include/
│   ├── openreg.h       # Public API header file, external users only need to include this file
│   └── openreg.inl     # Public API header file, as an extension of openreg.h, cannot be included separately.
└── csrc/
    ├── device.cpp      # Implementation of device management APIs
    ├── memory.cpp      # Implementation of memory management APIs
    └── stream.cpp      # Implementation of stream and event APIs.
```

* `include`: Defines all externally exposed APIs, data structures, and enums.
  * `openreg.h`: Defines all externally exposed C-style APIs.
  * `openreg.inl`: Defines all externally exposed C++ APIs.
* `csrc/`: Contains the C++ implementation source code for all core functionalities.
  * `device.cpp`: Implements the core functions of device management: device discovery and context management.
  * `memory.cpp`: Implements the core functions of memory management: allocation, free, copy and memory protection.
  * `stream.cpp`: Implements the core functions of stream and event: creation, destroy, record, synchronization and so on.
* `CMakeLists.txt`: Responsible for compiling and linking all source files under the `csrc/` directory to generate the final `libopenreg.so` shared library.

## Implemented APIs

OpenReg currently provides a set of APIs covering basic memory and device management.

### Device Management APIs

| OpenReg                          | CUDA                               | Feature Description                |
| :------------------------------- | :--------------------------------- | :--------------------------------- |
| `orGetDeviceCount`               | `cudaGetDeviceCount`               | Get the number of available GPUs   |
| `orSetDevice`                    | `cudaSetDevice`                    | Set the active GPU                 |
| `orGetDevice`                    | `cudaGetDevice`                    | Get the current GPU                |
| `orDeviceSynchronize`            | `cudaDeviceSynchronize`            | Wait for all GPU tasks to finish   |
| `orDeviceGetStreamPriorityRange` | `cudaDeviceGetStreamPriorityRange` | Get the range of stream priorities |

### Memory Management APIs

| OpenReg                  | CUDA                       | Feature Description                       |
| :----------------------- | :------------------------- | :---------------------------------------- |
| `orMalloc`               | `cudaMalloc`               | Allocate device memory                    |
| `orFree`                 | `cudaFree`                 | Free device memory                        |
| `orMallocHost`           | `cudaMallocHost`           | Allocate page-locked (Pinned) host memory |
| `orFreeHost`             | `cudaFreeHost`             | Free page-locked host memory              |
| `orMemcpy`               | `cudaMemcpy`               | Synchronous memory copy                   |
| `orMemcpyAsyn`           | `cudaMemcpyAsyn`           | Asynchronous memory copy                  |
| `orPointerGetAttributes` | `cudaPointerGetAttributes` | Get pointer attributes                    |

### Stream APIs

| OpenReg                      | CUDA                           | Feature Description                    |
| :--------------------------- | :----------------------------- | :------------------------------------- |
| `orStreamCreate`             | `cudaStreamCreate`             |  Create a default-priority stream      |
| `orStreamCreateWithPriority` | `cudaStreamCreateWithPriority` |  Create a stream with a given priority |
| `orStreamDestroy`            | `cudaStreamDestroy`            |  Destroy a stream                      |
| `orStreamQuery`              | `cudaStreamQuery`              |  Check if a stream has completed       |
| `orStreamSynchronize`        | `cudaStreamSynchronize`        |  Wait for a stream to complete         |
| `orStreamWaitEvent`          | `cudaStreamWaitEvent`          |  Make a stream wait for an event       |
| `orStreamGetPriority`        | `cudaStreamGetPriority`        |  Get a stream’s priority               |

### Event APIs

| OpenReg                  | CUDA                       | Feature Description                 |
| :----------------------- | :------------------------- | :---------------------------------- |
| `orEventCreate`          | `cudaEventCreate`          | Create an event with default flag   |
| `orEventCreateWithFlags` | `cudaEventCreateWithFlags` | Create an event with specific flag  |
| `orEventDestroy`         | `cudaEventDestroy`         | Destroy an event                    |
| `orEventRecord`          | `cudaEventRecord`          | Record an event in a stream         |
| `orEventSynchronize`     | `cudaEventSynchronize`     | Wait for an event to complete       |
| `orEventQuery`           | `cudaEventQuery`           | Check if an event has completed     |
| `orEventElapsedTime`     | `cudaEventElapsedTime`     | Get time elapsed between two events |

## Implementation Principles

### Device Management Principles

Simulating multiple devices and thread-safe device context switching:

1. **Device Count**: The total number of simulated devices is defined by the compile-time constant `constexpr int kDeviceCount`.
2. **Device Switching**: Device switching in multi-threaded scenarios is simulated using a **TLS (Thread-Local Storage) global variable**.

### Memory Management Principles

Simulating device memory, host memory, and memory copies:

1. **Allocation**: A page-aligned memory block is allocated using `mmap` + `mprotect` with the permission flag `PROT_NONE`. Read, write, and execute operations on this memory region are all prohibited.
2. **Deallocation**: Memory is freed using `munmap`.
3. **Authorization**: When a legitimate memory access is required, an RAII guard restores the memory permissions to `PROT_READ | PROT_WRITE`. The permissions are automatically reverted to `PROT_NONE` when the scope is exited.

### Stream&Event Principles

Simulating creation, release and synchronization for event and steam:

1. **Event**: Each event is encapsulated as a task function and placed into a stream, which acts as a thread. Upon completion of the task, a flag within the event is modified to simulate the event's status.
2. **Stream**: When each stream is requested, a new thread is created, which sequentially processes each task in the task queue within the stream structure. Tasks can be wrappers around kernel functions or events.
3. **Synchronization**: Synchronization between streams and events is achieved using multithreading, condition variables, and mutexes.

## Usage Example

Please refer to [example](example/example.cpp) for example.

The command to compile example.cpp is as follow:

```Shell
pushd third_party/openreg/

g++ -o out example/example.cpp -L ../../torch_openreg/lib -lopenreg
LD_LIBRARY_PATH=../../torch_openreg/lib ./out

popd
```

The output is as follow:

```Shell
Current environment have 2 devices
Current is 0 device
All tasks have been submitted.
Kernel execution time: 0.238168 ms
Verification PASSED!
```

## Next Steps

The most basic functions of the OpenReg backend are currently supported, and will be dynamically optimized and expanded based on the needs of PyTorch integration.
