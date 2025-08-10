# OpenReg: An Accelerator Backend that Simulates CUDA Behavior on a CPU

## Introduction

OpenReg is a C++ backend library that simulates the behavior of a CUDA-like device on a CPU. Its core objective is **not to accelerate computation or improve performance**, but rather to **simulate modern CUDA programming, enabling developers to prototype and test in an environment without actual GPU hardware**. The current design principles are as follows:

* **API Consistency**: Provide an interface consistent with the CUDA Runtime API, allowing upper-level applications (like PyTorch's PrivateUse1 backend) to switch and test seamlessly.
* **Functional Consistency**: Provide behavior consistent with the CUDA Runtime, such as memory isolation, device context management, etc.
* **Completeness**: Aim to support PrivateUse1 device integration and safeguard the third-party device integration mechanism, without striving to cover all capabilities of the CUDA Runtime.

## Directory Structure

The project's code is organized with a clear structure and separation of responsibilities:

```text
openreg/
├── CMakeLists.txt      # Top-level CMake build script, used to compile and generate libopenreg.so
├── include/
│   └── openreg.h       # Public API header file, external users only need to include this file
└── csrc/
    ├── device.cpp      # Implementation of device management-related APIs
    └── memory.cpp      # Implementation of APIs for memory management, copying, and protection
```

* `include/openreg.h`: Defines all externally exposed C-style APIs, data structures, and enums. It is the "public face" of this library.
* `csrc/`: Contains the C++ implementation source code for all core functionalities.
  * `device.cpp`: Implements device discovery (`orGetDeviceCount`) and thread context management (`orSetDevice`/`orGetDevice`).
  * `memory.cpp`: Implements the core functions of memory allocation (`orMalloc`/`orMallocHost`), deallocation, copying, and memory protection (`orMemoryProtect`, `orMemoryUnprotect`).
* `CMakeLists.txt`: Responsible for compiling and linking all source files under the `csrc/` directory to generate the final `libopenreg.so` shared library.

## Implemented APIs

OpenReg currently provides a set of APIs covering basic memory and device management.

### Device Management APIs

| OpenReg              | CUDA                 | Feature Description                               |
| :------------------- | :------------------- | :------------------------------------------------ |
| `orGetDeviceCount`   | `cudaGetDeviceCount` | Get the number of devices                         |
| `orSetDevice`        | `cudaSetDevice`      | Set the current device for the current thread     |
| `orGetDevice`        | `cudaGetDevice`      | Get the current device for the current thread     |

### Memory Management APIs

| OpenReg                  | CUDA                         | Feature Description                        |
| :----------------------- | :--------------------------- | :----------------------------------------- |
| `orMalloc`               | `cudaMalloc`                 | Allocate device memory                     |
| `orFree`                 | `cudaFree`                   | Free device memory                         |
| `orMallocHost`           | `cudaMallocHost`             | Allocate page-locked (Pinned) host memory  |
| `orFreeHost`             | `cudaFreeHost`               | Free page-locked host memory               |
| `orMemcpy`               | `cudaMemcpy`                 | Synchronous memory copy                    |
| `orMemcpyAsync`          | `cudaMemcpyAsync`            | Asynchronous memory copy                   |
| `orPointerGetAttributes` | `cudaPointerGetAttributes`   | Get pointer attributes                     |
| `orMemoryUnprotect`      | -                            | (Internal use) Unprotect memory            |
| `orMemoryProtect`        | -                            | (Internal use) Restore memory protection   |

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

## Usage Example

The following is a simple code snippet demonstrating how to use the core features of the OpenReg library.

```cpp
#include "openreg.h"
#include <iostream>
#include <vector>
#include <cstdio>

#define OR_CHECK(call) do { \
    orError_t err = call; \
    if (err != orSuccess) { \
        fprintf(stderr, "OR Error code %d in %s at line %d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    int device_count = 0;
    OR_CHECK(orGetDeviceCount(&device_count));
    std::cout << "Found " << device_count << " simulated devices." << std::endl;

    int current_device = -1;
    OR_CHECK(orSetDevice(1));
    OR_CHECK(orGetDevice(&current_device));
    std::cout << "Set current device to " << current_device << "." << std::endl;

    const int n = 1024;
    const size_t size = n * sizeof(int);
    int *h_a, *d_a;
    OR_CHECK(orMallocHost((void**)&h_a, size));
    OR_CHECK(orMalloc((void**)&d_a, size));

    orPointerAttributes attr;
    OR_CHECK(orPointerGetAttributes(&attr, d_a));
    std::cout << "Pointer " << (void*)d_a << " is of type " << attr.type
              << " on device " << attr.device << std::endl;

    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
    }
    OR_CHECK(orMemcpy(d_a, h_a, size, orMemcpyHostToDevice));
    std::cout << "Data copied from Host to Device." << std::endl;

    // std::cout << "Trying to access device memory directly from CPU..." << std::endl;
    // int val = d_a[0]; // CRASH!

    // Clean up resources
    OR_CHECK(orFree(d_a));
    OR_CHECK(orFreeHost(h_a));
    std::cout << "Resources freed." << std::endl;

    return 0;
}
```

## Next Steps

To better support PrivateUse1 device integration, the following capabilities are planned for the future:

* **Stream Support**: Provide the ability to simulate CUDA Streams.
* **Event Support**: Provide the ability to simulate CUDA Events.
* **Cross-Platform Support**: Add support for Windows and macOS (low priority).
