# PyTorch OpenReg

## Background

The third-party device integration mechanism based on PrivateUse1 has become the official mainstream method for new backends to integrate with PyTorch. Ensuring the availability of this mechanism is crucial for enriching PyTorch's hardware ecosystem.

**Note:**

The goal of `torch_openreg` is **not to implement a fully functional, high-performance PyTorch backend**, but to serve as a **minimalist reference implementation for mechanism verification**.

### Purpose

- **Test Backend**: To serve as an in-tree test backend for PrivateUse1, ensuring quality stability through CI/CD.
- **Integration Example**: To serve as a reference example for new backend integration.
- **Integration Documentation**: To provide module-level integration documentation that corresponds with the code.

### Design Principles

- **Minimality Principle**: The fundamental goal is to enable/verify all integration paths/mechanisms for a new backend to integrate to PyTorch. All functions follow a "just right" strategy to ensure the correctness of relevant integration capabilities.
- **Authenticity Principle**: To complete the OpenReg integration in the same way a real accelerator backend would integrate with PyTorch.

## Directory Structure

```shell
torch_openreg/
├── CMakeLists.txt
├── csrc
│   ├── aten
│   │   ├── native
│   │   │   ├── Extra.cpp
│   │   │   ├── Minimal.cpp
│   │   │   └── ...
│   │   ├── OpenRegExtra.cpp
│   │   └── OpenRegMinimal.cpp
│   ├── CMakeLists.txt
│   └── runtime
│       ├── OpenRegDeviceAllocator.cpp
│       ├── OpenRegDeviceAllocator.h
│       ├── OpenRegFunctions.cpp
│       ├── OpenRegFunctions.h
│       ├── OpenRegGenerator.cpp
│       ├── OpenRegGenerator.h
│       ├── OpenRegGuard.cpp
│       ├── OpenRegGuard.h
│       ├── OpenRegHooks.cpp
│       ├── OpenRegHooks.h
│       ├── OpenRegHostAllocator.cpp
│       ├── OpenRegHostAllocator.h
│       └── ...
├── pyproject.toml
├── README.md
├── setup.py
├── third_party
│   └── openreg
└── torch_openreg
    ├── csrc
    │   ├── CMakeLists.txt
    │   ├── Module.cpp
    │   └── stub.c
    ├── __init__.py
    └── openreg
        ├── __init__.py
        └── random.py
```

**Dependencies**:

```mermaid
graph LR
    A[Python]
    B[_C.so]
    C[libtorch_bindings.so]
    D[libtorch_openreg.so]
    E[libopenreg.so]

    A --> B --> C --> D --> E
```

There are 4 DSOs in torch_openreg, and the dependencies between them are as follows:

- `_C.so`:
  - **sources**: torch_openreg/csrc/stub.c
  - **description**: Python C module entry point.
- `libtorch_bindings.so`: The bridging code between Python and C++ should go here.
  - **sources**: torch_openreg/csrc
  - **description**: A thin glue layer between Python and C++.
- `libtorch_openreg.so`: All core implementations should go here.
  - **sources**: csrc
  - **description**: All core functionality, such as device runtime, operators, etc.
- `libopenreg.so`: A DSO that uses the CPU to emulate a CUDA-like device, you can ignore it.
  - **sources**: third_party/openreg
  - **description**: Provides low-level device functionality similar to libcudart.so.

**Key Directories**:

- `csrc/`: Core device implementation, including operator registration, runtime, etc.
  - `csrc/aten/`: Operator registration
    - `csrc/aten/native/`: Specific operator implementations for the OpenReg device.
      - `csrc/aten/OpenRegMinimal.cpp`: The most minimal set of operator implementations (allowing for the creation of Tensors and related operations upon completion).
      - `csrc/aten/OpenRegExtra.cpp`: Implementations for other types of operators.
    - `csrc/runtime/`: Implementations for Host memory, device memory, Guard, Hooks, etc.
- `third_party/`: A C++ library that simulates a CUDA-like device using the CPU.
- `torch_openreg/`: Python interface implementation (Python code and C++ Bindings).
  - `torch_openreg/csrc/`: Python C++ binding code.
  - `torch_openreg/openreg/`: Python API.

## Currently Implemented Features

### Operator Registration

- Operator Implementation

  - `TORCH_LIBRARY` form
    - Registering a specific operator for an existing schema: See `empty.memory_format`
    - Registering an operator with a custom schema
      - Extending an existing namespace: (TODO)
      - Custom namespace: See `custom_autograd_fn_returns_self`
    - Autograd: See `custom_autograd_fn_returns_self`
  - STUB form: See `abs_stub`

  - Fallback
    - Global Fallback: See `wrapper_cpu_fallback`
    - Per-operator Fallback: (TODO)

  - AMP (TODO)

### Memory Management

- Device Memory Management (TODO)
- Host Memory Management (TODO)

### Custom Storage

- Adding custom device descriptions (TODO)
- Serialization support (TODO)

### Autoload

#### Background

The **Autoload** mechanism in PyTorch is designed to enable seamless, on-demand registration and initialization of third-party device backends. Traditionally, integrating a new accelerator backend required explicit user imports or manual initialization code, which could be error-prone and inconvenient. With Autoload, PyTorch can automatically discover and initialize device backends at runtime, improving user experience and reducing integration friction.

The design of Autoload leverages Python entry points (such as `torch.backends`) and dynamic module loading. When PyTorch starts, it scans for registered entry points and invokes the corresponding initialization hooks, ensuring that all available device backends are properly registered and ready for use—without requiring users to import backend-specific Python modules manually.

#### How to Enable Autoload for OpenReg

##### 1. Implement an Initialization Hook

```python
# torch_openreg/__init__.py

import torch
import torch_openreg._C  # type: ignore[misc]
import torch_openreg.openreg

# Loading torch_openreg module here
torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", torch_openreg.openreg)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

# Initialization hook for autoload entry point
def _autoload():
    pass
```

##### 2. Register the Entry Point

```python
# torch_openreg/setup.py

# Register the autoload entry point. When PyTorch starts,
# it scans for _autoload function under torch_openreg package and invokes it.
setup(
    ...
    entry_points={
        "torch.backends": [
            "torch_openreg = torch_openreg:_autoload",
        ],
    },
    ...
)
```

##### 3. Build and Run OpenReg with Autoload

Build and install OpenReg package (see [Installation](#installation)). The entry point will be registered in the Python environment. When PyTorch starts, it will automatically discover and invoke _autoload function via the entry point. This means we can immediately use OpenReg as our device backend without any explicit import:

```python

import torch

# No need to import torch_openreg manually!
x = torch.tensor([1, 2, 3], device="openreg")
print(x)
```

...

## Installation and Usage

### Installation

```python
pip3 install --no-build-isolation -e . # for develop
pip3 install --no-build-isolation . # for install
```

### Usage Example

After installation, you can use the `openreg` device in Python just like any other regular device.

```python
import torch

if not torch.openreg.is_available():
    print("OpenReg backend is not available in this build.")
    exit()

print("OpenReg backend is available!")

device = torch.device("openreg")

x = torch.tensor([[1., 2.], [3., 4.]], device=device)
y = x + 2
print("Result y:\n", y)
print(f"Device of y: {y.device}")

z = y.cpu()
print("Result z:\n", z)
print(f"Device of z: {z.device}")
```

## Future Plans

- **Enhance Features**: AMP, memory management, generators, distributed computing, etc. (to reiterate, the fundamental goal is to verify the integration mechanism).
- **Improve Tests**: Add more test cases related to the integration mechanism.
- **Improve Documentation**: Add a new chapter on third-party device integration in the `Developer Notes` section of the PyTorch documentation.
- **Real-time Synchronization**: Keep the code and documentation updated iteratively and in sync.
