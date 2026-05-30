# Compiler Integration (torch.compile)

## Background

`torch.compile` is PyTorch's compiler stack for optimizing model execution. It provides two integration paths for out-of-tree accelerators:

- **Dynamo** – A Python-level tracing frontend that captures user code into FX graphs.
- **Inductor** – The default compiler backend that lowers FX graphs into optimized, fused kernels.

PyTorch exposes registration APIs at both layers, allowing out-of-tree accelerator vendors to integrate their device with the full compiler stack without modifying upstream code.

| Path | What It Enables | When to Use |
| :--- | :--- | :--- |
| **Dynamo Backend** | `torch.compile(backend="your_device")` — custom graph-level compilation | When you have your own compiler/codegen pipeline |
| **Inductor Backend** | `torch.compile(backend="inductor")` works on your device — full Inductor optimizations | When you want to leverage Inductor's fusion, memory planning, and kernel generation |

```{note}
[OpenReg][OpenReg URL] (`torch_openreg`) is PyTorch's official reference implementation for out-of-tree accelerator integration. Code examples in this chapter reference the OpenReg implementation as a concrete illustration of each step.
```

## Before You Start

Before following this guide, make sure you have:

- **An importable `torch_foo` extension package** that registers your device via `PrivateUse1`. See the earlier chapters in this guide for device registration, operators, and runtime hooks.
- **Operator implementations** sufficient for the workloads you intend to compile. See the [Operators](operators.md) chapter.
- **Device management primitives** (set device, get current device, device count) exposed to Python via your extension's `_C` module.

## Part 1: Dynamo Integration

Dynamo integration enables `torch.compile` to trace Python code into FX graphs and execute them on your device. This is the minimum required for `torch.compile` support.

### Registration APIs

Two registrations are required:

| API | Module | Purpose |
| :--- | :--- | :--- |
| `register_backend()` | `torch._dynamo.backends.registry` | Registers a named backend callable for `torch.compile(backend="name")` |
| `register_interface_for_device()` | `torch._dynamo.device_interface` | Registers a `DeviceInterface` so Dynamo can manage your device during tracing |

### Step 1: Implement a DeviceInterface

Subclass `torch._dynamo.device_interface.DeviceInterface` and implement the methods Dynamo needs for device management during compilation:

| Method | Description |
| :--- | :--- |
| `current_device()` | Return the current device index |
| `set_device(device)` | Set the active device |
| `device_count()` | Return number of available devices |
| `is_available()` | Return whether the device backend is available |
| `synchronize()` | Synchronize the device (can be no-op for synchronous devices) |
| `exchange_device(device)` | Set device and return previous device index |
| `maybe_exchange_device(device)` | Like `exchange_device` but no-op for negative indices |
| `get_raw_stream(device_idx)` | Return a raw stream handle (return 0 if not applicable) |
| `Worker.set_device(device)` | Worker-thread device setter (used by Dynamo's thread pool) |
| `Worker.current_device()` | Worker-thread current device getter |

### Step 2: Implement a Backend Function

The backend function receives a traced `GraphModule` and example inputs, and must return a callable:

```python
def my_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    # Optional: validate graph, apply transforms, optimize
    # Must return a callable that executes the graph
    return gm.forward
```

The simplest backend just returns `gm.forward` directly — Dynamo handles the tracing, and your existing eager-mode operators handle execution. More advanced backends can perform graph transformations, lowering, or dispatch to a vendor-specific compiler.

### Step 3: Register Both

Call both registration APIs at module load time. See the OpenReg reference for a complete example:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/compiler.py
    :language: python
    :linenos:
    :caption: DeviceInterface and backend registration (compiler.py)
```

### What the Dynamo Backend Gives You

With just the Dynamo backend:

- `torch.compile(backend="your_device")` traces and executes graphs on your device
- Graph breaks are handled automatically (multiple subgraphs)
- Dynamic shapes work via Dynamo's symbolic tracing
- Autograd integration (backward pass compilation) works
- FakeTensor mode correctly propagates your device type

## Part 2: Inductor Integration

Inductor integration enables `torch.compile(backend="inductor")` to generate optimized, fused kernels for your device. This gives vendors access to Inductor's full optimization pipeline: operator fusion, memory planning, loop optimization, and more.

### Registration APIs

| API | Module | Purpose |
| :--- | :--- | :--- |
| `register_backend_for_device()` | `torch._inductor.codegen.common` | Registers scheduling and wrapper codegen classes |
| `register_device_op_overrides()` | `torch._inductor.codegen.common` | Registers device-specific code snippets for generated code |

### Concepts

#### Scheduling

The **scheduling class** controls how Inductor groups operations into fused kernels and decides execution order. PyTorch provides:

- `CppScheduling` — Generates C++ kernels for CPU-like execution
- `TritonScheduling` — Generates Triton kernels for GPU-like execution

#### Wrapper Codegen

The **wrapper codegen class** generates the Python host-side code that orchestrates kernel launches. It controls:

- How kernels are called from Python
- Device/stream management around kernel calls
- Any pre/post-processing needed (e.g., memory management)

Subclass `PythonWrapperCodegen` to customize this behavior. Key methods to override:

| Method | Purpose |
| :--- | :--- |
| `create()` | Factory method — must handle both regular and subgraph cases |
| `write_header()` | Add custom imports to the generated code header |
| `generate_kernel_call()` | Wrap or modify how each kernel is called |
| `_generate_kernel_call_helper()` | Control low-level kernel dispatch (e.g., device remapping) |

#### DeviceOpOverrides

The `DeviceOpOverrides` class provides code **snippets** that Inductor embeds into generated wrapper code. Each method returns a string of Python code:

| Method | What the Returned Code Should Do |
| :--- | :--- |
| `import_get_raw_stream_as(name)` | Define a `get_raw_stream` function for stream management |
| `set_device(device_idx)` | Set the active device |
| `synchronize()` | Synchronize the device |
| `device_guard(device_idx)` | Enter a device guard context manager |
| `cpp_kernel_type()` | Return the C++ type for kernel function pointers |

For devices without stream semantics, these can return no-op implementations (e.g., `"pass"` for synchronize, a null context for device guard).

### Step 1: Implement DeviceOpOverrides

Subclass `DeviceOpOverrides` and implement the methods that return code snippets for your device:

```python
from torch._inductor.codegen.common import DeviceOpOverrides

class MyDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return """
def get_raw_stream(_):
    return 0
"""

    def set_device(self, device_idx):
        return f"my_extension._C._set_device({device_idx})"

    def synchronize(self):
        return "my_extension._C._synchronize()"

    def device_guard(self, device_idx):
        return f"my_extension.device_guard({device_idx})"

    def cpp_kernel_type(self):
        return "void*"
```

### Step 2: Implement Wrapper Codegen (if needed)

If your device requires special handling around kernel calls (e.g., memory management, stream selection), subclass `PythonWrapperCodegen`:

```python
from torch._inductor.codegen.wrapper import PythonWrapperCodegen

class MyWrapperCodegen(PythonWrapperCodegen):
    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        if is_subgraph:
            from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return MyWrapperCodegen()

    def write_header(self):
        super().write_header()
        self.header.splice("import my_extension")

    def generate_kernel_call(self, kernel_name, call_args, *, device=None, **kwargs):
        # Add pre/post kernel call logic if needed
        super().generate_kernel_call(kernel_name, call_args, device=device, **kwargs)
```

### Step 3: Register with Inductor

Call both registration APIs at module level in a dedicated module:

```python
from torch._inductor.codegen.common import (
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpp import CppScheduling

register_device_op_overrides("my_device", MyDeviceOpOverrides())
register_backend_for_device(
    "my_device",
    CppScheduling,           # or your custom scheduling class
    MyWrapperCodegen,        # or PythonWrapperCodegen if no customization needed
)
```

### Reference Implementation

The OpenReg reference demonstrates the full Inductor integration, including memory protection handling specific to its simulated device:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/inductor_backend.py
    :language: python
    :linenos:
    :caption: OpenReg Inductor integration (inductor_backend.py)
```

### `register_backend_for_device()` Optional Parameters

| Parameter | When to Use |
| :--- | :--- |
| `device_cpp_wrapper_codegen` | When supporting AOTInductor (ahead-of-time C++ wrapper generation) |
| `device_fx_wrapper_codegen` | When supporting FX-based wrapper generation |
| `device_custom_pass` | When you need a custom graph-level optimization pass for your device |
| `device_custom_config` | When you need device-specific Inductor configuration overrides |

### Usage

After registration, Inductor works transparently on your device:

```python
import torch
import my_extension.inductor_backend  # Triggers registration

@torch.compile(backend="inductor")
def fn(x, y):
    return torch.relu(x * 2 + y)

x = torch.randn(4, device="my_device")
y = torch.randn(4, device="my_device")
result = fn(x, y)  # Inductor generates fused kernel for your device
```

## Testing

### Dynamo Tests

- Verify your backend appears in `torch._dynamo.backends.registry.list_backends()`
- Test `torch.compile(backend="your_device")` produces correct results vs eager
- Test graph breaks, dynamic shapes, autograd, and `nn.Module` compilation
- Verify `DeviceInterface` methods return valid values

### Inductor Tests

- Verify registration with `get_scheduling_for_device()` and `get_wrapper_codegen_for_device()`
- Test `torch.compile(backend="inductor")` produces correct results vs eager on your device
- Verify output tensors preserve the correct device type
- Import your Inductor module lazily inside test methods to avoid import-order issues with native extensions

See the [OpenReg compiler test suite][test_compile.py] for a comprehensive reference covering both Dynamo and Inductor testing patterns.

[OpenReg URL]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg "OpenReg URL"
[test_compile.py]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_compile.py "test_compile.py"