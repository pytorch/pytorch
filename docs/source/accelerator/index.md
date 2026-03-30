# Accelerator Integration

Since PyTorch 2.1, the community has made significant progress in streamlining the process of integrating new accelerators into the PyTorch ecosystem. These improvements include, but are not limited to: refinements to the `PrivateUse1` Dispatch Key, the introduction and enhancement of core subsystem extension mechanisms, and the device-agnostic refactoring of key modules (e.g., `torch.accelerator`, `memory management`). Taken together, these advances provide the foundation for a **robust**, **flexible**, and **developer-friendly** pathway for accelerator integration.

```{note}
This guide is a work in progress. For more details, please refer to the [roadmap](https://github.com/pytorch/pytorch/issues/158917).
```

## Why Does This Matter?

This integration pathway offers several major benefits:

* **Speed**: Extensibility is built into all core PyTorch modules. Developers can integrate new accelerators into their downstream codebases independently—without modifying upstream code and without being limited by community review bandwidth.
* **Future-proofing**: This is the default integration path for all future PyTorch features, meaning that as new modules and features are added, they will automatically support scaling to new accelerators if this path is followed.
* **Autonomy**: Vendors maintain full control over their accelerator integration timelines, enabling fast iteration cycles and reducing reliance on upstream coordination.

## Target Audience

This document is intended for:

* **Accelerator Developers** who are integrating accelerator into PyTorch;
* **Advanced PyTorch Users** interested in the inner workings of key modules;

## About This Document

This guide aims to provide a **comprehensive overview of the modern integration pathway** for new accelerator in PyTorch. It walks through the full integration surface, from low-level device primitives to higher-level domain modules like compilation and quantization. The structure follows a **modular and scenario-driven approach**, where each topic is paired with corresponding code examples from [torch_openreg][OpenReg URL], an official reference implementation, and this series is structured around four major axes:

* **Runtime**: Covers core components such as Event, Stream, Memory, Generator, Guard, Hooks, as well as the supporting C++ scaffolding.
* **Operators**: Involve the minimum necessary set of operators, forward and backward operators, fallback operators, fallthroughs, STUBs, etc. in both C++ and Python implementations.
* **Python Frontend**: Focuses on Python bindings for modules and device-agnostic APIs.
* **High-level Modules**: Explores integration with major subsystems such as `AMP`, `Compiler`, `ONNX`, and `Distributed` and so on.

The goal is to help developers:

* Understand the full scope of accelerator integration;
* Follow best practices to quickly launch new accelerators;
* Avoid common pitfalls through clear, targeted examples.

Next, we will delve into each chapter of this guide. Each chapter focuses on a key aspect of integration, providing detailed explanations and illustrative examples. Since some chapters build upon previous ones, readers are encouraged to follow the sequence to achieve a more coherent understanding.

```{toctree}
:glob:
:maxdepth: 1

device
hooks
guard
autoload
operators
amp
profiler
```

[OpenReg URL]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg "OpenReg URL"
