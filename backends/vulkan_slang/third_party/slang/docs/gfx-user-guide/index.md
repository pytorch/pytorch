---
layout: user-guide
---

Slang Graphics Layer
=============

The Slang Graphics Layer is an abstraction library of graphics APIs to support cross-platform applications that utilize GPU graphics/compute capabilities. The Slang Graphics Layer tightly integrates the Slang shading language to provide the most complete cross-platform GPU application development experience. The Slang language and compilation API is designed to work best when the application assumes several best practices in terms of shader specialization and parameter binding. The Slang Graphics Layer is following exactly the same best practices supported by Slang's compilation model. Outside of shader-related areas, the graphics layer's interface is designed to closely follow the modern graphics API models in Direct3D 12, Vulkan and Metal, such that the layer is only purposed to abstracting the differences between these underlying APIs instead of providing a higher level abstract that simplifies the interface. This design philosophy allows users to benefit from the ideas in the Slang shading language without giving up precise control on other aspects of the graphics API.

The current support status of operating system and graphics APIs is shown in the following matrix.

|               | Windows            | Linux              |
| :------------ | :----------------: | :----------------: |
| Direct3D 12   | Yes                | No                 |
| Direct3D 11   | Yes                | No                 |
| Vulkan        | Yes                | Yes                |
| OpenGL        | Yes                | No                 |
| CPU emulation | Yes (Compute Only) | Yes (Compute Only) |
| CUDA          | Yes (Compute Only) | Yes (Compute Only) |


> #### Note
> The graphics layer is still under active development and we intend to add more platforms and APIs in the future.

In this documentation, we will walk through various parts of the library and demonstrate how it can be used in your application.