Model Viewer Example
====================

This example expands on the simple Slang API integration from the "Hello, World" example by actually loading and rendering model data with extremely basic surface and light shading.

This time, the shader code is making use of various Slang language features, so readers may want to read through `shaders.slang` to see an example of how the various mechanisms can be used to build out a more complicated shader library.
While the shader code in this example is still simplistic, it shows examples of:

* Using multiple Slang `ParameterBlock`s to manage the space of shader parameter bindings in a graphics-API-independent fashion, while still taking advantage of the performance opportunities afforded by D3D12 and Vulkan.

* Using `interface`s and generics to express multiple variations of a feature with static specialization, in place of more traditional preprocessor techniques.

The application code in `main.cpp` also shows a more advanced integration of the Slang API than that in the "Hello, World" example, including examples of:

* Loading a library of Slang shader code to perform reflection on its types *without* specifying a particular entry point to generate code for

* Using Slang's reflection information to allocate graphics-API objects to implement parameter blocks (e.g., D3D12/Vulkan descriptor tables/sets)

* Performing on-demand specialization of Slang's generics using type information from parameter blocks to achieve simple shader specialization

It is perhaps worth taking note of the two things this example intentionally does *not* do:

* There is no use of the C-style preprocessor in the shader code presented, in order to demonstrate that shader specialization can be achieved without preprocessor techniques.

* There is no use of explicit parameter binding decorations (e.g., HLSL `regsiter` or GLSL `layout` modifiers), in order to demonstrate that these are not needed in order to achieve high-performance shader parameter binding.
