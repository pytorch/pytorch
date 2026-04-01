Frequently Asked Questions
==========================

### How did this project start?

The Slang project forked off from the ["Spire"](https://github.com/spire-lang/spire) shading language research project.
In particular, Slang aims to take the lessons learned in that research effort (about how to make more productive shader compilation languages and tools) and apply them to a stystem that is easier to adopt, and hopefully more amenable to production use.

### Why should I use Slang instead of glslang, hlsl2glslfork, the Microsoft open-source HLSL compiler, etc.?

If you are mostly just shopping around for a tool to get HLSL shaders working on other graphics APIs, then [this](http://aras-p.info/blog/2014/03/28/cross-platform-shaders-in-2014/) blog post is probably a good place to start.

If one of those tools meets your requirements, then you should probably use it.
Slang is a small project, and early in development, so you might find that you hit fewer bumps in the road with one of the more established tools out there.

The goal of the Slang project is not to make "yet another HLSL-to-GLSL translator," but rather to create a shading language and supporting toolchain that improves developer productivity (and happiness) over the existing HLSL language and toolchain, while providing a reasonable adoption path for developers who have an existing investment in HLSL shader code.
If you think that is something interesting and worth supporting, then please get involved!

### What would make a shading language more productive?

This is probably best answered by pointing to the most recent publication from the Spire research project:

[Shader Components: Modular and High Performance Shader Development](http://graphics.cs.cmu.edu/projects/shadercomp/)

Some other papers for those who would like to read up on our inspiration:

[A System for Rapid Exploration of Shader Optimization Choices](http://graphics.cs.cmu.edu/projects/spire/)
[Spark: Modular, Composable Shaders for Graphics Hardware](https://graphics.stanford.edu/papers/spark/)

### Who is using Slang?

Right now the only user of Slang is the [Falcor](https://github.com/NVIDIA/Falcor) real-time rendering framework developed and used by NVIDIA Research.
The implementation of Slang has so far focused heavily on the needs of Falcor.

### Won't we all just be using C/C++ for shaders soon?

The great thing about both Vulkan and D3D12 moving to publicly-documented binary intermediate languages (SPIR-V and DXIL, respectively) is that there is plenty of room for language innovation on top of these interfaces.

Having support for writing GPU shaders in a reasonably-complete C/C++ language would be great.
We are supportive of efforts in the "C++ for shaders" direction.

The Slang effort is about trying to solve the challenges that are unique to the real-time graphics domain, and that won't magically get better by switching to C++.
