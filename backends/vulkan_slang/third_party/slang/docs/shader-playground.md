Using Slang on Shader Playground
================================

A fast and simple way to try out Slang is by using the [Shader Playground](http://shader-playground.timjones.io/) website. This site allows easy and interactive testing of shader code across several compilers including Slang without having to install anything on your local machine.

Using the Slang compiler is as simple as selecting 'Slang' from the box in the top left corner from the [Shader Playground](http://shader-playground.timjones.io/). This selects the Slang language for input, and the Slang compiler for compilation. The output of the compilation is shown in the right hand panel.

The default 'Output format' is HLSL. For graphics shaders the 'Output format' can be changed to

* DXIL 
* SPIR-V
* DXBC
* HLSL
* GLSL

Additionally for compute based shaders it can be set to

* C++
* CUDA
* PTX

For binary formats (such as DXIL/SPIR-V/DXBC) the output will be displayed as the applicable disassembly.

Note that C++ and CUDA output include a 'prelude'. The prelude remains the same across compilations, with the code generated for the input Slang source placed at the very end of the output. 

