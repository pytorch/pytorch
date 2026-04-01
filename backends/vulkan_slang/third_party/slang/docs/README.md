Slang Documentation
===================

This directory contains documentation for the Slang system.
Some of the documentation is intended for users of the language and compiler, while other documentation is intended for developers contributing to the project.

Getting Started
---------------

The Slang [User's Guide](https://shader-slang.github.io/slang/user-guide/) provides an introduction to the Slang language and its major features, as well as the compilation and reflection API.

There is also documentation specific to using the [slangc](https://shader-slang.github.io/slang/user-guide/compiling.html#command-line-compilation-with-slangc) command-line tool.

Advanced Users
--------------

For the benefit of advanced users we provide detailed documentation on how Slang compiles code for specific platforms.
The [target compatibility guide](target-compatibility.md) gives an overview of feature compatibility for targets. 

The [CPU target guide](cpu-target.md) gives information on compiling Slang or C++ source into shared libraries/executables or functions that can be directly executed. It also covers how to generate C++ code from Slang source.  

The [CUDA target guide](cuda-target.md) provides information on compiling Slang/HLSL or CUDA source. Slang can compile to equivalent CUDA source, as well as to PTX via the nvrtc CUDA compiler.

Contributors
------------

For contributors to the Slang project, the information under the [`design/`](design/) directory may help explain the rationale behind certain design decisions and help when ramping up in the codebase.

Research
--------

The Slang project is based on a long history of research work. While understanding this research is not necessary for working with Slang, it may be instructive for understanding the big-picture goals of the language, as well as why certain critical decisions were made.

A [paper](http://graphics.cs.cmu.edu/projects/slang/) on the Slang system was accepted into SIGGRAPH 2018, and it provides an overview of the language and the compiler implementation.
Yong He's [dissertation](http://graphics.cs.cmu.edu/projects/renderergenerator/yong_he_thesis.pdf) provided more detailed discussion of the design of the Slang system.
