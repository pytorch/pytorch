# THNN

THNN is a library that gathers nn's C implementations of neural network modules. It's entirely free of Lua dependency and therefore can be used in any application that has a C FFI. Please note that it only contains quite low level functions, and an object oriented C/C++ wrapper will be created soon as another library.

There is also a CUDA counterpart of THNN (THCUNN) in the [cunn repository](https://github.com/torch/cunn/tree/master/lib/THCUNN).

## Links

* [API reference](doc/api_reference.md)
* [Style guidelines](doc/style_guidelines.md)

## Motivation

Torch's neural network package (nn) provided many optimized C implementations of modules, but the source files contained Lua specific code and headers so they couldn't be easily compiled and included anywhere else.

THNN is based on the same code, but is written in pure C, so it can be easily included in other code. **Future C implementations should be committed to THNN.**

## API

THNN is a purely functional library. It provides 2-3 functions for each module, that perform the most important operations:

* **updateOutput** - applies the module to an input
* **updateGradInput** - accepts gradient w.r.t. output and previous module input, and computes a gradient w.r.t. that input
* **accGradParameters** - *(optional, only modules with parameters)* accepts gradient w.r.t. output and previous module input, and computes gradient w.r.t. the parameters

For information on argument types please check the [API reference](doc/api_reference.md).

## Developer docs

* [Style guidelines](doc/style_guidelines.md)

This section will be expanded when FFI refactoring will be finished.
