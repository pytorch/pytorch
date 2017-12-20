# THNN

THNN is a library that gathers nn's C implementations of neural network modules. It's entirely free of Lua dependency and therefore can be used in any application that has a C FFI. Please note that it only contains quite low level functions; most users will want to use ATen, which provides a C++ wrapper around these functions.

There is also a CUDA counterpart of THNN, THCUNN.

Looking to add an implementation?  Consider writing an ATen native function
instead!  See [../ATen/native](ATen/native).

## Links

* [API reference](doc/api_reference.md)
* [Style guidelines](doc/style_guidelines.md)

## API

THNN is a purely functional library. It provides 2-3 functions for each module, that perform the most important operations:

* **updateOutput** - applies the module to an input
* **updateGradInput** - accepts gradient w.r.t. output and previous module input, and computes a gradient w.r.t. that input
* **accGradParameters** - *(optional, only modules with parameters)* accepts gradient w.r.t. output and previous module input, and computes gradient w.r.t. the parameters

For information on argument types please check the [API reference](doc/api_reference.md).

## Developer docs

* [Style guidelines](doc/style_guidelines.md)
