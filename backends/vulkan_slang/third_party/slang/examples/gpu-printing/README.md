GPU Printing
============

This example demonstrates how supporting for printing formatted messages from GPU shader code can be implemented in application code, using language and API features provided by Slang.

Overview
--------

If you want to read the code here, start with `kernels.slang`, which contains a compute shader entry point showing how simple printing operations in shader code can be made.
Once you see the client code, you will probably want to understand the implementation, so that you can add these features to your own codebase.

The GPU/shader part of the implementation resides in `printing.slang`, which provides a stand-alone Slang module intended to be brought into your code with `import`.
The comments in that file explain how the low-level implementaiton encoding of print data into buffers is performed, and then also shows how Slang language mechanisms can be used to wrap that low-level implementation in usable and extensible syntax.

The CPU part of the implementation resides in `gpu-printing.{h,cpp}`, which are responsible for taking GPU-generated buffers encoded by the code above, and translating it into host-side calls to C `printf()` and other console printing operations.
The CPU code also shows how to use the Slang reflection API to extract information from a compiled program to enable printing of strings by their hash codes.

The `main.cpp` file implements a small host application that loads the compute shader and executes it using the D3D11 API.
The code in this file is not especially relevant to the printing system.

Adding printing support to your own codebase
--------------------------------------------

The code in this example is meant to provide a starting point for applications/frameworks/engines that want to allow shader code to print messages, for debugging logging, etc.
You can start by copying the `gpu-printing.{h,cpp}`, `gpu-printing-ops.h`, and `printing.slang` files into your codebase, and then modifying them to meet your needs.

The implementation presented here is not feature-complete, so you may want to extend and customize it by:

* Adapting it to use the graphics API or wrapper layer appropriate to your codebase

* Making more GPU data types printable (including types specific to your application)

* Adding overloads of `println()` and `printf()` to support more arguments

* Customizing the encoding of print commands to make better use of space based on application-specific constraints

* Handling extended `printf()` formatting (width, precision, etc.) in the CPU code

Caveats
-------

This code is not battle-tested, and it makes no promises about security.
It is probable that a malformed or malicious GPU shader could write data into the "print buffer" that causes the CPU code to invoke `printf()` or other C standard library functions with invalid arguments.

In this implementation, GPU printing commands are only "flushed" by the CPU on draw/dispatch boundaries.
This means that the printing approach here cannot easily be used to diagnose deadlocks, infinite loops, or hardware/driver crashes.
Extending the implementation to better support such cases would likely depend on using platform- or hardware-specific knowledge or functionality.
