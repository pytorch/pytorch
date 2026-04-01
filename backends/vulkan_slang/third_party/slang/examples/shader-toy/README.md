Slang "Shader Toy" Example
==========================

This example shows how to use Slang's support for generics and interfaces to define shader effects as separately-compiled modules.
The effects in this case are based on the popular [Shader Toy](https://www.shadertoy.com/) site.

Goals
-----

The big-picture goals of this example is to define effects as separately-compiled modules of Slang code, and to also allow different methods of executing those effects (e.g., via vertex/fragment shaders or compute shaders) to be defined as modules.
Each module should be something the Slang compiler can compile and check independently.

Combining modules (e.g., a particular shader effect with a particular execution method) should be accomplished using first-class operations supported by the Slang API, instead of by ad hoc preprocessing or pasting of strings.

Approach
--------

The key idea here is to codify the rules for what a shader toy effect needs to provide in an interface (here called `IShaderToyImageShader`), and to use that interface as a contract when checking both implementers and users of that interface.

Individual effects become `struct` types that declare conformance to `IShaderToyImageShader`; the compiler can thus issue error messages when a time fails to satisfy its requirements.
Execution methods become generic functions that abstract over any type that implements `IShaderToyImageShader`; the compiler can confirm that they only use operations that are guaranteed by the interface to be present.

Composition thus consists of "plugging in" a type that implements the interface as a type argument of the generic function that implements an execution method.

While the interfaces and modules that this example works with are relatively low in complexity, these same techniques can be applied to modularize more complex shader code without the need for preprocessor or metaprogramming tricks.
