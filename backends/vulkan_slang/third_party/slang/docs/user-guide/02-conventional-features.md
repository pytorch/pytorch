---
layout: user-guide
permalink: /user-guide/conventional-features
---

Conventional Language Features
==============================

Many of the language concepts in Slang are similar to those in other real-time shading languages like HLSL and GLSL, and also to general-purpose programming languages in the "C family."
This chapter covers those parts of the Slang language that are _conventional_ and thus unlikely to surprise users who are already familiar with other shading languages, or languages in the C family.

Readers who are comfortable with HLSL variables, types, functions, statements, as well as conventions for shader parameters and entry points may prefer to skip this chapter.
Readers who are not familiar with HLSL, but who are comfortable with GLSL and/or C/C++, may want to carefully read the sections on types, expressions, shader parameters, and entry points while skimming the others.

Types
-----

Slang supports conventional shading language types including scalars, vectors, matrices, arrays, structures, enumerations, and resources.

> #### Note ####
> Slang has limited support for pointers when targeting platforms with native pointer support, including SPIRV, C++, and CUDA.

### Scalar Types

#### Integer Types

The following integer types are provided:

| Name          | Description |
|---------------|-------------|
| `int8_t`      | 8-bit signed integer |
| `int16_t`     | 16-bit signed integer |
| `int`         | 32-bit signed integer |
| `int64_t`     | 64-bit signed integer |
| `uint8_t`     | 8-bit unsigned integer |
| `uint16_t`    | 16-bit unsigned integer |
| `uint`        | 32-bit unsigned integer |
| `uint64_t`    | 64-bit unsigned integer |

All targets support the 32-bit `int` and `uint` types, but support for the other types depends on the capabilities of each target platform.

Integer literals can be both decimal and hexadecimal. An integer literal can be explicitly made unsigned 
with a `u` suffix, and explicitly made 64-bit with the `ll` suffix. The type of a decimal non-suffixed integer literal is the first integer type from
the list [`int`, `int64_t`] which can represent the specified literal value. If the value cannot fit, the literal is represented as 
an `uint64_t` and a warning is given. The type of hexadecimal non-suffixed integer literal is the first type from the list 
[`int`, `uint`, `int64_t`, `uint64_t`] that can represent the specified literal value. For more information on 64 bit integer literals see the documentation on [64 bit type support](../64bit-type-support.md).

The following floating-point types are provided:

| Name          | Description                  |
|---------------|------------------------------|
| `half`        | 16-bit floating-point number |
| `float`       | 32-bit floating-point number |
| `double`      | 64-bit floating-point number |

All targets support the 32-bit `float`, but support for the other types depends on the capabilities of each target platform.

### Boolean Type

The type `bool` is used to represent Boolean truth values: `true` and `false`. 

For compatibility reasons, the `sizeof(bool)` depends on the target. 

| Target |      sizeof(bool)      |
|--------| ---------------------- |
| GLSL   | 4 bytes / 32-bit value |
| HLSL   | 4 bytes / 32-bit value |
| CUDA   | 1 bytes /  8-bit value |

> #### Note ####
> When storing bool types in structures, make sure to either pad host-side data structures accordingly, or store booleans as, eg, `uint8_t`, to guarantee
> consistency with the host language's boolean type.

#### The Void Type

The type `void` is used as a placeholder to represent the result type of functions that don't return anything.

### Vector Types

Vector types can be written as `vector<T,N>` where `T` is a scalar type and `N` is an integer from 2 to 4 (inclusive).
The type `vector<T,N>` is a vector of `N` _elements_ (also called _components_) each of type `T`.

As a convenience, pre-defined vector types exist for each scalar type and valid element count, with a name using the formula `<<scalar-type>><<element-count>>`.
For example, `float3` is a convenient name for `vector<float,3>`.

> Note: Slang doesn't support vectors longer than 4 elements. They map to native vector types on many platforms, including CUDA, and none of these platforms support vectors longer than 4 elements. If needed, you can use an array like `float myArray[8]`.

### Matrix Types

Matrix types can be written as `matrix<T,R,C>` where `T` is a scalar type and both `R` and `C` are integers from 2 to 4 (inclusive).
The type `matrix<T,R,C>` is a matrix with _elements_ of type `T`, and comprising `R` rows and `C` columns.

As a convenience, pre-defined matrix types exist for each scalar type and valid row/column count, with a name using the formula `<<scalar-type>><<row-count>>x<<column-count>>`.
For example, a `float3x4` is a convenient name for `matrix<float,3,4>`.

> #### Note ####
> Readers familiar with GLSL should be aware that a Slang `float3x4` represents a matrix with three rows and four columns, while a GLSL `mat3x4` represents a matrix with three *columns* and four *rows*.
> In most cases, this difference is immaterial because the subscript expression `m[i]` returns a `float4` (`vec4`) in either language.
> For now it is enough to be aware that there is a difference in convention between Slang/HLSL/D3D and GLSL/OpenGL.

### Array Types

An array type `T[N]` represents an array of `N` elements of type `T`.
When declaring a variable with an array type, the `[]` brackets come after the variable name, following the C convention for variable declarations:

```hlsl
// the type of `a` is `int[3]`
int a[3];
```

Sometimes a value with an array type can be declared without an explicit element count.
In some cases the element count is then inferred from the initial value of a variable:

```hlsl
// the type of `a` is `int[3]`
int a[] = { 1, 2, 3 };
```

In other cases, the result is a _unsized_ array, where the actual element count will be determined later:

```hlsl
// the type of `b` is `int[]`
void f( int b[] )
{ ... }
```

It is allowed to pass a sized array as argument to an unsized array parameter when calling a function.

Array types has a `getCount()` member function that returns the length of the array.

```hlsl
int f( int b[] )
{
    return b.getCount(); // Note: all arguments to `b` must be resolvable to sized arrays.
}

void test()
{
    int arr[3] = { 1, 2, 3 };
    int x = f(arr); // OK, passing sized array to unsized array parameter, x will be 3.
}
```

Please note that if a function calls `getCount()` method on an unsized array parameter, then all
calls to that function must provide a sized array argument, otherwise the compiler will not be able
to resolve the size and will report an error. The following code shows an example of valid and
invalid cases.

```hlsl
int f( int b[] )
{
    return b.getCount();
}
int g( int b[] )
{
    return f(b); // transitive calls are allowed.
}
uniform int unsizedParam[];
void test()
{
    g(unsizedParam); // Not OK, `unsizedParam` doesn't have a known size at compile time.
    int arr[3];
    g(arr); // OK.
}
```

There are more limits on how runtime-sized arrays can be used than on arrays of statically-known element count.

> #### Note ####
> In Slang arrays are _value types_, meaning that assignment, parameter passing, etc. semantically copy values of array type.
> In some languages -- notably C, C++, C#, and Java -- assignment and parameter passing for treat arrays as _reference types_,
> meaning that these operations assign/pass a reference to the same underlying storage.

### Structure Types

Structure types can be introduced with the `struct` keyword, as in most C-family languages:

```hlsl
struct MyData
{
    int a;
    float b;
}
```

> #### Note ####
> Unlike C, and like most other C-family languages, the `struct` keyword in Slang introduces a type directly, and there is no need to combine it with a `typedef`.

> #### Note ####
> Slang allows for a trailing semicolon (`;`) on `struct` declarations, but does not require it.

> #### Note ####
> Unlike C/C++, `class` is not a valid keyword for GPU code and it is reserved for CPU/host side logic.

Structure types can have constructors. Constructors are defined with the `__init` keyword:

```hlsl
struct MyData
{
     int a;
     __init() { a = 5; }
     __init(int t) { a = t; }
}
void test()
{
     MyData d;  // invokes default constructor, d.a = 5
     MyData h = MyData(4); // invokes overloaded constructor, h.a = 4
}
```

> #### Note ####
> Slang currently does not allow default values on struct members, but we intend to support them in the future.

### Enumeration Types

Enumeration types can be introduced with the `enum` keyword to provide type-safe constants for a range of values:

```hlsl
enum Channel
{
    Red,
    Green,
    Blue
}
```

Unlike C/C++, `enum` types in Slang are always scoped by default (like `enum class` in C++). You can write `enum class` in Slang if it makes you happy, but it isn't required. If you want a `enum` type to be unscoped, you can use the `[UnscopedEnum]` attribute:
```csharp
[UnscopedEnum]
enum Channel
{
    Red, Green, Blue
}
void test(Channel c)
{
    if (c == Red) { /*...*/ }
}
```

You can specify an explicit underlying integer type for `enum` types:
```csharp
enum Channel : uint16_t
{
    Red, Green, Blue
}
```

By default, the underlying type of an enumeration type is `int`. Enumeration types are implicitly convertible to their underlying type. All enumeration types conform to the builtin `ILogical` interface, which provides operator overloads for bit operations. The following code is allowed:

```csharp
void test()
{
    Channel c = Channel.Red | Channel.Green;
}
```

You can explicitly assign values to each enum case:
```csharp
enum Channel
{
    Red = 5,
    Green,   // = 6
    Blue     // = 7
}
```
Slang automatically assigns integer values to enum cases without an explicit value. By default, the value starts from 0 and is increment by 1 for each
enum case.

You can override the implicit value assignment behavior with the `[Flags]` attribute, which will make value assignment start from 1 and increment by power of 2, making it suitable for enums that represent bit flags. For example:
```csharp
[Flags]
enum Channel
{
    Red,   //  = 1
    Green, //  = 2
    Blue,  //  = 4
    Alpha, //  = 8
}
```

### Opaque Types

The Slang core module defines a large number of _opaque_ types which provide access to objects that are allocated via GPU APIs.

What all opaque types have in common is that they are not "first-class" types on most platforms.
Opaque types (and structure or array types that contain them) may be limited in the following ways (depending on the platform):

* Functions that return opaque types may not be allowed
* Global and `static` variables that use opaque types may not be allowed
* Opaque types may not appear in the element types of buffers, except where explicitly noted as allowed

#### Texture Types

Texture types -- including `Texture2D`, `TextureCubeArray`, `RWTexture2D`, and more -- are used to access formatted data for read, write, and sampling operations.
Textures can be used to represent simple images, but also support _mipmapping_ as a way to reduce noise when sampling at lower than full resolution.
The full space of texture types follows the formula:

    <<access>>Texture<<base shape>><<multisampleness>><<arrayness>><<element type>>

where:

* The _access_ can be read-only (no prefix), read-write (`RW`), or read-write with a guarantee of rasterization order for operations on the given resource (`RasterizerOrdered`).
* The _base shape_ can be `1D`, `2D`, `3D`, or `Cube`.
* The _multisample-ness_ can be non-multiple-sample, or multi-sampled (`MS`).
* The _array-ness_  can either be non-arrayed, or arrayed (`Array`).
* The _element type_ can either be explicitly specified (`<T>`) or left as the default of `float4`

Not all combinations of these options are supported, and some combinations may be unsupported on some targets.

#### Sampler

Sampler types encapsulate parameters that control addressing and filtering for texture-sampling operations.
There are two sampler types: `SamplerState` and `SamplerComparisonState`.
`SamplerState` is applicable to most texture sampling operations, while `SamplerComparisonState` is used for "shadow" texture sampling operations which compare texels to a reference value before filtering.

> #### Note ####
> Some target platforms and graphics APIs do not support separation of textures and sampling state into distinct types in shader code.
> On these platforms the Slang texture types include their own sampling state, and the sampler types are placeholder types that carry no data.

#### Buffers

There are multiple buffer types supported by modern graphics APIs, with substantially different semantics.

##### Formatted Buffers

Formatted buffers (sometimes referred to as "typed buffers" or "buffer textures") are similar to 1D textures (in that they support format conversion on loads), without support for mipmapping.
The formula for formatted buffer types is:

    <<access>>Buffer<<arrayness>><<element type>>

Where the _access_, _array-ness_, and _element type_ are the same as for textures, with the difference that _element type_ is not optional.

A buffer type like `Buffer<float4>` represents a GPU resource that stores one or more values that may be fetched as a `float4` (but might internally be stored in another format, like RGBA8).

##### Flat Buffers

Flat buffers differ from formatted buffers in that they do not support format conversion.
Flat buffers are either _structured_ buffers or _byte-addressed_ buffers.

Structured buffer types like `StructuredBuffer<T>` include an explicit element type `T` that will be loaded and stored from the buffer.
Byte-addressed buffer types like `ByteAddressBuffer` do not specify any particular element type, and instead allow for values to be loaded or stored from any (suitably aligned) byte offset in the buffer.
Both structured and byte-addressed buffers can use an _access_ to distinguish between read-only and read-write usage.

##### Constant Buffers

Constant buffers (sometimes also called "uniform buffers") are typically used to pass immutable parameter data from a host application to GPU code.
The constant buffer type `ConstantBuffer<T>` includes an explicit element type.
Unlike formatted or flat buffers, a constant buffer conceptually contains only a *single* value of its element type, rather than one or more values.

Expressions
-----------

Slang supports the following expression forms with nearly identical syntax to HLSL, GLSL, and C/C++:

* Literals: `123`, `4.56`, `false`

> #### Note ####
> Unlike C/C++, but like HLSL/GLSL, an unsuffixed floating-point literal has the `float` type in Slang, rather than `double`

* Member lookup: `structValue.someField`, `MyEnumType.FirstCase`

* Function calls: `sin(a)`

* Vector/matrix initialization: `int4(1, 2, 3, 4)`

* Casts: `(int)x`, `double(0.0)`

* Subscript (indexing): `a[i]`

* Initializer lists: `int b[] = { 1, 2, 3 };`

* Assignment: `l = r`

* Operators: `-a`, `b + c`, `d++`, `e %= f`

> #### Note ####
> Like HLSL but unlike most other C-family languages, the `&&` and `||` operators do *not* currently perform "short-circuiting". 
> they evaluate all of their operands unconditionally.
> However, the `?:` operator does perform short-circuiting if the condition is a scalar. Use of `?:` where the condition is a vector is deprecated in Slang. The vector version of `?:` operator does *not* perform short-circuiting, and the user is advised to call `select` instead.
> The default behavior of these operators is likely to change in a future Slang release.

Additional expression forms specific to shading languages follow.

### Operators on Vectors and Matrices

The ordinary unary and binary operators can also be applied to vectors and matrices, where they apply element-wise.

> #### Note ####
> In GLSL, most operators apply component-wise to vectors and matrices, but the multiplication operator `*` computes the traditional linear-algebraic product of two matrices, or a matrix and a vector.
> Where a GLSL programmer would write `m * v` to multiply a `mat3x4` by a `vec3`, a Slang programmer should write `mul(v,m)` to multiply a `float3` by a `float3x4`.
> In this example, the order of operands is reversed to account for the difference in row/column conventions.

### Swizzles

Given a value of vector type, a _swizzle_ expression extracts one or more of the elements of the vector to produce a new vector.
For example, if `v` is a vector of type `float4`, then `v.xy` is a `float2` consisting of the `x` and `y` elements of `v`.
Swizzles can reorder elements (`v.yx`) or include duplicate elements (`v.yyy`).

> #### Note ####
> Unlike GLSL, Slang only supports `xyzw` and `rgba` as swizzle elements, and not the seldom-used `stpq`.

> #### Note ####
> Unlike HLSL, Slang does not currently support matrix swizzle syntax.

Statements
----------

Slang supports the following statement forms with nearly identical syntax to HLSL, GLSL, and C/C++:

* Expression statements: `f(a, 3);`, `a = b * c;`

* Local variable declarations: `int x = 99;`

* Blocks: `{ ... }`

* Empty statement: `;`

* `if` statements

* `switch` statements

> #### Note ####
> Unlike C/C++, `case` and `default` statements must be directly nested under a `switch`, rather than being allowed under nested control flow (Duff's Device and similar idioms are not allowed).
> In addition, while multiple `case`s can be grouped together, all other forms of "fall through" are unsupported.

* `for` statements

* `while` statements

* `do`-`while` statements

* `break` statements

* `continue` statements

* `return` statements

* `defer` statements

> #### Note ####
> The `defer` statement in Slang is tied to scope. The deferred statement runs at the end of the scope like in Swift, not just at the end of the function like in Go.
> `defer` supports but does not require block statements: both `defer f();` and `defer { f(); g(); }` are legal.

> #### Note ####
> Slang does not support the C/C++ `goto` keyword.

> #### Note ####
> Slang does not support the C++ `throw` keyword.

Additional statement forms specific to shading languages follow.

### Discard Statements

A `discard` statement can be used in the context of a fragment shader to terminate shader execution for the current fragment, and to cause the graphics system to discard the corresponding fragment.

Functions
---------

Slang supports function definitions with traditional C syntax:

```hlsl
float addSomeThings(int x, float y)
{
    return x + y;
}
```

In addition to the traditional C syntax, you can use the modern syntax to define functions with the `func` keyword:
```swift
func addSomeThings(x : int, y : float) -> float
{
    return x + y;
}
```

Slang supports overloading of functions based on parameter types.

Function parameters may be marked with a _direction_ qualifier:

* `in` (the default) indicates a by-value input parameter
* `out` indicates an output parameter
* `inout` or `in out` indicates an input/output parameter

> #### Note ####
> The `out` and `inout` directions are superficially similar to non-`const` reference parameters in C++.
> In cases that do not involve aliasing of mutable memory, the semantics should be equivalent.

Preprocessor
------------

Slang supports a C-style preprocessor with the following directives;

* `#include`
* `#define`
* `#undef`
* `#if`, `#ifdef`, `#ifndef`
* `#else`, `#elif`
* `#endif`
* `#error`
* `#warning`
* `#line`
* `#pragma`, including `#pragma once`

Variadic macros are supported by the Slang preprocessor.

> #### Note ####
> The use of `#include` in new code is discouraged as this functionality has
> been superseded by the module system, please refer to
> [./04-modules-and-access-control.md](./04-modules-and-access-control.md)

Attributes
----------

_Attributes_ are a general syntax for decorating declarations and statements with additional semantic information or meta-data.
Attributes are surrounded with square brackets (`[]`) and prefix the declaration or statement they apply to.

For example, an attribute can indicate the programmer's desire that a loop be unrolled as much as possible:

```hlsl
[unroll]
for(int i = 0; i < n; i++)
{ /* ... */ }
```

> #### Note ####
> Traditionally, all attributes in HLSL used a single layer of `[]` brackets, matching C#.
> Later, C++ borrowed the idea from C# but used two layers of brackets (`[[]]`).
> Some recent extensions to HLSL have used the C++-style double brackets instead of the existing single brackets syntax.
> Slang tries to support both alternatives uniformly.

Global Variables and Shader Parameters
--------------------------------------

By default, global-scope variable declarations in Slang represent _shader parameters_ passed from host application code into GPU code.
Programmers must explicitly mark a global-scope variable with `static` for it not to be treated as a shader parameter, even if the variable is marked `const`:

```hlsl
// a shader parameter:
float a;

// also a shader parameter (despite `const`):
const int b = 2;

// a "thread-local" global variable
static int c = 3;

// a compile-time constant
static const int d = 4;
```

### Global Constants

A global-scope `static const` variable defines a compile-time constant for use in shader code.

### Global-Scope Static Variables

A non-`const` global-scope  `static` variable is conceptually similar to a global variable in C/C++, with the key difference that it has distinct storage per *thread* rather than being truly global.
Each logical thread of shader execution initiated by the GPU will be allocated fresh storage for these `static` variables, and values written to those variables will be lost when a shader thread terminates.

> #### Note ####
> Some target platforms do not support `static` global variables in all use cases.
> Support for `static` global variables should be seen as a legacy feature, and further use is discouraged.

### Global Shader Parameters

Global shader parameters may use any type, including both opaque and non-opaque types:

```hlsl
ConstantBuffer<MyData> c;
Texture2D t;
float4 color;
```

To avoid confusion, the Slang compiler will warn on any global shader parameter that includes non-opaque types, because it is likely that a user thought they were declaring a global constant or a traditional global variable.
This warning may be suppressed by marking the parameter as `uniform`:

```hlsl
// WARNING: this declares a global shader parameter, not a global variable
int gCounter = 0;

// OK:
uniform float scaleFactor;
```

#### Legacy Constant Buffer Syntax

For compatibility with existing HLSL code, Slang also supports global-scope `cbuffer` declarations to introduce constant buffers:

```hlsl
cbuffer PerFrameCB
{
    float4x4 mvp;
    float4 skyColor;
    // ...
}
```

A `cbuffer` declaration like this is semantically equivalent to a shader parameter declared using the `ConstantBuffer` type:

```hlsl
struct PerFrameData
{
    float4x4 mvp;
    float4 skyColor;
    // ...
}
ConstantBuffer<PerFrameData> PerFrameCB;
```

#### Explicit Binding Markup

For compatibility with existing codebases, Slang supports pre-existing markup syntax for associating shader parameters of opaque types with binding information for specific APIs.

Binding information for Direct3D platforms may be specified using `register` syntax:

```hlsl
Texture2D a : register(t0);
Texture2D b : register(t1, space0);
```

Binding information for Vulkan (and OpenGL) may be specified using `[[vk::binding(...)]]` attributes

```hlsl
[[vk::binding(0)]]
Texture2D a;

[[vk::binding(1, 0)]]
Texture2D b;
```

A single parameter may use both the D3D-style and Vulkan-style markup, but in each case explicit binding markup only applies to the API family for which it was designed.

> #### Note ####
> Explicit binding markup is tedious to write and error-prone to maintain.
> It is almost never required in Slang codebases.
> The Slang compiler can automatically synthesize bindings in a completely deterministic fashion and in most cases the bindings it generates are what a programmer would have written manually.

Shader Entry Points
-------------------

An _entry point_ is a function that can be used as the starting point for execution of a GPU thread.

Here is an example of an entry-point function in Slang:

```hlsl
[shader("vertex")]
float4 vertexMain(
    float3 modelPosition : POSITION,
    uint vertexID : SV_VertexID,
    uniform float4x4 mvp)
    : SV_Position
{ /* ... */ }
```

In the following sections we will use this example to explain important facets of entry point declarations in Slang.

### Entry Point Attribute and Stages

The `[shader(...)]` attribute is used to mark a function in Slang as a shader entry point, and also to specify which pipeline stage it is meant for.
In this example, the `vertexMain` shader indicates that it is meant for the `vertex` stage of the traditional rasterization pipeline.
Rasterization, compute, and ray-tracing pipelines each define their own stages, and new versions of graphics APIs may introduce new stages.

For compatibility with legacy codebases, Slang supports code that leaves off `[shader(...)]` attributes; in these cases application developers must specify the names and stages for their entry points via explicit command-line or API options.
Such entry points will not be found via `IModule::findEntryPointByName()`. Instead `IModule::findAndCheckEntryPoint()` must be used, and a stage must be specified.
It is recommended that new codebases always use `[shader(...)]` attributes both to simplify their workflow, and to make code more explicit and "self-documenting."

> #### Note ####
> In GLSL, a file of shader code may only include one entry point, and all code `#include`d into that file must be compatible with the stage of that entry point. By default, GLSL requires that an entry point be called `main`.
> Slang allows for multiple entry points to appear in a file, for any combination of stage, and with any valid identifier as a name.

### Parameters

The parameter of an entry-point function represent either _varying_ or _uniform_ inputs.
Varying inputs are those that may vary over threads invoked as part of the same batch (a draw call, compute dispatch, etc.), while uniform inputs are those that are guaranteed to be the same for all threads in a batch.
Entry-point parameters in Slang default to varying, but may be explicitly marked `uniform`.

If an entry-point function declares a non-`void` result type, then its result behaves like an anonymous `out` parameter that is varying.

### Binding Semantics

The varying parameters of an entry point must declare a _binding semantic_ to indicate how those parameters should be connected to the execution environment.
A binding semantic for a parameter may be introduced by suffixing the variable name with a colon (`:`) and an identifier for the chosen binding semantic.
A binding semantic for a function result is introduced similarly, but comes after the parameter list.

It is not shown in this example, but binding semantics may also be applied to individual `struct` fields, in cases where a varying parameter of `struct` type is used.

#### System-Defined Binding Semantics

In the `vertexMain` entry point, the `vertexID` parameter uses the `SV_VertexID` binding semantic, which is a _system-defined_ binding semantic.
Standard system-defined semantics are distinguished by the `SV_` prefix.

A system-defined binding semantic on an input parameter indicates that the parameter should receive specific data from the GPU as defined by the pipeline and stage being used.
For example, in a vertex shader the `SV_VertexID` binding semantic on an input yields the ID of the particular vertex being processed on the current thread.

A system-defined binding semantic on an output parameter or function result indicates that when a shader thread returns from the entry point the value stored in that output should be used by the GPU in a specific way defined by the pipeline and stage being used.
For example, in a vertex shader the `SV_Position` binding semantic on an output indicates that it represents a clip-space position that should be communicated to the rasterizer.

The set of allowed system-defined binding semantics for inputs and outputs depends on the pipeline and stage of an entry point.
Some system-defined binding semantics may only be available on specific targets or specific versions of those targets.

> #### Note ####
> Instead of using ordinary function parameters with system-defined binding semantics, GLSL uses special system-defined global variables with the `gl_` name prefix.
> Some recent HLSL features have introduced special globally-defined functions that behave similarly to these `gl_` globals.

#### User-Defined Binding Semantics

In the `vertexMain` entry point, the `modelPosition` parameter used the `POSITION` binding semantic, which is a _user-defined_ binding semantic.

A user-defined binding semantic on an input indicates that the parameter should receive data with a matching binding semantic from a preceding stage.
A user-defined binding semantic on an output indicates that the parameter should provide data to a parameter with a matching binding semantic in a following stage.

Whether or not inputs and outputs with user-defined binding semantics are allowed depends on the pipeline and stage of an entry point.

Different APIs and different stages within the same API may match up entry point inputs/outputs with user-defined binding semantics in one of two ways:

* By-index matching: user-defined outputs from one stage and inputs to the next are matched up by order of declaration. The types of matching output/input parameters must either be identical or compatible (according to API-specific rules). Some APIs also require that the binding semantics of matching output/input parameters are identical.

* By-name matching: user-defined outputs from one stage and inputs to the next are matched up by their binding semantics. The types of matching output/input parameters must either be identical or compatible (according to API-specific rules). The order of declaration of the parameters need not match.

Because the matching policy may differ across APIs, the only completely safe option is for parameters passed between pipeline stages to match in terms of order, type, *and* binding semantic.

> #### Note ####
> Instead of using ordinary function parameters for user-defined varying inputs/outputs, GLSL uses global-scope variable declarations marked with the `in` or `out` modifier.

### Entry-Point Uniform Parameters

In the `vertexMain` entry point, the `mvp` parameter is an _entry-point uniform parameter_.

Entry-point uniform parameters are semantically similar to global-scope shader parameters, but do not pollute the global scope.

> #### Note ####
> GLSL does not support entry-point `uniform` parameters; all shader parameters must be declared at the global scope.
> Historically, HLSL has supported entry-point `uniform` parameters, but this feature was dropped by recent compilers.

Mixed Shader Entry Points
--------------------------

Through the `[shader(...)]` syntax, users of slang can freely combine multiple entry points into the same file. This can be especially convenient for reuse between entry points which have a logical connection.

For example, mixed entry points offer a convenient way for ray tracing applications to concisely define a complete pipeline in one source file, while also providing users with additional opportunities to improve type safety of 
shared structure definitions:

```hlsl
struct Payload { float3 color; };

[shader("raygeneration")]
void rayGenerationProgram() {
    Payload payload;
    TraceRay(/*...*/, payload);
    /* ... */ 
}

[shader("closesthit")]
void closestHitProgram(out Payload payload) { 
    payload.color = {1.0};
}

[shader("miss")]
void missProgram(out Payload payload) { 
    payload.color = {1.0};
}
```

> #### Note ####
> GLSL does not support multiple entry-points; however, SPIR-V does. Vulkan users wanting to take advantage of Slang mixed entry points must pass `-fvk-use-entrypoint-name` and `-emit-spirv-directly` as compiler arguments.

### Mixed Entry-Point Uniform Parameters

Like with the previous `vertexMain` example, mixed entry point setups also support _entry-point uniform parameters_.

However, because of certain systematic differences between entry point types, a uniform being _global_ or _local_ will have very important consequences on the underlying layout and behavior.

For most all entry point types, D3D12 will use one common root signature to define both global and local uniform parameters. 
Likewise, Vulkan descriptors will bind to a common pipeline layout. For both of these cases, Slang maps uniforms to the common root signature / pipeline layout. 

However, for ray tracing entry points and D3D12, these parameters map to either _global_ root signatures or to _local_ root signatures, with the latter being stored in the shader binding table.
In Vulkan, D3D12's global root signatures translate to a shared ray tracing pipeline layout, while local root signatures map again to shader binding table records. 

When entry points match a "ray tracing" type, we bind uniforms which are in the _global_ scope to the _global_ root signature (or ray tracing pipeline layout), while uniforms which are _local_ are bound to shader binding table records, which depend on the underlying runtime record indexing. 

Consider the following:

```hlsl
uniform float3 globalUniform;

[shader("compute")][numThreads(1,2,3)]
void computeMain1(uniform float3 localUniform1) 
{ /* ... */ }

[shader("compute")][numThreads(1,2,3)]
void computeMain2(uniform float3 localUniform2) 
{ /* ... */ }

[shader("raygeneration")]
void rayGenerationMain(uniform float3 localUniform3) 
{ /* ... */ }

[shader("closesthit")]
void closestHitMain(uniform float3 localUniform4) 
{ /* ... */ }
```

In this example, `globalUniform` is appended to the global root signature / pipeline layouts for _both_ compute _and_ ray generation stages for all four entry points. 
Compute entry points lack "local root signatures" in D3D12, and likewise Vulkan has no concept of "local" vs "global" compute pipeline layouts, so `localUniform1` is "pushed" to the stack of reserved global uniform parameters for use in `computeMain1`. 
Leaving that entry point scope "pops" that global uniform parameter such that `localUniform2` can reuse the same binding location for `computeMain2`.
However, local uniforms for ray tracing shaders map to the corresponding "local" hit records in the shader binding table, and so no "push" or "pop" to the global root signature / pipeline layouts occurs for these parameters. 

Auto-Generated Constructors
----------

### Auto-Generated Constructors - Struct

Slang has the following rules:
1. Auto-generate a `__init()` if not already defined.

   Assume:
   ```csharp
   struct DontGenerateCtor
   {
       int a;
       int b = 5;

       // Since the user has explicitly defined a constructor
       // here, Slang will not synthesize a conflicting 
       // constructor.
       __init()
       {
           // b = 5;
           a = 5;
           b = 6;
       }
   };

   struct GenerateCtor
   {
       int a;
       int b = 5;
   
       // Slang will automatically generate an implicit constructor:
       // __init()
       // {
       //     b = 5;
       // }
   };
   ```

2. If all members have equal visibility, auto-generate a 'member-wise constructor' if not conflicting with a user defined constructor.
   ```csharp
   struct GenerateCtorInner
   {
       int a;

       // Slang will automatically generate an implicit
       // __init(int in_a)
       // {
       //     a = in_a;
       // }
   };
   struct GenerateCtor : GenerateCtorInner
   {
       int b;
       int c = 5;

       // Slang will automatically generate an implicit
       // __init(int in_a, int in_b, int in_c)
       // {
       //     c = 5;
       //
       //     this = GenerateCtorInner(in_a);
       //
       //     b = in_b;
       //     c = in_c;
       // }
   };
   ```

3. If not all members have equal visibility, auto-generate a 'member-wise constructor' based on member visibility if not conflicting with a user defined constructor. 

   We generate 3 different visibilities of 'member-wise constructor's in order:
      1. `public` 'member-wise constructor'
         - Contains members of visibility: `public`
         - Do not generate if `internal` or `private` member lacks an init expression
      2. `internal` 'member-wise constructor'
         - Contains members of visibility: `internal`, `public`
         - Do not generate if `private` member lacks an init expression
      3. `private` 'member-wise constructor'
         - Contains members of visibility: `private`, `internal`, `public`

   ```csharp
   struct GenerateCtorInner1
   {
       internal int a = 0;
    
       // Slang will automatically generate an implicit
       // internal __init(int in_a)
       // {
       //     a = 0;
       //
       //     a = in_a;
       // }
   };
   struct GenerateCtor1 : GenerateCtorInner1
   {
       internal int b = 0;
       public int c;

       // Slang will automatically generate an implicit
       // internal __init(int in_a, int in_b, int in_c)
       // {
       //     b = 0;
       //
       //     this = GenerateCtorInner1(in_a);
       //
       //     b = in_b;
       //     c = in_c;
       // }
       //
       // public __init(int in_c)
       // {
       //     b = 0;
       //
       //     this = GenerateCtorInner1();
       //
       //     c = in_c;
       // }
   };

   struct GenerateCtorInner2
   {
       internal int a;
       // Slang will automatically generate an implicit
       // internal __init(int in_a)
       // {
       //     a = in_a;
       // }
   };
   struct GenerateCtor2 : GenerateCtorInner2
   {
       internal int b;
       public int c;

       /// Note: `internal b` is missing init expression,
       // Do not generate a `public` 'member-wise' constructor.

       // Slang will automatically generate an implicit
       // internal __init(int in_a, int in_b, int in_c)
       // {
       //     this = GenerateCtorInner2(in_a);
       //
       //     b = in_b;
       //     c = in_c;
       // }
   };
   ```

Initializer Lists
----------
Initializer Lists are an expression of the form `{...}`.

```csharp
int myFunc()
{
    int a = {}; // Initializer List
}
```

### Initializer Lists - Scalar

```csharp
// Equivalent to `int a = 1`
int a = {1};
```

### Initializer Lists - Vectors

```csharp
// Equivalent to `float3 a = float3(1,2,3)`
float3 a = {1, 2, 3};
```

### Initializer Lists - Arrays/Matrices

#### Array Of Scalars

```csharp
// Equivalent to `int[2] a; a[0] = 1; a[1] = 2;`
int a[2] = {1, 2}
```

#### Array Of Aggregates

```csharp
// Equivalent to `float3 a[2]; a[0] = {1,2,3}; b[1] = {4,5,6};`
float3 a[2] = { {1,2,3}, {4,5,6} };
```

#### Flattened Array Initializer

```csharp
// Equivalent to `float3 a[2] = { {1,2,3}, {4,5,6} };`
float3 a[3] = {1,2,3, 4,5,6}; 
```

### Initializer Lists - Struct

In most scenarios, using an initializer list to create a struct typed value is equivalent to calling the struct's constructor using the elements in the initializer list as arguments for the constructor, for example:
```csharp
struct GenerateCtorInner1
{
    internal int a = 0;
    
    // Slang will automatically generate an implicit
    // internal __init(int in_a)
    // {
    //     a = 0;
    //
    //     a = in_a;
    // }

    static GenerateCtorInner1 callGenerateCtorInner1()
    {
        // Calls `GenerateCtorInner1::__init(1);`
        return {1};
    }
};
struct GenerateCtor1 : GenerateCtorInner1
{
    internal int b = 0;
    public int c;

    // Slang will automatically generate an implicit
    // internal __init(int in_a, int in_b, int in_c)
    // {
    //     this = GenerateCtorInner1(in_a);
    //
    //     b = 0;
    //
    //     b = in_b;
    //     c = in_c;
    // }
    //
    // public __init(int in_c)
    // {
    //     this = GenerateCtorInner1();
    //
    //     b = 0;
    //
    //     c = in_c;
    // }
    static GenerateCtorInner1 callInternalGenerateCtor()
    {
        // Calls `GenerateCtor1::__init(1, 2, 3);`
        return {1, 2, 3};
    }
    static GenerateCtorInner1 callPublicGenerateCtor()
    {
        // Calls `GenerateCtor1::__init(1);`
        return {1}; 
    }
};

...

// Calls `{ GenerateCtor1::__init(3), GenerateCtor1::__init(2) }`
GenerateCtor1 val[2] = { { 3 }, { 2 } };
```

In addition, Slang also provides compatibility support for C-style initializer lists with `struct`s. C-style initializer lists can use [Partial Initializer List's](#Partial-Initializer-List's) and [Flattened Array Initializer With Struct's](#Flattened-Array-Initializer-With-Struct)

A struct is considered a C-style struct if:
1. User never defines a custom constructor with **more than** 0 parameters
2. All member variables in a `struct` have the same visibility (`public` or `internal` or `private`).

#### Partial Initializer List's

```csharp
struct Foo
{
    int a;
    int b;
    int c;
};

...

// Equivalent to `Foo val; val.a = 1; val.b = 0; val.c = 0;`
Foo val = {1}; 

// Equivalent to `Foo val; val.a = 2; val.b = 3; val.c = 0;`
Foo val = {2, 3};
```

#### Flattened Array Initializer With Struct's

```csharp
struct Foo
{
    int a;
    int b;
    int c;
};

...

// Equivalent to `Foo val[2] = { {0,1,2}, {3,4,5} };`
Foo val[2] = {0,1,2, 3,4,5};
```


### Initializer Lists - Default Initializer

`{}` will default initialize a value:

#### Non-Struct Type

Value will zero-initialize
```csharp
// Equivalent to `int val1 = 0;`
int val1 = {};

// Equivalent to `float3 val2 = float3(0);`
float3 val2 = {};
```

#### Struct Type

1. Attempt to call default constructor (`__init()`) of a `struct`

   ```csharp
   struct Foo
   {
       int a;
       int b;
       __init()
       {
           a = 5;
           b = 5;
       }
   };

   ...

   // Equivalent to `Foo val = Foo();`
   Foo val = {};
   ```

2. As a fallback, zero-initialize the struct

   ```csharp
   struct Foo
   {
       int a;
       int b;
   };

   ...

   // Equivalent to `Foo val; val.a = 0; val.b = 0;` 
   Foo val = {};
   ```

### Initializer Lists - Other features

Slang allows calling a default-initializer inside a default-constructor.

```c#
__init()
{
    this = {}; //zero-initialize `this`
}
```
