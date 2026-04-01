Parameter Layout Rules
======================

An important goal of the Slang project is that the rules for how shader parameters get assigned to `register`s/`binding`s is completely deterministic, so that users can rely on the compiler's behavior.
This document will attempt to explain the rules that Slang employs at a high level.
Eventually it might evolve into a formal specification of the expected behavior.

Guarantees
----------

The whole point of having a deterministic layout approach is the guarantees that it gives to users, so we will start by explicitly stating the guarantees that users can rely upon:

* A single top-level shader parameter will always occupy a contiguous range of bindings/registers for each resource type it consumes (e.g., a contiguous range of `t` registers, a contiguous range of bytes in a `cbuffer`, etc.).

* The amount of resources a parameter consumes depends only on its type, and top-level context in which it appears (e.g., is it in a `cbuffer`? an entry-point varying parameter? etc.).

* A shader parameter that is declared the same way in two different programs will get the same *amount* of resources (registers/bytes) allocated for it in both programs, but it might get a different starting offset/register.

* Changing the bodies of functions in shader code cannot change the layout of shader parameters. In particular, just because a shader parameter is "dead" does not mean it gets eliminated.

* If the user doesn't use explicit `register`/`layout` modifiers to bind parameters, then each module will get a contiguous range of bindings, and the overall program will always use a contiguous range starting from zero for each resource type.

Overview of the Layout Algorithm
--------------------------------

Layout is applied to a Slang *compile request* which comprises one or more *translation units* of user code, and zero or more `import`ed modules.
The compile request also specifies zero or more *entry points* to be compiled, where each entry point identifies a function and a profile to use.

Layout is always done with respect to a chosen *target*, and different targets might compute the resource usage for types differently, or apply different alignment.
Within a single target there may also be different layout rules (e.g., the difference between GLSL `std140` and `std430`).

Layout proceeds in four main phases:

1. Establish a global ordering on shader parameters
2. Compute the resource requirements of each shader parameter
3. Process shader parameters with fixed binding modifiers
4. Allocate bindings to parameter without fixed binding modifiers

Ordering (and Collapsing) Shader Parameters
-------------------------------------------

Shader parameters from the user's code always precede shader parameters from imported modules.
The order of parameters in the user's code is derived by "walking" through the code as follows:

* Walk through each translation unit in the order they were added via API (or the order they were listed on the command line)

* Walk through each source file of a translation unit in the order they were added/listed

* Walk through global-scope shader parameter declarations (global variables, `cbuffer`s, etc.) in the order they are listed in the (preprocessed) file.

* After all global parameters for a translation unit have been walked, walk through any entry points in the translation unit.

* When walking through an entry point, walk through all of its function parameters (both uniforms and varyings) in order, and then walk the function result as a varying output parameter.

When dealing with global-scope parameters in the user's code, it is possible for the "same" parameter to appear in multiple translation units.
Any two global shader parameters in user code with the same name are assumed to represent the same parameter, and will only be included in the global order at the first location where they are seen.
It is an error for the different declarations to have a mismatch in type, or conflicting explicit bindings.

Parameters from `import`ed modules are enumerated after the user code, using the order in which modules were first `import`ed.
The order of parameters within each module is the same as when the module was compiled, which matches the ordering given above.

Computing Resource Requirements
-------------------------------

Each shader parameter computes its resource requirements based on its type, and how it is declared.

* Global-scope parameters, entry point `uniform` parameters, and `cbuffer` declarations all use the "default" layout rules

* Entry point non-`uniform` parameters use "varying" layout rules, either input or output

* A few other special case rules exist (e.g., for laying out the elements of a `StructuredBuffer`), but most users will not need to worry about these

Note that the "default" rules are different for D3D and GL/Vulkan targets, because they have slightly different packing behavior.

### Plain Old Data

Under the default rules simple scalar types (`bool`, `int`, `float`, etc.) are laid out as "uniform" data (that is, bytes of ordinary memory).
In most cases, the size matches the expected data type size (although be aware that most targets treat `bool` as a synonym for `int`) and the alignment is the same as the size.

### Vectors

Vectors are laid out as N sequential scalars.
Under HLSL rules, a vector has the same alignment as its scalar type.
Under GLSL `std140` rules, a vector has an alignment that is its size rounded up to the next power of two (so a `float3` has `float4` alignment).

### Opaque Types

"Opaque" types include resource/sampler types like `Texture2D` and `SamplerState`.
These consume a single "slot" of the appropriate category for the chosen API.

Note that when compiling for D3D, a `Texture2D` and a `SamplerState` will consume different resources (`t` and `s` registers, respectively), but when compiling for Vulkan, they both consume the same resource ("descriptor table slot").

Opaque types currently all have an alignment of one.

### Structures

A structure is laid out by initializing a counter for each resource type, and then processing fields sequential (in declaration order):

* Compute resource usage for the field's type

* Adjust counters based on the alignment of the field for each resource type where it has non-zero usage

* Assign an offset to the field for each resource type where it has non-zero usage

* Add the resource usage of the field to the counters


An important wrinkle is that when doing layout for HLSL, we must ensure that if a field with uniform data that is smaller than 16 bytes would straddle a 16-byte boundary, we advance to the next 16-byte aligned offset.

The overall alignment of a `struct` is the maximum alignment of its fields or the default alignment (if it is larger).
The default alignment is 16 for both D3D and Vulkan targets.

The final resource usage of a `struct` is rounded up to a multiple of the alignment for each resource type. Note that we allow a `struct` to consume zero bytes of uniform storage.

It is important to note that a `struct` type can use resources of many different kinds, so in general we cannot talk about the "size" of a type, but only its size for a particular kind of resource (uniform bytes, texture registers, etc.).

### Sized Arrays

For uniform data, the size of the element type is rounded up to the target-specific minimum (e.g., 16 for D3D and Vulkan constant buffers) to arrive at the *stride* of the array. The total size of the array is then the stride times the element count.

For opaque resource types, the D3D case simply takes the stride to be the number of registers consumed by each element, and multiplies this by the element count.

For Vulkan, an array of resources uses only a single `binding`, so that the stride is always zero for these resource kinds, and the resource usage of an array is the same as its element type.

### Unsized Arrays

The uniform part of an unsized array has the same stride as for the sized case, but an effectively infinite size.

For register/binding resource usage, a Vulkan unsized array is just like a sized one, while a D3D array will consume a full register *space* instead of individual registers.

### Constant Buffers

To determine the resource usage of a constant buffer (either a `cbuffer { ... }` declaration or a `ConstantBuffer<T>`) we look at the resource usage of its element type.

If the element uses any uniform data, the constant buffer will use at least one constant-buffer register (or whatever the target-specific resource is).
If the element uses any non-uniform data, that usage will be added to that of the constant buffer.

### Parameter Blocks

A parameter block is similar to a constant buffer.
If the element type uses any uniform data, we compute resource usage for a constant buffer.
We then add in any non-uniform resource usage for the element types.

If the target requires use of register spaces (e.g., for Vulkan), then a parameter block uses a single register space; otherwise it exposes the resource usage of its element type directly.

Processing Explicit Binding Modifiers
-------------------------------------

If the user put an explicit binding modifier on a parameter, and that modifier applies to the current target, then we use it and "reserve" space in the overall binding range.

Traditional HLSL `register` modifiers only apply for D3D targets.
Slang currently allows GLSL-style `layout(binding =...)` modifiers to be attached to shader parameters, and will use those modifiers for GL/Vulkan targets.

If two parameters reserve overlapping ranges, we currently issue an error.
This may be downgraded to a warning for targets that support overlapping ranges.

Allocating Bindings to Parameters
---------------------------------

Once ranges have been reserved for parameters with explicit bindings, the compiler goes through all parameters again, in the global order and assigns them bindings based on their resource requirements.

For each resource type used by a parameter, it is allocated the first contiguous range of resources of that type that have not been reserved.

Splitting of Arrays
-------------------

In order to support `struct` types that mix uniform and non-uniform data, the Slang compiler always "splits" these types.
For example, given:

```hlsl
struct LightInfo { float3 pos; Texture2D shadowMap; };

LightInfo gLight;
```

Slang will generate code like:

```hlsl
float3 gLight_pos;
Texture2D gLight_shadowMap;
```

In a simple case like the above, this doesn't affect layout at all, but once arrays get involved, the layout can be more complicated. Consider this case:

```hlsl
struct Pair { Texture2D a; Texture2D b; };
Pair gPairs[8];
```

The output from the splitting step is equivalent to:

```hlsl
Texture2D gPairs_a[8];
Texture2D gPairs_b[8];
```

While this transformation is critical for having a type layout algorithm that applies across all APIs (and also it is pretty much required to work around various bugs in downstream compilers), it has the important down-side that the value `gPairs[0]` does not occupy a contiguous range of registers (although the top-level shader parameter `gPairs` *does*).

The Slang reflection API will correctly report the information about this situation:

* The "stride" of the `gPairs` array will be reported as one, because `gPairs[n+1].a` is always one register after `gPairs[n].a`.

* The offset of the `gPairs.b` field will be reported as 8, because `gPairs[0].b` will be 8 registers after the starting register for `gPairs`.

The Slang API tries to provide the best information it can in this case, but it is still important for users who mix arrays and complex `struct` types to know how the compiler will lay them out.

Generics
--------

Generic type parameters complicate these layout rules.
For example, we cannot compute the exact resource requirements for a `vector<T,3>` without knowing what the type `T` is.

When computing layouts for fully specialized types or programs, no special considerations are needed: the rules as described in this document still apply.
One important consequence to understand is that given a type like:

```hlsl
struct MyStuff<T>
{
	int a;
	T b;
	int c;
}
```

the offset computed for the `c` field depends on the concrete type that gets plugged in for `T`.
We think this is the least surprising behavior for programmers who might be familiar with things like C++ template specialization.

In cases where confusion about a field like `c` getting different offsets in different specializations is a concern, users are encouraged to declare types so that all non-generic-dependent fields come before generic-dependent ones.
