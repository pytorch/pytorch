Core Module Intrinsics
======================

The following document aims to cover a variety of systems used to add target specific features. They are most extensively used in the slang core module.

**NOTE!** These features should *not* be considered stable! They can be used in regular slang code to add features, but they risk breaking with any Slang version change. Additionally the features implementation can be very particular to what is required for a specific feature set, so might not work as expected in all scenarios.

As these features are in flux, it is quite possible this document is behind the current features available within the Slang code base.

If you want to add support for a feature for a target to Slang, implementing it as a part of the Slang standard modules is typically a good way to progress. Depending on the extension/feature it may not be possible to add support exclusively via changes to the standard module alone. That said most support for target specific extensions and features involve at least some changes to the slang standard modules including the core module, and typically using the mechanisms described here.

## Core Module

The main place these features are used are within the slang core module. This is implemented with a set of slang files within the slang project

* core.meta.slang 
* hlsl.meta.slang
* diff.meta.slang

Looking at these files will demonstrate the features in use. 

Most of the intrinsics and attributes have names that indicate that they are not for normal use. This is typically via a `__` prefix.

The `.meta.slang` files look largely like Slang source files, but their contents can also be generated programmatically with C++ code. A section of code can drop into `C++` code if it is proceeded by `${{{{`. The C++ section is closed with a closing `}}}}`. This mechanism is typically used to generate different versions of a similar code sequence. Values from the C++ code can be accessed via the `$()`, where the contents of the brackets specifies something that can be calculated from within the C++ code.

As an example, to produce an an array with values 0 to 9 we could write...

```slang

// Slang code
${{{{
// C++ code, calling out to a C++ function getTime, the result is held in variable time
int cppTime = getTime();
}}}}

// Back to Slang code, can access the C++ variable previously defined as cppTime. Due to $().
// The code inside the $() is executed on the C++ side, so can do calculations. In practice it would be easier
// to just use call $(getTime() + 1), but this demonstrates variables are accessible.
int slangTime = $(cppTime + 1);
```

# Attributes

## [__readNone]

A `[__readNone]` indicates a function that computes its results strictly based on argument values, without reading or writing through any pointer arguments, or any other state that could be observed by a caller.

## [__NoSideEffect]

Specifies a function declaration has no observable side effects. 

## [__unsafeForceInlineEarly]

Inlines the contained code, but does so very early stage. Being earlier allows allows some kinds of inlining transformations to work, that wouldn't work with regular inlining. It also means it must be used with *care*, because it may produce unexpected results for more complex scenarios.  

## [__NonCopyableType]

Marks a type to be non-copyable, causing SSA pass to skip turning variables of the the type into SSA values.

## [__AlwaysFoldIntoUseSiteAttribute]

A call to the decorated function should always be folded into its use site.

## [KnownBuiltin("name")]

A `[KnownBuiltin("name")]` attribute allows the compiler to identify this declaration during compilation, despite obfuscation or linkage removing optimizations

# Intrinsics

<a id="target-intrinsic"></a>
## __target_intrinsic(target, expansion)

This is a widely used and somewhat complicated intrinsic. Placed on a declaration it describes how the declaration should be emitted for a target. The complexity is that `expansion` is applied via a variety of rules. `target` is a "target capability", commonly it's just the emit target for the intrinsic, so one of...

* hlsl
* glsl
* cuda - CUDA
* cpp - C++ output (used for exe, shared-library or host-callable)

* spirv - Used for slangs SPIR-V direct mechanism

A function definition can have a `target_intrinsic` *and* a body. In that case, the body will be used for targets where the `target_intrinsic` isn't defined. 

If the intrinsic can be emitted as is, the expansion need not be specified. If only the *name* needs to changed (params can be passed as is), only the name to be expanded to needs to be specified *without* `()`. In this scenario it is not necessary to specify as a string in quotes, and just the identifier name can be used.

Currently `HLSL` has a special handling in that it is *assumed* if a declaration exists that it can be emitted verbatim to HLSL.  

The target can also be a capability atom. The atoms are listed in "slang-capability-defs.h".

What is perhaps of importance here is that for some features for a specific target can have multiple ways of achieving the same effect - for example "GL_NV_ray_tracing" and "GL_EXT_ray_tracing" are two different ray tracing extensions available for Vulkan through GLSL. The `-profile` option can disambiguate which extension is actually desired, and the capability with that name on the `target_intrinsic` specifies how to implement that feature for that specific extension.

The expansion mechanism is implemented in "slang-intrinsic-expand.cpp" which will be most up to date.

The `expansion` value can be a string or an identifier. If it is an identifier, it will just be emitted as is replacing the name of the declaration the intrinsics is associated with.

Sections of the `expansion` string that are to be replaced are prefixed by the `$` sigil.

* $0-9 - Indicates the parameter at that index. For a method call $0 is `this`.
* $T0-9 - The type for the param at the index. If the type is a texture resource derived type, returns the *element* type.
* $TR - The return type
* $G0-9 - Replaced by the type/value at that index of specialization
* $S0-9 - The scalar type of the generic at the index.
* $p - Used on texturing operations. Produces the combined texture sampler arguments as needed for GLSL.
* $C - The $C intrinsic is a mechanism to change the name of an invocation depending on if there is a format conversion required between the type associated by the resource and the backing ImageFormat. Currently this is only implemented on CUDA, where there are specialized versions of the RWTexture writes that will do a format conversion.
* $E - Sometimes accesses need to be scaled. For example in CUDA the x coordinate for surface access is byte addressed. $E will return the byte size of the *backing element*.
* $c - When doing texture access in GLSL the result may need to be cast. In particular if the underlying texture is 'half' based, GLSL only accesses (read/write) as float. So we need to cast to a half type on output. When storing into a texture it is still the case the value written must be half - but we don't need to do any casting there as half is coerced to float without a problem.
* $z - If we are calling a D3D texturing operation in the form t.Foo(s, ...), where `t` is a Texture&lt;T&gt;, then this is the step where we try to properly swizzle the output of the equivalent GLSL call into the right shape.
* $N0-9 - Extract the element count from a vector argument so that we can use it in the constructed expression.
* $V0-9 - Take an argument of some scalar/vector type and pad it out to a 4-vector with the same element type (this is the inverse of `$z`).
* $a - We have an operation that needs to lower to either `atomic*` or `imageAtomic*` for GLSL, depending on whether its first operand is a subscript into an array. This `$a` is the first `a` in `atomic`, so we will replace it accordingly.
* $A - We have an operand that represents the destination of an atomic operation in GLSL, and it should be lowered based on whether it is an ordinary l-value, or an image subscript. In the image subscript case this operand will turn into multiple arguments to the `imageAtomic*` function.
* $XP - Ray tracing ray payload
* $XC - Ray tracing callable payload
* $XH - Ray tracing hit object attribute
* $P - Type-based prefix as used for CUDA and C++ targets (I8 for int8_t, F32 - float etc)

## __attributeTarget(astClassName)

For an attribute, specifies the AST class (and derived class) the attribute can be applied to.

## __builtin

Identifies the declaration is being "builtin".

## __builtin_requirement(requirementKind)

A modifier that indicates a built-in associated type requirement (e.g., `Differential`). The requirement is one of `BuiltinRequirementKind`.

The requirement value can just be specified via the `$()` mechanism. 

## __builtin_type(tag)

Specifies a builtin type - the integer value of one of the enumeration BaseType.

## __magic_type(clsName, tag)

Used before a type declaration. The clsName is the name of the class that is used to represent the type in the AST in Slang *C++* code. The tag is an optional integer value that is in addition and meaningful in the context of the class type.

##__intrinsic_type(op)

Used to specify the IR opcode associated with a type. The IR opcode is listed as something like `$(kIROp_HLSLByteAddressBufferType)`, which will expand to the integer value of the opcode (because the opcode value is an enum value that is visible from C++). It is possible to just write the opcode number, but that is generally inadvisable as the ids for ops are not stable. If a code change in Slang C++ adds or removes an opcode the number is likely to be incorrect.

As an example from the core module

```slang
__magic_type(HLSLByteAddressBufferType)
__intrinsic_type($(kIROp_HLSLByteAddressBufferType))
struct ByteAddressBuffer
{
    // ...
};
```

# General

## __generic<>

Is an alternate syntax for specifying a declaration that is generic. The more commonly used form is to list the generic parameters in `<>` after the name of the declaration.

## attribute_syntax

Attribute syntax provides a mechanism to introduce an attribute type in Slang.

Right now the basic form is:

```
attribute_syntax [name(parmName: paramType, ...)] : syntaxClass;
```

There can be 0 or more params associated with the attribute, and if so the () are not needed.

* `name` gives the name of the attribute to define.
* `paramName` is the name of param that are specified with attribute use
* `paramType` is the type of the value associated with the param 
* `syntaxClass` is the name of an AST node class that we expect this attribute to create when checked.

For example 

```
__attributeTarget(FuncDecl)
attribute_syntax [CudaDeviceExport] : CudaDeviceExportAttribute;
```

Defines an attribute `CudaDeviceExport` which can only be applied to FuncDecl or derived AST types. Once semantically checked will be turned into a `CudaDeviceExportAttribute` attribute in the AST.

With a parameter

```
__attributeTarget(InterfaceDecl)
attribute_syntax [anyValueSize(size:int)] : AnyValueSizeAttribute;
```

Defines an attribute `anyValueSize` that can be applied to `InterfaceDecl` and derived types. It takes a single parameter called `anyValueSize` of `int` type.

## Ref<T>

Allows returning or passing a value "by reference".

# GLSL/Vulkan specific

## __glsl_version(version)

Used to specify the GLSL version number that is required for the subsequent declaration. When Slang emits GLSL source, the version at the start of the file, will be the largest version seen that emitted code uses.

For example

```slang
__glsl_version(430)
```

## __glsl_extension

Specifies the GLSL extension that is required for the declaration to work. A declaration that has the intrinsic, when output to GLSL will additionally add `#extension` to the the GLSL or SPIR-V output.  

Multiple extensions can be applied to a decoration if that is applicable, if there are multiple ways of implementing that can be emitted in the same manner (see the section around [target](#target-intrinsic)) for more details.

## __spirv_version

When declaration is used for SPIR-V target will take the highest value seen to be the SPIR-V version required. For compilation through GLSLANG, the value is passed down to to GLSLANG specifying this SPIR-V is being targeted. 

Example 

```
__spirv_version(1.3)
```

## vk::spirv_instruction

Provides a way to use a limited amount of `GL_EXT_spirv_intrinsics` the extension.  

```
vk::spirv_instruction(op, set)
```

Op is the integer *value* for the op. The `set` is optional string which specifies the instruction set the op is associated with. 
For example

```
__specialized_for_target(glsl)
[[vk::spirv_instruction(1, "NonSemantic.DebugBreak")]]
void debugBreak();
``` 

# CUDA specific 

## __cuda_sm_version

When declaration is used with this intrinsic for a CUDA target, the highest shader model seen will be passed down to the downstream CUDA compile (NVRTC).

# NVAPI 

## [__requiresNVAPI]

If declaration is reached during a compilation for an applicable target (D3D11/12), will indicate that [NVAPI support](../nvapi-support.md) is required for declaration to work. 
