---
layout: user-guide
permalink: /user-guide/capabilities
---

# Capabilities

One of the biggest challenges in maintaining cross-platform shader code is to manage the differences in hardware capabilities across different GPUs, graphics APIs, and shader stages.
Each graphics API or shader stage may expose operations that are not available on other platforms. Instead of restricting Slang's features to the lowest common denominator of different platforms,
Slang exposes operations from all target platforms to allow the user to take maximum advantage on a specific target.

A consequence of this approach is that the user is now responsible for maintaining compatibility of their code. For example, if the user writes code that uses a Vulkan extension currently not
available on D3D/HLSL, they will get an error when attempting to compile that code to D3D.

To help the user to maintain compatibility of their shader code on platforms that matter to their applications, Slang's type system can now infer and enforce capability requirements
to provide assurance that the shader code will be compatible with the specific set of platforms before compiling for that platform.

For example, `Texture2D.SampleCmp` is available on D3D and Vulkan, but not available on CUDA. If the user is intended to write cross-platform code that targets CUDA, they will
receive a type-checking error when attempting to use `SampleCmp` before the code generation stage of compilation. When using Slang's intellisense plugin, the programmer should
get a diagnostic message directly in their code editor.

As another example, `discard` is a statement that is only meaningful when used in fragment shaders. If a vertex shader contains a `discard` statement or calling a function that contains
a `discard` statement, it shall be a type-check error.

## Capability Atoms and Capability Requirements

Slang models code generation targets, shader stages, API extensions and hardware features as distinct capability atoms. For example, `GLSL_460` is a capability atom that stands for the GLSL 460 code generation target,
`compute` is an atom that represents the compute shader stage, `_sm_6_7` is an atom representing the shader model 6.7 feature set in D3D, `SPV_KHR_ray_tracing` is an atom representing the `SPV_KHR_ray_tracing` SPIR-V extension, and `spvShaderClockKHR` is an atom for the `ShaderClockKHR` SPIRV capability. For a complete list of capabilities supported by the Slang compiler, check the [capability definition file](https://github.com/shader-slang/slang/blob/master/source/slang/slang-capabilities.capdef).

A capability **requirement** can be a single capability atom, a conjunction of capability atoms, or a disjunction of conjunction of capability atoms. A function can declare its
capability requirement with the following syntax:

```csharp
[require(spvShaderClockKHR)]
[require(glsl, GL_EXT_shader_realtime_clock)]
[require(hlsl_nvapi)]
uint2 getClock() {...}
```

Each `[require]` attribute declares a conjunction of capability atoms, and all `[require]` attributes form the final requirement of the `getClock()` function as a disjunction of capabilities:
```
(spvShaderClockKHR | glsl + GL_EXT_shader_realtime_clock | hlsl_nvapi)
```

A capability can __imply__ other capabilities. Here `spvShaderClockKHR` is a capability that implies `SPV_KHR_shader_clock`, which represents the SPIRV `SPV_KHR_shader_clock` extension, and the `SPV_KHR_shader_clock` capability implies `spirv_1_0`, which stands for the spirv code generation target.

When evaluating capability requirements, Slang will expand all implications. Therefore the final capability requirement for `getClock` is:
```
  spirv_1_0 + SPV_KHR_shader_clock + spvShaderClockKHR
| glsl + _GL_EXT_shader_realtime_clock
| hlsl + hlsl_nvapi
```
Which means the function can be called from locations where the `spvShaderClockKHR` capability is available (when targeting SPIRV), or where the `GL_EXT_shader_realtime_clock` extension is available when targeting GLSL,
or where `nvapi` is available when targeting HLSL.

## Conflicting Capabilities

Certain groups of capabilities are mutually exclusive such that only one capability in the group is allowed to exist. For example, all stage capabilities are mutual exclusive: a requirement for both `fragment` and `vertex` is impossible to satisfy. Currently, capabilities that model different code generation targets (e.g. `hlsl`, `glsl`) or different shader stages (`vertex`, `fragment`, etc.) are mutually exclusive within
their corresponding group.

If two capability requirements contain different atoms that are conflicting with each other, these two requirements are considered __incompatible__.
For example, requirement `spvShaderClockKHR + fragment` and requirement `spvShaderClockKHR + vertex` are incompatible, because `fragment` conflicts with `vertex`.

## Requirements in Parent Scope

The capability requirement of a decl is always merged with the requirements declared in its parents. If the decl declares requirements for additional compilation targets, they are added
to the requirement set as a separate disjunction.
For example, given:
```csharp
[require(glsl)]
[require(hlsl)]
struct MyType
{
    [require(hlsl, hlsl_nvapi)]
    [require(spirv)]
    static void method() { ... }
}
```
`MyType.method` will have requirement `glsl | hlsl + hlsl_nvapi | spirv`.

The `[require]` attribute can also be used on module declarations, so that the requirement will
apply to all decls within the module. For example:
```csharp
[require(glsl)]
[require(hlsl)]
[require(spirv)]
module myModule;

// myFunc has requirement glsl|hlsl|spirv
public void myFunc()
{
}
```

## Inference of Capability Requirements

By default, Slang will infer the capability requirements of a function given its definition, as long as the function has `internal` or `private` visibility. For example, given:
```csharp
void myFunc()
{
    if (getClock().x % 1000 == 0)
        discard;
}
```
Slang will automatically deduce that `myFunc` has capability
```
  spirv_1_0 + SPV_KHR_shader_clock + spvShaderClockKHR + fragment
| glsl + _GL_EXT_shader_realtime_clock + fragment
| hlsl + hlsl_nvapi + fragment
```
Since `discard` statement requires capability `fragment`.

## Inference on target_switch

A `__target_switch` statement will introduce disjunctions in its inferred capability requirement. For example:
```csharp
void myFunc()
{
    __target_switch
    {
    case spirv: ...;
    case hlsl: ...;
    }
}
```
The capability requirement of `myFunc` is `(spirv | hlsl)`, meaning that the function can be called from a context where either `spirv` or `hlsl` capability
is available.

## Capability Aliases

To make it easy to specify capabilities on different platforms, Slang also defines many aliases that can be used in `[require]` attributes.
For example, Slang declares:
```
alias sm_6_6 = _sm_6_6
             | glsl_spirv_1_5 + sm_6_5
                + GL_EXT_shader_atomic_int64 + atomicfloat2
             | spirv_1_5 + sm_6_5
                + GL_EXT_shader_atomic_int64 + atomicfloat2
                + SPV_EXT_descriptor_indexing
             | cuda
             | cpp;
```
So user code can write `[require(sm_6_6)]` to mean that the function requires shader model 6.6 on D3D or equivalent set of GLSL/SPIRV extensions when targeting GLSL or SPIRV.
Note that in the above definition, `GL_EXT_shader_atomic_int64` is also an alias that is defined as:
```
alias GL_EXT_shader_atomic_int64 = _GL_EXT_shader_atomic_int64 | spvInt64Atomics;
```
Where `_GL_EXT_shader_atomic_int64` is the atom that represent the true `GL_EXT_shader_atomic_int64` GLSL extension.
The `GL_EXT_shader_atomic_int64` alias is defined as a disjunction of `_GL_EXT_shader_atomic_int64` and the `Int64Atomics` SPIRV capability so that
it can be used in both the contexts of GLSL and SPIRV target.

When aliases are used in a `[require]` attribute, the compiler will expand the alias to evaluate the capability set, and remove all incompatible conjunctions.
For example, `[require(hlsl, sm_6_6)]` will be evaluated to `(hlsl+_sm_6_6)` because all other conjunctions in `sm_6_6` are incompatible with `hlsl`.

## Validation of Capability Requirements

Slang requires all public methods and interface methods to have explicit capability requirements declarations. Omitting capability declaration on a public method means that the method does not require any
specific capability. Functions with explicit requirement declarations will be verified by the compiler to ensure that it does not use any capability beyond what is declared.

Slang recommends but does not require explicit declaration of capability requirements for entrypoints. If explicit capability requirements are declared on an entrypoint, they will be used to validate the entrypoint the same way as other public methods, providing assurance that the function will work on all intended targets. If an entrypoint does not define explicit capability requirements, Slang will infer the requirements, and only issue a compiler error when the inferred capability is incompatible with the current code generation target.
