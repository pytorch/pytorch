---
layout: user-guide
---

Interoperation with Target-Specific Code
===========

Slang provides low-level interoperation mechanisms to allow developers to use target-specific features or invoke code written in the target language from Slang code. These mechanisms are:
- `__intrinsic_asm` construct to map a function invocation to specific textual target code.
- `__require_prelude` construct to inject arbitrary text to the generated textual target code. 
- `__target_switch` construct to use different implementations for different targets.
- `spirv_asm` construct to define inline SPIRV assembly blocks.

> #### Note
> The language mechanisms described in this chapter are considered internal compiler features.
> The compiler does not provide comprehensive checks around their uses. These mechanisms are also subject
> to breaking changes in future releases.

## Defining Intrinsic Functions for Textual Targets

When using Slang to generate code for a textual target, e.g. HLSL, GLSL, CUDA or C++, you can use `__intrinsic_asm` to define what code to generate for an invocation to an intrinsic function. For example, the following Slang code defines an intrinsic function `myPrint`, that when called, will produce a call to `printf` in the target code:
```cpp
void myPrint(float v)
{
    __intrinsic_asm R"(printf("v is %f", $0))";
}

void test()
{
    myPrint(1.0f);
}
```
Compiling the above code to CUDA or C++ will yield the following output:

```cpp
// ...
void test_0()
{
    printf("v is %f", 1.0f);
}
```

The `__intrinsic_asm` statement in `myPrint` serves as the definition for the function. When a function body contains `__intrinsic_asm`, the function is treated by the compiler as an intrinsic and it must not contain other ordinary statements. Calls to an intrinsic function will be translated using the definition string of the intrinsic. In this example, the intrinsic is defined by the string literal `R"(printf("v is %f", $0))"`, which is used to translate the call from `test()`. The `"$0"` in the literal is replaced with the first argument. Besides `"$<index>"`, you may also use the following macros in an intrinsic definition:

| Macro     |  Expands to |
|-----------|-------------|
| `$<index>`  |  Argument `<index>`, starting from 0 |
| `$T<index>` |  Type of argument `<index>` |
| `$TR`       |  The return type. |
| `$N<index>` |  The element count of argument `<index>`, if the argument is a vector. |
| `$S<index>` |  The scalar type of argument `<index>`, if the argument is a matrix or vector. |
| `$*<index>` |  Emit all arguments starting from `<index>` as comma separated list |

## Defining Intrinsic Types

You can use `__target_intrinsic` modifier on a `struct` type to cause the type being emitted as a specific string for a given target. For example:
```
__target_intrinsic(cpp, "std::string")
struct CppString
{
    uint size()
    {
        __intrinsic_asm "static_cast<uint32_t>(($0).size())";
    }
}
```
When compiling the above code to C++, the `CppString` struct will not be emitted as a C++ struct. Instead, all uses of `CppString` will be emitted as `std::string`.

## Injecting Preludes

If you have code written in the target language that you want to include in the generated code, you can use `__requirePrelude`.
For example:
```cpp
int getMyEnvVariable()
{
    __requirePrelude(R"(#include <stdlib.h>)");
    __requirePrelude(R"(#include <string>)");
    __requirePrelude(R"(
            int getEnvVarImpl()
            {
                char* var = getenv("MY_ENVIRONMENT_VAR");
                return std::stoi(var);
            }
        )");
    __intrinsic_asm "getEnvVarImpl()";
}
void test()
{
    if (getMyEnvVariable() == 0)
        return;
}
```
In this code, `getMyEnvVariable()` is defined as an intrinsic Slang function that will translate to a call to `getEnvVarImpl()` in the target code. The first two `__requirePrelude` calls causes include directives being emitted in the resulting code, and the third `__requirePrelude` call causes a definition of `getEnvVarImpl()`, written in C++, being emitted before other Slang functions are emitted. The above code will translate to the following output:
```cpp
// ...
#include <stdlib.h>
#include <string>
int getEnvVarImpl()
{
    char* var = getenv("MY_ENVIRONMENT_VAR");
    return std::stoi(var);
}
void test_0()
{
    if (getEnvVarImpl() == 0)
        return;
}
```

The strings in `__requirePrelude` are deduplicated: the same prelude string will only be emitted once no matter how many times an intrinsic function is invoked. Therefore, it is good practice to put `#include` lines as separate `__requirePrelude` statements to prevent duplicate `#include`s being generated in the output code.

## Managing Cross-Platform Code
If you are defining an intrinsic function that maps to multiple targets in different ways, you can use `__target_switch` construct to manage the target-specific definitions. For example, here is a snippet from the Slang core module that defines `getRealtimeClock`:
```hlsl
[__requiresNVAPI]
__glsl_extension(GL_EXT_shader_realtime_clock)
uint2 getRealtimeClock()
{
    __target_switch
    {
    case hlsl:
        __intrinsic_asm "uint2(NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO), NvGetSpecial( NV_SPECIALOP_GLOBAL_TIMER_HI))";
    case glsl:
        __intrinsic_asm "clockRealtime2x32EXT()";
    case spirv:
        return spirv_asm
        {
            OpCapability ShaderClockKHR;
            OpExtension "SPV_KHR_shader_clock";
            result : $$uint2 = OpReadClockKHR Device
        };
    default:
        return uint2(0, 0);
    }
}
```
This definition causes `getRealtimeClock()` to translate to a call to NVAPI when targeting HLSL, to `clockRealtime2x32EXT()` when targeting
GLSL, and to the `OpReadClockKHR` instruction when compiling directly to SPIRV through the inline SPIRV assembly block. The `default` case is
used for target not specified in the `__target_switch` statement.

Currently, the following target names are supported in a `case` statement: `cpp`, `cuda`, `glsl`, `hlsl`, and `spirv`.

## Inline SPIRV Assembly

When targeting SPIRV, Slang allows you to directly write a SPIRV assembly block and use it as part of an expression. For example:
```cpp
int test()
{
    int localVar = 5;
    return 1 + spirv_asm {
            %temp: $$int = OpIMul $localVar $(2);
            result: $$int = OpIAdd %temp %temp
        };
    // returns 21
}
```
A SPIRV assembly block contains one or more SPIRV instructions, separated by semicolons. Each SPIRV instruction has the form:
```
%identifier : <type> = <opcode> <operand> ... ;
```
where `<opcode>` defines a value named `identifier` of `<type>`, or simply:
```
<opcode> <operand> ... ;
```
When `<opcode>` does not define a return value.

When used as part of an expression, the Slang type of the `spirv_asm` construct is defined by the last instruction, which must be in the form of:
```
result: <type> = ...
```

You can use the `$` prefix to begin an anti-quote of a Slang expression inside a `spirv_asm` block. This is commonly used to refer to a Slang variable, such as `localVar` in the example, as an operand. Additionally, the `$$` prefix is used to reference a Slang type, such as the `$$uint` references in the example. 

You can also use the `&` prefix to refer to an l-value as a pointer-typed value in SPIRV, for example:
```cpp
float modf(float x, out float ip)
{
    return spirv_asm
    {
        result:$$float = OpExtInst glsl450 Modf $x &ip
    };
}
```

Opcodes such as `OpCapability`, `OpExtension` and type definitions are allowed inside a `spirv_asm` block. These instructions will be deduplicated and inserted into the correct sections defined by the SPIRV specification, for example:
```cpp
uint4 WaveMatch(T value)
{
    return spirv_asm
    {
        OpCapability GroupNonUniformPartitionedNV;
        OpExtension "SPV_NV_shader_subgroup_partitioned";
        OpGroupNonUniformPartitionNV $$uint4 result $value
    };
}
```

You may use SPIRV enum values directly as operands, for example:
```cpp
void memoryBarrierImage()
{
    spirv_asm
    {
        OpMemoryBarrier Device AcquireRelease|ImageMemory
    };
}
```

To access SPIRV builtin variables, you can use the `builtin(VarName:type)` syntax as an operand:
```cpp
uint InstanceIndex()
{
    return spirv_asm {
        result:$$uint = OpLoad builtin(InstanceId:uint);
    };
}
```

