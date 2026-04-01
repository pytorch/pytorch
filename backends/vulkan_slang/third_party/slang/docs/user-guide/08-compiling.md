---
layout: user-guide
permalink: /user-guide/compiling
---

Compiling Code with Slang
=========================

This chapter presents the ways that the Slang system supports compiling and composing shader code.
We will start with a discussion of the mental model that Slang uses for compilation.
Next we will cover the command-line Slang compiler, `slangc`, and how to use it to perform offline compilation.
Finally we will discuss the Slang compilation API, which can be used to integrate Slang compilation into an application at runtime, or to build custom tools that implement application-specific compilation policy.

## Concepts

For simple scenarios it may be enough to think of a shader compiler as a box where source code goes in and compiled kernels come out.
Most real-time graphics applications end up needing more control over shader compilation, and/or more information about the results of compilation.
In order to make use of the services provided by the Slang compilation system, it is useful to start with a clear model of the concepts that are involved in compilation.

### Source Units

At the finest granularity, code is fed to the compiler in _source units_ which are most often stored as files on disk or strings of text in memory.
The compilation model largely does not care whether source units have been authored by human programmers or automatically assembled by other tools.

If multiple source units are specified as part of the same compile, they will be preprocessed and parsed independently.
However, a source unit might contain `#include` directives, so that the preprocessed text of that source unit includes the content of other files.
Note that the `#include`d files do not become additional source units; they are just part of the text of a source unit that was fed to the compiler.

### Translation Units and Modules

Source units (such as files) are grouped into _translation units_, and each translation unit will produce a single _module_ when compiled.

While the source units are all preprocessed and parsed independently, semantic checking is applied to a translation unit as a whole.
One source file in a translation unit may freely refer to declarations in another source file from the same translation unit without any need for forward declarations. For example:

```hlsl
// A.slang

float getFactor() { return 10.0; }
```

```hlsl
// B.slang

float scaleValue(float value)
{
    return value * getFactor();
}
```

In this example, the `scaleValue()` function in `B.slang` can freely refer to the `getFactor()` function in `A.slang` because they are part of the same translation unit.

It is allowed, and indeed common, for a translation unit to contain only a single source unit.
For example, when adapting an existing codebase with many `.hlsl` files, it is appropriate to compile each `.hlsl` file as its own translation unit.
A modernized codebase that uses modular `include` feature as documented in [Modules and Access Control](modules) might decide to compile multiple `.slang` files in a single directory as a single translation unit.

The result of compiling a translation unit is a module in Slang's internal intermediate representation (IR). The compiled module can then be serialized to a `.slang-module` binary file. The binary file can then be loaded via the
`ISession::loadModuleFromIRBlob` function or `import`ed in slang code the same way as modules written in `.slang` files.

### Entry Points

A translation unit / module may contain zero or more entry points.
Slang supports two models for identifying entry points when compiling.

#### Entry Point Attributes

By default, the compiler will scan a translation unit for function declarations marked with the `[shader(...)]` attribute; each such function will be identified as an entry point in the module.
Developers are encouraged to use this model because it directly documents intention and makes source code less dependent on external compiler configuration options.

#### Explicit Entry Point Options

For compatibility with existing code, the Slang compiler also supports explicit specification of entry point functions using configuration options external to shader source code.
When these options are used the compiler will *ignore* all `[shader(...)]` attributes and only use the explicitly-specified entry points instead.

### Shader Parameters

A translation unit / module may contain zero or more global shader parameters.
Similarly, each entry point may define zero or more entry-point `uniform` shader parameters.

The shader parameters of a module or entry point are significant because they describe the interface between host application code and GPU code.
It is important that both the application and generated GPU kernel code agree on how parameters are laid out in memory and/or how they are assigned to particular API-defined registers, locations, or other "slots."

### Targets

Within the Slang system a _target_ represents a particular platform and set of capabilities that output code can be generated for.
A target includes information such as:

* The _format_ that code should be generated in: SPIR-V, DXIL, etc.

* A _profile_ that specifies a general feature/capability level for the target: D3D Shader Model 5.1, GLSL version 4.60, etc.

* Optional _capabilities_ that should be assumed available on the target: for example, specific Vulkan GLSL extensions

* Options that impact code generation: floating-point strictness, level of debug information to generate, etc.

Slang supports compiling for multiple targets in the same compilation session.
When using multiple targets at a time, it is important to understand the distinction between the _front-end_ of the compiler, and the _back-end_:

* The compiler front-end comprises preprocessing, parsing, and semantic checking. The front-end runs once for each translation unit and its results are shared across all targets.

* The compiler back-end generates output code, and thus runs once per target.

> #### Note ####
> Because front-end actions, including preprocessing, only run once, across all targets, the Slang compiler does not automatically provide any target-specific preprocessor `#define`s that can be used for preprocessor conditionals.
> Applications that need target-specific `#define`s should always compile for one target at a time, and set up their per-target preprocessor state manually.

### Layout

While the front-end of the compiler determines what the shader parameters of a module or entry point are, the _layout_ for those parameters is dependent on a particular compilation target.
A `Texture2D` might consume a `t` register for Direct3D, a `binding` for Vulkan, or just plain bytes for CUDA.

The details of layout in Slang will come in a later chapter.
For the purposes of the compilation model it is important to note that the layout computed for shader parameters depends on:

* What modules and entry points are being used together; these define which parameters are relevant.

* Some well-defined ordering of those parameters; this defines which parameters should be laid out before which others.

* The rules and constraints that the target imposes on layout.

An important design choice in Slang is give the user of the compiler control over these choices.

### Composition

The user of the Slang compiler communicates the modules and entry points that will be used together, as well as their relative order, using a system for _composition_.

A _component type_ is a unit of shader code composition; both modules and entry points are examples of component types.
A _composite_ component type is formed from a list of other component types (for example, one module and two entry points) and can be used to define a unit of shader code that is meant to be used together.

Once a programmer has formed a composite of all the code they intend to use together, they can query the layout of the shader parameters in that composite, or invoke the linking step to
resolve all cross module references.

### Linking

A user-composed program may have transitive module dependencies and cross references between module boundaries. The linking step in Slang is to resolve all the cross references in the IR and produce a
new self-contained IR module that has everything needed for target code generation. The user will have an opportunity to specialize precompiled modules or provide additional compiler backend options
at the linking step.

### Kernels

Once a program is linked, the user can request generation of the _kernel_ code for an entry point.
The same entry point can be used to generate many different kernels.
First, an entry point can be compiled for different targets, resulting in different kernels in the appropriate format for each target.
Second, different compositions of shader code can result in different layouts, which leads to different kernels being required.

## Command-Line Compilation with `slangc`

The `slangc` tool, included in binary distributions of Slang, is a command-line compiler that can handle most simple compilation tasks.
`slangc` is intended to be usable as a replacement for tools like `fxc` and `dxc`, and covers most of the same use cases.

### All Available Options

See [slangc command line reference](https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md) for a complete list of compiler options supported by the `slangc` tool.


### A Simple `slangc` Example

Here we will repeat the example used in the [Getting Started](01-get-started.md) chapter.
Given the following Slang code:

```hlsl
// hello-world.slang
StructuredBuffer<float> buffer0;
StructuredBuffer<float> buffer1;
RWStructuredBuffer<float> result;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    result[index] = buffer0[index] + buffer1[index];
}
```

we can compile the `computeMain()` entry point to SPIR-V using the following command line:

```bat
slangc hello-world.slang -target spirv -o hello-world.spv
```

> #### Note ####
> Some targets require additional parameters. See [`slangc` Entry Points](#slangc-entry-points) for details. For example, to target HLSL, the equivalent command is:
>
> ```bat
> slangc hello-world.slang -target hlsl -entry computeMain -o hello-world.hlsl
> ```

### Source Files and Translation Units

The `hello-world.slang` argument here is specifying an input file.
Each input file specified on the command line will be a distinct source unit during compilation.
Slang supports multiple file-name extensions for input files, but the most common ones will be `.hlsl` for existing HLSL code, and `.slang` for files written specifically for Slang.

If multiple source files are passed to `slangc`, they will be grouped into translation units using the following rules:

* If there are any `.slang` files, then all of them will be grouped into a single translation unit

* Each `.hlsl` file will be grouped into a distinct translation unit of its own.

* Each `.slang-module` file forms its own translation unit.

### `slangc` Entry Points

When using `slangc`, you will typically want to identify which entry point(s) you intend to compile.
The `-entry computeMain` option selects an entry point to be compiled to output code in this invocation of `slangc`.

Because the `computeMain()` entry point in this example has a `[shader(...)]` attribute, the compiler is able to deduce that it should be compiled for the `compute` stage.

```bat
slangc hello-world.slang -target spirv -o hello-world.spv
```

In code that does not use `[shader(...)]` attributes, a `-entry` option should be followed by a `-stage` option to specify the stage of the entry point:

```bat
slangc hello-world.slang -entry computeMain -stage compute -target spirv -o hello-world.spv
```

> #### Note ####
> The `slangc` CLI [currently](https://github.com/shader-slang/slang/issues/5541) cannot automatically deduce `-entrypoint` and `-stage`/`-profile` options from `[shader(...)]` attributes when generating code for targets other than SPIRV, Metal, CUDA, or Optix. For targets such as HLSL, please continue to specify `-entry` and `-stage` options, even when compiling a file with the `[shader(...)]` attribute on its entry point.

### `slangc` Targets

Our example uses the option `-target spirv` to introduce a compilation target; in this case, code will be generated as SPIR-V.
The argument of a `-target` option specified the format to use for the target; common values are `dxbc`, `dxil`, and `spirv`.

Additional options for a target can be specified after the `-target` option.
For example, a `-profile` option can be used to specify a profile that should be used.
Slang provides two main kinds of profiles for use with `slangc`:

* Direct3D "Shader Model" profiles have names like `sm_5_1` and `sm_6_3`

* GLSL versions can be used as profile with names like `glsl_430` and `glsl_460`

### `slangc` Kernels

A `-o` option indicates that kernel code should be written to a file on disk.
In our example, the SPIR-V kernel code for the `computeMain()` entry point will be written to the file `hello-world.spv`.

### Working with Multiples

It is possible to use `slangc` with multiple input files, entry points, or targets.
In these cases, the ordering of arguments on the command line becomes significant.

When an option modifies or relates to another command-line argument, it implicitly applies to the most recent relevant argument.
For example:

* If there are multiple input files, then an `-entry` option applies to the preceding input file

* If there are multiple entry points, then a `-stage` option applies to the preceding `-entry` option

* If there are multiple targets, then a `-profile` option applies to the preceding `-target` option

Kernel `-o` options are the most complicated case, because they depend on both a target and entry point.
A `-o` option applies to the preceding entry point, and the compiler will try to apply it to a matching target based on its file extension.
For example, a `.spv` output file will be matched to a `-target spirv`.

The compiler makes a best effort to support complicated cases with multiple files, entry points, and targets.
Users with very complicated compilation requirements will probably be better off using multiple `slangc` invocations or migrating to the compilation API.

### Additional Options

The main other options are:

* `-D<name>` or `-D<name>=<value>` can be used to introduce preprocessor macros.

* `-I<path>` or `-I <path>` can be used to introduce a _search path_ to be used when resolving `#include` directives and `import` declarations.

* `-g` can be used to enable inclusion of debug information in output files (where possible and implemented)

* `-O<level>` can be used to control optimization levels when the Slang compiler invokes downstream code generator

See [slangc command line reference](https://github.com/shader-slang/slang/blob/master/docs/command-line-slangc-reference.md) for a complete list of compiler options supported by the `slangc` tool.

### Downstream Arguments

`slangc` may leverage a 'downstream' tool like 'dxc', 'fxc', 'glslang', or 'gcc' for some target compilations. Rather than replicate every possible downstream option, arguments can be passed directly to the downstream tool using the "-X" option in `slangc`.

The mechanism used here is based on the `-X` mechanism used in GCC, to specify arguments to the linker.

```
-Xlinker option
```

When used, `option` is not interpreted by GCC, but is passed to the linker once compilation is complete. Slang extends this idea in several ways. First there are many more 'downstream' stages available to Slang than just `linker`. These different stages are known as `SlangPassThrough` types in the API and have the following names

* `fxc` - FXC HLSL compiler
* `dxc` - DXC HLSL compiler
* `glslang` - GLSLANG GLSL compiler
* `visualstudio` - Visual Studio C/C++ compiler
* `clang` - Clang C/C++ compiler
* `gcc` - GCC C/C++ compiler
* `genericcpp` - A generic C++ compiler (can be any one of visual studio, clang or gcc depending on system and availability)
* `nvrtc` - NVRTC CUDA compiler

The Slang command line allows you to specify an argument to these downstream compilers, by using their name after the `-X`. So for example to send an option `-Gfa` through to DXC you can use 

```
-Xdxc -Gfa
```

Note that if an option is available via normal Slang command line options then these should be used. This will generally work across multiple targets, but also avoids options clashing which is undefined behavior currently. The `-X` mechanism is best used for options that are unavailable through normal Slang mechanisms. 

If you want to pass multiple options using this mechanism the `-Xdxc` needs to be in front of every options. For example 

```
-Xdxc -Gfa -Xdxc -Vd
```

Would reach `dxc` as 

```
-Gfa -Vd
```

This can get a little repetitive especially if there are many parameters, so Slang adds a mechanism to have multiple options passed by using an ellipsis `...`. The syntax is as follows

```
-Xdxc... -Gfa -Vd -X.
```

The `...` at the end indicates all the following parameters should be sent to `dxc` until it reaches the matching terminating `-X.` or the end of the command line. 

It is also worth noting that `-X...` options can be nested. This would allow a GCC downstream compilation to control linking, for example with

```
-Xgcc -Xlinker --split -X.
```

In this example gcc would see

```
-Xlinker --split
```

And the linker would see (as passed through by gcc) 

```
--split
```

Setting options for tools that aren't used in a Slang compilation has no effect. This allows for setting `-X` options specific for all downstream tools on a command line, and they are only used as part of a compilation that needs them.

NOTE! Not all tools that Slang uses downstream make command line argument parsing available. `FXC` and `GLSLANG` currently do not have any command line argument passing as part of their integration, although this could change in the future.

The `-X` mechanism is also supported by render-test tool. In this usage `slang` becomes a downstream tool. Thus you can use the `dxc` option `-Gfa` in a render-test via 

```
-Xslang... -Xdxc -Gfa -X.
```

Means that the dxc compilation in the render test (assuming dxc is invoked) will receive 

```
-Gfa
```

Some options are made available via the same mechanism for all downstream compilers. 

* Use `-I` to specify include path for downstream compilers

For example to specify an include path "somePath" to DXC you can use...

```
-Xdxc -IsomePath
```


### Convenience Features

The `slangc` compiler provides a few conveniences for command-line compilation:

* Most options can appear out of order when they are unambiguous. For example, if there is only a single translation unit a `-entry` option can appear before or after any file.

* A `-target` option can be left out if it can be inferred from the only `-o` option present. For example, `-o hello-world.spv` already implies `-target spirv`.

* If a `-o` option is left out then kernel code will be written to the standard output. This output can be piped to a file, or can be printed to a console. In the latter case, the compiler will automatically disassemble binary formats for printing.

### Precompiled Modules

You can compile a `.slang` file into a binary IR module. For example, given the following source:

```hlsl
// my_library.slang
float myLibFunc() { return 5.0; }
```

You can compile it into `my_library.slang-module` with the following slangc command line:

```bat
slangc my_library.slang -o my_library.slang-module
```

This allows you to deploy just the `my_library.slang-module` file to users of the module, and it can be consumed in the user code with the same `import` syntax:
```hlsl
import my_library;
```

### Limitations

The `slangc` tool is meant to serve the needs of many developers, including those who are currently using `fxc`, `dxc`, or similar tools.
However, some applications will benefit from deeper integration of the Slang compiler into application-specific code and workflows.
Notable features that Slang supports which cannot be accessed from `slangc` include:

* Slang can provide _reflection_ information about shader parameters and their layouts for particular targets; this information is not currently output by `slangc`.

* Slang allows applications to control the way that shader modules and entry points are composed (which in turn influences their layout); `slangc` currently implements a single default policy for how to generate a composition of shader code.

Applications that need more control over compilation are encouraged to use the C++ compilation API described in the next section.

### Examples of `slangc` usage

#### Multiple targets and multiple entrypoints

In this example, there are two shader entrypoints defined in one source file.

```hlsl
// targets.slang

struct VertexOutput
{
    nointerpolation int a : SOME_VALUE;
    float3              b : SV_Position;
};

[shader("pixel")]
float4 psMain() : SV_Target
{
    return float4(1, 0, 0, 1);
}

[shader("vertex")]
VertexOutput vsMain()
{
    VertexOutput out;
    out.a = 0;
    out.b = float4(0, 1, 0, 1);
    return out;
}
```

A single entrypoint from the preceding shader can be compiled to both SPIR-V Assembly and HLSL targets in one command:
```bat
slangc targets.slang -entry psMain -target spirv-asm -o targets.spv-asm -target hlsl -o targets.hlsl
```

The following command compiles both entrypoints to SPIR-V:

```bat
slangc targets.slang -entry vsMain -entry psMain -target spirv -o targets.spv
```

#### Creating a standalone executable example

This example compiles and runs a CPU host-callable style Slang unit.

```hlsl
// cpu.slang

class MyClass
{
    int intMember;
    __init()
    {
        intMember = 0;
    }
    int method()
    {
        printf("method\n");
        return intMember;
    }
}

export __extern_cpp int main()
{
    MyClass obj = new MyClass();
    return obj.method();
}

```

Compile the above code as standalone executable, using -I option to find dependent header files:
```bat
slangc cpu.slang -target executable -o cpu.exe -Xgenericcpp -I./include -Xgenericcpp -I./external/unordered_dense/include/
```

Execute the resulting executable:
```bat
C:\slang> cpu
method

```

#### Compiling and linking slang-modules

This example demonstrates the compilation of a slang-module, and linking to a shader which uses that module.
Two scenarios are provided, one in which the entry-point is compiled in the same `slangc` invocation that links in the dependent slang-module, and another scenario where linking is a separate invocation.

```hlsl
// lib.slang
public int foo(int a) 
{ 
    return a + 1;
}
```

```hlsl
// entry.slang
import "lib";

RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = (int)dispatchThreadID.x;
    outputBuffer[index] = foo(index);
}
```

Compile lib.slang to lib.slang-module:
```bat
slangc lib.slang -o lib.slang-module
```

Scenario 1: Compile entry.slang and link lib and entry together in one step:
```bat
slangc entry.slang -target spirv -o program.spv # Compile and link
```

Scenario 2: Compile entry.slang to entry.slang-module and then link together lib and entry in a second invocation:
```bat
slangc entry.slang -o entry.slang-module # Compile
slangc lib.slang-module entry.slang-module -target spirv -o program.spv # Link
```

#### Compiling with debug symbols

Debug symbols can be added with the "-g<debug-level>" option.

Adding '-g1' (or higher) to a SPIR-V compilation will emit extended 'DebugInfo' instructions.
```bat
slangc vertex.slang -target spirv-asm -o v.spv-asm -g0 # Omit debug symbols
slangc vertex.slang -target spirv-asm -o v.spv-asm -g1 # Add debug symbols
```


#### Compiling with additional preprocessor macros

User-defined macros can be set on the command-line with the "-D<macro>" or "-D<macro>=<value>" option.

```hlsl
// macrodefine.slang

[shader("pixel")]
float4 psMain() : SV_Target
{
#if defined(mymacro)
    return float4(1, 0, 0, 1);
#else
    return float4(0, 1, 0, 1);
#endif
}
```

* Setting a user-defined macro "mymacro"
```bat
slangc macrodefine.slang -entry psMain -target spirv-asm -o targets.spvasm -Dmymacro
```

## Using the Compilation API

The C++ API provided by Slang is meant to provide more complete control over compilation for applications that need it.
The additional level of control means that some tasks require more individual steps than they would when using a one-size-fits-all tool like `slangc`.

### "COM-lite" Components

Many parts of the Slang C++ API use interfaces that follow the design of COM (the Component Object Model).
Some key Slang interfaces are binary-compatible with existing COM interfaces.
However, the Slang API does not depend on any runtime aspects of the COM system, even on Windows; the Slang system can be seen as a "COM-lite" API.

The `ISlangUnknown` interface is equivalent to (and binary-compatible with) the standard COM `IUnknown`.
Application code is expected to correctly maintain the reference counts of `ISlangUnknown` objects returned from API calls; the `Slang::ComPtr<T>` "smart pointer" type is provided as an optional convenience for applications that want to use it.

Many Slang API calls return `SlangResult` values; this type is equivalent to (and binary-compatible with) the standard COM `HRESULT` type.
As a matter of convention, Slang API calls return a zero value (`SLANG_OK`) on success, and a negative value on errors.

> #### Note ####
> Slang API interfaces may be named with the suffix "_Experimental", indicating that the interface is not complete, may have known bugs, and may change or be removed between Slang API releases.

### Creating a Global Session

A Slang _global session_ uses the interface `slang::IGlobalSession` and it represents a connection from an application to a particular implementation of the Slang API.
A global session is created using the function `slang::createGlobalSession()`:

```c++
using namespace slang;

Slang::ComPtr<IGlobalSession> globalSession;
SlangGlobalSessionDesc desc = {};
createGlobalSession(&desc, globalSession.writeRef());
```

When a global session is created, the Slang system will load its internal representation of the _core module_ that the compiler provides to user code.
The core module can take a significant amount of time to load, so applications are advised to use a single global session if possible, rather than creating and then disposing of one for each compile.

If you want to enable GLSL compatibility mode, you need to set `SlangGlobalSessionDesc::enableGLSL` to `true` when calling `createGlobalSession()`. This will load the necessary GLSL intrinsic module
for compiling GLSL code. Without this setting, compiling GLSL code will result in an error.

> #### Note ####
> Currently, the global session type is *not* thread-safe.
> Applications that wish to compile on multiple threads will need to ensure that each concurrent thread compiles with a distinct global session.

> #### Note ####
> Currently, the global session should be freed after any objects created from it.
> See [issue 6344](https://github.com/shader-slang/slang/issues/6344).

### Creating a Session

A _session_ uses the interface `slang::ISession`, and represents a scope for compilation with a consistent set of compiler options.
In particular, all compilation with a single session will share:

* A list of enabled compilation targets (with their options)

* A list of search paths (for `#include` and `import`)

* A list of pre-defined macros

In addition, a session provides a scope for the loading and re-use of modules.
If two pieces of code compiled in a session both `import`  the same module, then that module will only be loaded and compiled once.

To create a session, use the `IGlobalSession::createSession()` method:

```c++
SessionDesc sessionDesc;
/* ... fill in `sessionDesc` ... */
Slang::ComPtr<ISession> session;
globalSession->createSession(sessionDesc, session.writeRef());
```

The definition of `SessionDesc` structure is:
```C++
struct SessionDesc
{
    /** The size of this structure, in bytes.
     */
    size_t structureSize = sizeof(SessionDesc);

    /** Code generation targets to include in the session.
    */
    TargetDesc const*   targets = nullptr;
    SlangInt            targetCount = 0;

    /** Flags to configure the session.
    */
    SessionFlags flags = kSessionFlags_None;

    /** Default layout to assume for variables with matrix types.
    */
    SlangMatrixLayoutMode defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    /** Paths to use when searching for `#include`d or `import`ed files.
    */
    char const* const*  searchPaths = nullptr;
    SlangInt            searchPathCount = 0;

    PreprocessorMacroDesc const*    preprocessorMacros = nullptr;
    SlangInt                        preprocessorMacroCount = 0;

    ISlangFileSystem* fileSystem = nullptr;

    bool enableEffectAnnotations = false;
    bool allowGLSLSyntax = false;

    /** Pointer to an array of compiler option entries, whose size is compilerOptionEntryCount.
    */
    CompilerOptionEntry* compilerOptionEntries = nullptr;

    /** Number of additional compiler option entries.
    */
    uint32_t compilerOptionEntryCount = 0;
};
```
The user can specify a set of commonly used compiler options directly in the `SessionDesc` struct, such as `searchPath` and `preprocessMacros`.
Additional compiler options can be specified via the `compilerOptionEntries` field, which is an array of `CompilerOptionEntry` that defines a key-value
pair of a compiler option setting, see the [Compiler Options](#compiler-options) section.

#### Targets

The `SessionDesc::targets` array can be used to describe the list of targets that the application wants to support in a session.
Often, this will consist of a single target.

Each target is described with a `TargetDesc` which includes options to control code generation for the target.
The most important fields of the `TargetDesc` are the `format` and `profile`; most others can be left at their default values.

The `format` field should be set to one of the values from the `SlangCompileTarget` enumeration.
For example:

```c++
TargetDesc targetDesc;
targetDesc.format = SLANG_SPIRV;
```

The `profile` field must be set with the ID of one of the profiles supported by the Slang compiler.
The exact numeric value of the different profiles is not currently stable across compiler versions, so applications should look up a chosen profile using `IGlobalSession::findProfile`.
For example:

```c++
targetDesc.profile = globalSession->findProfile("glsl_450");
```

Once the chosen `TargetDesc`s have been initialized, they can be attached to the `SessionDesc`:

```c++
sessionDesc.targets = &targetDesc;
sessionDesc.targetCount = 1;
```

#### Search Paths

The search paths on a session provide the paths where the compiler will look when trying to resolve a `#include` directive or `import` declaration.
The search paths can be set in the `SessionDesc` as an array of `const char*`:

```c++
const char* searchPaths[] = { "myapp/shaders/" };
sessionDesc.searchPaths = searchPaths;
sessionDesc.searchPathCount = 1;
```

#### Pre-Defined Macros

The pre-defined macros in a session will be visible at the start of each source unit that is compiled, including source units loaded via `import`.
Each pre-defined macro is described with a `PreprocessorMacroDesc`, which has `name` and `value` fields:

```c++
PreprocessorMacroDesc fancyFlag = { "ENABLE_FANCY_FEATURE", "1" };
sessionDesc.preprocessorMacros = &fancyFlag;
sessionDesc.preprocessorMacroCount = 1;
```

#### More Options

You can specify other compiler options for the session or for a specific target through the `compilerOptionEntries` and `compilerOptionEntryCount` fields
of the `SessionDesc` or `TargetDesc` structures. See the [Compiler Options](#compiler-options) section for more details on how to encode such an array.

### Loading a Module

The simplest way to load code into a session is with `ISession::loadModule()`:

```c++
IModule* module = session->loadModule("MyShaders");
```

Executing `loadModule("MyShaders")` in host C++ code is similar to using `import MyShaders` in Slang code.
The session will search for a matching module (usually in a file called `MyShaders.slang`) and will load and compile it (if it hasn't been done already).

Note that `loadModule()` does not provide any ways to customize the compiler configuration for that specific module.
The preprocessor environment, search paths, and targets will always be those specified for the session.

### Capturing Diagnostic Output

Compilers produce various kinds of _diagnostic_ output when compiling code.
This includes not only error messages when compilation fails, but also warnings and other helpful messages that may be produced even for successful compiles.

Many operations in Slang, such as `ISession::loadModule()` can optionally produce a _blob_ of diagnostic output.
For example:

```c++
Slang::ComPtr<IBlob> diagnostics;
Slang::ComPtr<IModule> module = session->loadModule("MyShaders", diagnostics.writeRef());
```

In this example, if any diagnostic messages were produced when loading `MyShaders`, then the `diagnostics` pointer will be set to a blob that contains the textual content of those diagnostics.

The content of a blob can be accessed with `getBufferPointer()`, and the size of the content can be accessed with `getBufferSize()`.
Diagnostic blobs produces by the Slang compiler are always null-terminated, so that they can be used with C-style string APIs:

```c++
if(diagnostics)
{
    fprintf(stderr, "%s\n", (const char*) diagnostics->getBufferPointer());
}
```

> #### Note ####
> The `slang::IBlob` interface is binary-compatible with the `ID3D10Blob` and `ID3DBlob` interfaces used by some Direct3D compilation APIs.

### Entry Points

When using `loadModule()` applications should ensure that entry points in their shader code are always marked with appropriate `[shader(...)]` attributes.
For example, if `MyShaders.slang` contained:

```hlsl
[shader("compute")]
void myComputeMain(...) { ... }
```

then the Slang system will automatically detect and validate this entry point as part of a `loadModule("MyShaders")` call.

After a module has been loaded, the application can look up entry points in that module using `IModule::findEntryPointByName()`:

```c++
Slang::ComPtr<IEntryPoint> computeEntryPoint;
module->findEntryPointByName("myComputeMain", computeEntryPoint.writeRef());
```

### Composition

An application might load any number of modules with `loadModule()`, and those modules might contain any number of entry points.
Before GPU kernel code can be generated it is first necessary to decide which pieces of GPU code will be used together.

Both `slang::IModule` and `slang::IEntryPoint` inherit from `slang::IComponentType`, because both can be used as components when composing a shader program.
A composition can be created with `ISession::createCompositeComponentType()`:

```c++
IComponentType* components[] = { module, entryPoint };
Slang::ComPtr<IComponentType> program;
session->createCompositeComponentType(components, 2, program.writeRef());
```

As discussed earlier in this chapter, the composition operation serves two important purposes.
First, it establishes which code is part of a compiled shader program and which is not.
Second, it established an ordering for the code in a program, which can be used for layout.

### Layout and Reflection

Some applications need to perform reflection on shader parameters and their layout, whether at runtime or as part of an offline compilation tool.
The Slang API allows layout to be queried on any `IComponentType` using `getLayout()`:

```c++
slang::ProgramLayout* layout = program->getLayout();
```

> #### Note ####
> In  the current Slang API, the `ProgramLayout` type is not reference-counted.
> Currently, the lifetime of a `ProgramLayout` is tied to the `IComponentType` that returned it.
> An application must ensure that it retains the given `IComponentType` for as long as it uses the `ProgramLayout`.

Note that because both `IModule` and `IEntryPoint` inherit from `IComponentType`, they can also be queried for their layouts individually.
The layout for a module comprises just its global-scope parameters.
The layout for an entry point comprises just its entry-point parameters (both `uniform` and varying).

The details of how Slang computes layout, what guarantees it makes, and how to inspect the reflection information will be discussed in a later chapter.

Because the layout computed for shader parameters may depend on the compilation target, the `getLayout()` method actually takes a `targetIndex` parameter that is the zero-based index of the target for which layout information is being queried.
This parameter defaults to zero as a convenience for the common case where applications use only a single compilation target at runtime.

See [Using the Reflection API](reflection) chapter for more details on the reflection API.

### Linking

Before generating code, you must link the program to resolve all cross-module references. This can be done by calling
`IComponentType::link` or `IComponentType::linkWithOptions` if you wish to specify additional compiler options for the program.
For example:
```c++
Slang::ComPtr<IComponentType> linkedProgram;
Slang::ComPtr<ISlangBlob> diagnosticBlob;
program->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());
```

The linking step is also used to perform link-time specialization, which is a recommended approach for shader specialization
compared to preprocessor based specialization. Please see [Link-time Specialization and Precompiled Modules](link-time-specialization) for more details.

Any diagnostic messages related to linking (for example, if an external symbol cannot be resolved) will be written to `diagnosticBlob`.

### Kernel Code

Given a linked `IComponentType`, an application can extract kernel code for one of its entry points using `IComponentType::getEntryPointCode()`:

```c++
int entryPointIndex = 0; // only one entry point
int targetIndex = 0; // only one target
Slang::ComPtr<IBlob> kernelBlob;
linkedProgram->getEntryPointCode(
    entryPointIndex,
    targetIndex,
    kernelBlob.writeRef(),
    diagnostics.writeRef());
```

Any diagnostic messages related to back-end code generation (for example, if the chosen entry point requires features not available on the chosen target) will be written to `diagnostics`.
The `kernelBlob` output is a `slang::IBlob` that can be used to access the generated code (whether binary or textual).
In many cases `kernelBlob->getBufferPointer()` can be passed directly to the appropriate graphics API to load kernel code onto a GPU.


## Multithreading

The only functions which are currently thread safe are 

```C++
SlangSession* spCreateSession(const char* deprecated);
SlangResult slang_createGlobalSession(SlangInt apiVersion, slang::IGlobalSession** outGlobalSession);
SlangResult slang_createGlobalSession2(const SlangGlobalSessionDesc* desc, slang::IGlobalSession** outGlobalSession);
SlangResult slang_createGlobalSessionWithoutCoreModule(SlangInt apiVersion, slang::IGlobalSession** outGlobalSession);
ISlangBlob* slang_getEmbeddedCoreModule();
SlangResult slang::createGlobalSession(slang::IGlobalSession** outGlobalSession);
const char* spGetBuildTagString();
```

This assumes Slang has been built with the C++ multithreaded runtime, as is the default.

All other functions and methods are not [reentrant](https://en.wikipedia.org/wiki/Reentrancy_(computing)) and can only execute on a single thread. More precisely, functions and methods can only be called on a *single* thread at *any one time*. This means for example a global session can be used across multiple threads, as long as some synchronization enforces that only one thread can be in a Slang call at any one time.

Much of the Slang API is available through [COM interfaces](https://en.wikipedia.org/wiki/Component_Object_Model). In strict COM, interfaces should be atomically reference counted. Currently *MOST* Slang API COM interfaces are *NOT* atomic reference counted. One exception is the `ISlangSharedLibrary` interface when produced from [host-callable](cpu-target.md#host-callable). It is atomically reference counted, allowing it to persist and be used beyond the original compilation and be freed on a different thread. 


## Compiler Options

Both the `SessionDesc`, `TargetDesc` structures contain fields that encodes a `CompilerOptionEntry` array for additional compiler options to apply on the session or the target. In addition,
the `IComponentType::linkWithOptions()` method allow you to specify additional compiler options when linking a program. All these places accepts the same encoding of compiler options, which is
documented in this section.

The `CompilerOptionEntry` structure is defined as follows:
```c++
struct CompilerOptionEntry
{
    CompilerOptionName name;
    CompilerOptionValue value;
};
```
Where `CompilerOptionName` is an `enum` specifying the compiler option to set, and `value` encodes the value of the option.
`CompilerOptionValue` is a structure that allows you to endcode up to two integer or string values for a compiler option:
```c++
enum class CompilerOptionValueKind
{
    Int,
    String
};

struct CompilerOptionValue
{
    CompilerOptionValueKind kind = CompilerOptionValueKind::Int;
    int32_t intValue0 = 0;
    int32_t intValue1 = 0;
    const char* stringValue0 = nullptr;
    const char* stringValue1 = nullptr;
};
```
The meaning of each integer or string value is dependent on the compiler option. The following table lists all available compiler options that can be set and
meanings of their `CompilerOptionValue` encodings.

|CompilerOptionName | Description |
|:------------------ |:----------- |
| MacroDefine        | Specifies a preprocessor macro define entry. `stringValue0` encodes macro name, `stringValue1` encodes the macro value.
| Include            | Specifies an additional search path. `stringValue0` encodes the additional path. |
| Language           | Specifies the input language. `intValue0` encodes a value defined in `SlangSourceLanguage`. |
| MatrixLayoutColumn | Use column major matrix layout as default. `intValue0` encodes a bool value for the setting. |
| MatrixLayoutRow    | Use row major matrix layout as default. `intValue0` encodes a bool value for the setting. |
| Profile            | Specifies the target profile. `intValue0` encodes the raw profile representation returned by `IGlobalSession::findProfile()`. |
| Stage              | Specifies the target entry point stage. `intValue0` encodes the stage defined in `SlangStage` enum. |
| Target             | Specifies the target format. Has same effect as setting TargetDesc::format. |
| WarningsAsErrors   | Specifies a list of warnings to be treated as errors. `stringValue0` encodes a comma separated list of warning codes or names, or can be "all" to indicate all warnings. |
| DisableWarnings    | Specifies a list of warnings to disable. `stringValue0` encodes comma separated list of warning codes or names. |
| EnableWarning      | Specifies a list of warnings to enable. `stringValue0` encodes comma separated list of warning codes or names. |
| DisableWarning     | Specify a warning to disable. `stringValue0` encodes the warning code or name. |
| ReportDownstreamTime | Turn on/off downstream compilation time report. `intValue0` encodes a bool value for the setting. |
| ReportPerfBenchmark | Turn on/off reporting of time spend in different parts of the compiler. `intValue0` encodes a bool value for the setting. |
| SkipSPIRVValidation | Specifies whether or not to skip the validation step after emitting SPIRV. `intValue0` encodes a bool value for the setting. |
| Capability | Specify an additional capability available in the compilation target. `intValue0` encodes a capability defined in the `CapabilityName` enum. |
| DefaultImageFormatUnknown | Whether or not to use `unknown` as the image format when emitting SPIRV for a texture/image resource parameter without a format specifier. `intValue0` encodes a bool value for the setting. |
| DisableDynamicDispatch | (Internal use only) Disables generation of dynamic dispatch code. `intValue0` encodes a bool value for the setting. |
| DisableSpecialization | (Internal use only) Disables specialization pass.  `intValue0` encodes a bool value for the setting. |
| FloatingPointMode | Specifies the floating point mode. `intValue0` encodes the floating mode point defined in the `SlangFloatingPointMode` enum. |
| DebugInformation | Specifies the level of debug information to include in the generated code. `intValue0` encodes an value defined in the  `SlangDebugInfoLevel` enum. |
| LineDirectiveMode | Specifies the line directive mode to use the generated textual code such as HLSL or CUDA. `intValue0` encodes an value defined in the  `SlangLineDirectiveMode` enum. |
| Optimization | Specifies the optimization level. `intValue0` encodes the value for the setting defined in the `SlangOptimizationLevel` enum. |
| Obfuscate | Specifies whether or not to turn on obfuscation. When obfuscation is on, Slang will strip variable and function names from the target code and replace them with hash values. `intValue0` encodes a bool value for the setting. |
| VulkanBindShift | Specifies the `-fvk-bind-shift` option. `intValue0` (higher 8 bits): kind, `intValue0` (lower bits): set; `intValue1`: shift. |
| VulkanBindGlobals | Specifies the `-fvk-bind-globals` option. `intValue0`: index, `intValue`: set. |
| VulkanInvertY | Specifies the `-fvk-invert-y` option. `intValue0` specifies a bool value for the setting. |
| VulkanUseDxPositionW | Specifies the `-fvk-use-dx-position-w` option. `intValue0` specifies a bool value for the setting. |
| VulkanUseEntryPointName | When set, will keep the original name of entrypoints as they are defined in the source instead of renaming them to `main`. `intValue0` specifies a bool value for the setting. |
| VulkanUseGLLayout | When set, will use std430 layout instead of D3D buffer layout for raw buffer load/stores. `intValue0` specifies a bool value for the setting. |
| VulkanEmitReflection | Specifies the `-fspv-reflect` option. When set will include additional reflection instructions in the output SPIRV. `intValue0` specifies a bool value for the setting. |
| GLSLForceScalarLayout | Specifies the `-force-glsl-scalar-layout` option. When set will use `scalar` layout for all buffers when generating SPIRV. `intValue0` specifies a bool value for the setting. |
| EnableEffectAnnotations | When set will turn on compatibility mode to parse legacy HLSL effect annotation syntax. `intValue0` specifies a bool value for the setting. |
| EmitSpirvViaGLSL | When set will emit SPIRV by emitting GLSL first and then use glslang to produce the final SPIRV code. `intValue0` specifies a bool value for the setting. |
| EmitSpirvDirectly | When set will use Slang's direct-to-SPIRV backend to generate SPIRV directly from Slang IR. `intValue0` specifies a bool value for the setting. |
| SPIRVCoreGrammarJSON | When set will use the provided SPIRV grammar file to parse SPIRV assembly blocks. `stringValue0` specifies a path to the spirv core grammar json file. |
| IncompleteLibrary | When set will not issue an error when the linked program has unresolved extern function symbols. `intValue0` specifies a bool value for the setting. |
| DownstreamArgs | Provide additional arguments to the downstream compiler. `stringValue0` encodes the downstream compiler name, `stringValue1` encodes the argument list, one argument per line. |
| DumpIntermediates | When set will dump the intermediate source output. `intValue0` specifies a bool value for the setting. |
| DumpIntermediatePrefix | The file name prefix for the intermediate source output. `stringValue0` specifies a string value for the setting. |
| DebugInformationFormat | Specifies the format of debug info. `intValue0` a value defined in the `SlangDebugInfoFormat` enum. |
| VulkanBindShiftAll | Specifies the `-fvk-bind-shift` option for all spaces. `intValue0`: kind, `intValue1`: shift. |
| GenerateWholeProgram | When set will emit target code for the entire program instead of for a specific entrypoint. `intValue0` specifies a bool value for the setting. |
| UseUpToDateBinaryModule | When set will only load precompiled modules if it is up-to-date with its source. `intValue0` specifies a bool value for the setting. |
| ValidateUniformity | When set will perform [uniformity analysis](a1-05-uniformity.md).|

## Debugging

Slang's SPIRV backend supports generating debug information using the [NonSemantic Shader DebugInfo Instructions](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc).
To enable debugging information when targeting SPIRV, specify the `-emit-spirv-directly` and the `-g2` argument when using `slangc` tool, or set `EmitSpirvDirectly` to `1` and `DebugInformation` to `SLANG_DEBUG_INFO_LEVEL_STANDARD` when using the API.
Debugging support has been tested with RenderDoc.
