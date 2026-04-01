Slang CPU Target Support
========================

Slang has preliminary support for producing CPU source and binaries. 

# Features

* Can compile C/C++/Slang source to binaries (executables, shared libraries or [directly executable](#host-callable))
* Does *not* require a C/C++ be installed if [slang-llvm](#slang-llvm) is available (as distributed with slang binary distributions)
* Can compile Slang source into C++ source code
* Supports compute style shaders 

# Limitations

These limitations apply to Slang transpiling to C++. 

* Barriers are not supported (making these work would require an ABI change)
* Atomics are not currently supported
* Limited support for [out of bounds](#out-of-bounds) accesses handling
* Entry point/s cannot be named `main` (this is because downstream C++ compiler/s expecting a regular `main`)
* `float16_t` type is not currently supported

For current C++ source output, the compiler needs to support partial specialization. 

# How it works

The initial version works by using a 'downstream' C/C++ compiler. A C++ compiler does *not* in general need to be installed on a system to compile and execute code as long as [slang-llvm](#slang-llvm) is available. A [regular C/C++](#regular-cpp) compiler can also be used, allowing access to tooling, such as profiling and debuggers, as well as being able to use regular host development features such as linking, libraries, shared libraries/dlls and executables. 

The C/C++ backend can be directly accessed much like 'dxc', 'fxc' of 'glslang' can, using the pass-through mechanism with the following new backends... 

```
SLANG_PASS_THROUGH_CLANG,                   ///< Clang C/C++ compiler 
SLANG_PASS_THROUGH_VISUAL_STUDIO,           ///< Visual studio C/C++ compiler
SLANG_PASS_THROUGH_GCC,                     ///< GCC C/C++ compiler
SLANG_PASS_THROUGH_LLVM,                    ///< slang-llvm 'compiler' - includes LLVM and Clang
SLANG_PASS_THROUGH_GENERIC_C_CPP,           ///< Generic C or C++ compiler, which is decided by the source type
```

Sometimes it is not important which C/C++ compiler is used, and this can be specified via the 'Generic C/C++' option. This will aim to use the compiler that is most likely binary compatible with the compiler that was used to build the Slang binary being used. 

To make it possible for Slang to produce CPU code, in this first iteration we convert Slang code into C/C++ which can subsequently be compiled into CPU code. If source is desired instead of a binary this can be specified via the SlangCompileTarget. These can be specified on the `slangc` command line as `-target cpp`.

When using the 'pass through' mode for a CPU based target it is currently necessary to set an entry point, even though it's basically ignored. 

In the API the `SlangCompileTarget`s are 

```
SLANG_C_SOURCE             ///< The C language
SLANG_CPP_SOURCE           ///< The C++ language
SLANG_HOST_CPP_SOURCE,     ///< C++ code for `host` style 
```        
   
Using the `-target` command line option

* `C_SOURCE`: c
* `CPP_SOURCE`: cpp,c++,cxx
* `HOST_CPP_SOURCE`: host-cpp,host-c++,host-cxx

Note! Output of C source is not currently supported.

If a CPU binary is required this can be specified as a `SlangCompileTarget` of 
   
```   
SLANG_EXECUTABLE                ///< Executable (for hosting CPU/OS)
SLANG_SHADER_SHARED_LIBRARY     ///< A shared library/Dll (for hosting CPU/OS)
SLANG_SHADER_HOST_CALLABLE      ///< A CPU target that makes `compute kernel` compiled code available to be run immediately 
SLANG_HOST_HOST_CALLABLE        ///< A CPU target that makes `scalar` compiled code available to be run immediately
SLANG_OBJECT_CODE,              ///< Object code that can be used for later linking
```

Using the `-target` command line option

* `EXECUTABLE`: exe, executable
* `SHADER_SHARED_LIBRARY`: sharedlib, sharedlibrary, dll
* `SHADER_HOST_CALLABLE`: callable, host-callable
* `OBJECT_CODE`: object-conde
* `HOST_HOST_CALLABLE`: host-host-callable
    
Using `host-callable` types from the the command line, other than to test such code compile and can be loaded for host execution.

For launching a [shader like](#compile-style) Slang code on the CPU, there typically needs to be binding of values passed the entry point function. How this works is described in the [ABI section](#abi). Functions *can* be executed directly but care must be taken to [export](#visibility) them and such that there isn't an issue with [context threading](#context-threading).

If a binary target is requested, the binary contents can be returned in a ISlangBlob just like for other targets. When using a [regular C/C++ compiler](#regular-cpp) the CPU binary typically must be saved as a file and then potentially marked for execution by the OS. It may be possible to load shared libraries or dlls from memory - but doing so is a non standard feature, that requires unusual work arounds. If possible it is typically fastest and easiest to use [slang-llvm](#slang-llvm) to directly execute slang or C/C++ code.

## <a id="compile-style"/>Compilation Styles

There are currently two styles of *compilation style* supported - `host` and `shader`. 

The `shader` style implies 

* The code *can* be executed in a GPU-kernel like execution model, launched across multiple threads (as described in the [ABI](#abi)) 
* Currently no reference counting
* Only functionality from the Slang core module, built in HLSL or anything supplied by a [COM interfaces](#com-interface) is available
* Currently [slang-llvm](#slang-llvm) only supports the `shader` style

The `host` style implies 

* Execution style is akin to more regular CPU scalar code
* Typically requires linking with `slang-rt` and use of `slang-rt` types such as `Slang::String` 
* Allows use of `new` 
* Allows the use of `class` for reference counted types
* COM interfaces are reference counted

The styles as used with [host-callable](#host-callable) are indicated via the API by 

```
SLANG_SHADER_HOST_CALLABLE  ///< A CPU target that makes `compute kernel` compiled code available to be run immediately 
SLANG_HOST_HOST_CALLABLE    ///< A CPU target that makes `scalar` compiled code available to be run immediately
```

Or via the `-target` command line options

* For 'shader' `callable` `host-callable`
* For 'host' `host-host-callable`

For an example of the `host` style please look at "examples/cpu-hello-world".

## <a id="host-callable"/>Host callable

Slang supports `host-callable` compilation targets which allow for the direct execution of the compiled code on the CPU. Currently this style of execution is supported if [slang-llvm](#slang-llvm) or a [regular C/C++ compiler](#regular-cpp) are available.

There are currently two [compilation styles](#compile-style) supported. 

In order to call into `host-callable` code after compilation it's necessary to access the result via the `ISlangSharedLibrary` interface. 

Please look at the [ABI](#abi) section for more specifics around ABI usage especially for `shader` [compile styles](#compile-style).

```C++
    slang::ICompileRequest* request = ...;

    const SlangResult compileRes = request->compile();

    // Even if there were no errors that forced compilation to fail, the
    // compiler may have produced "diagnostic" output such as warnings.
    // We will go ahead and print that output here.
    //
    if(auto diagnostics = request->getDiagnosticOutput())
    {
        printf("%s", diagnostics);
    }

    // Get the 'shared library' (note that this doesn't necessarily have to be implemented as a shared library
    // it's just an interface to executable code).
    ComPtr<ISlangSharedLibrary> sharedLibrary;
    SLANG_RETURN_ON_FAIL(request->getTargetHostCallable(0, sharedLibrary.writeRef()));

    // We can now find exported functions/variables via findSymbolAddressByName

    // For a __global public __extern_cpp int myGlobal;
    {    
        auto myGlobalPtr = (int*)sharedLibrary->findSymbolAddressByName("myGlobal");
        if (myGlobalPtr)
        {
            *myGlobalPtr = 10;
        }
    }
    
    // To get a function 
    // 
    // public __extern_cpp int add(int a, int b);
    
    // Test a free function
    {
        typedef int (*AddFunc)(int a, int b); 
        auto func = (AddFunc)sharedLibrary->findFuncByName("add");

        if (func)
        {
            // Let's add!
            int c = func(10, 20):
        }
    }
```

## <a id="slang-llvm"/>slang-llvm

`slang-llvm` is a special Slang version of [LLVM](https://llvm.org/). It's current main purpose is to allow compiling C/C++ such that it is [directly available](#host-callable) for execution using the LLVM JIT feature. If `slang-llvm` is available it is the default downstream compiler for [host-callable](#host-callable). This is because it allows for faster compilation, avoids the file system, and can execute the compiled code directly. [Regular C/C++ compilers](#regular-cpp) can be used for [host-callable](#host-callable) but requires writing source files to the file system and creating/loading shared-libraries/dlls to make the feature work.  Additionally using `slang-llvm` avoids the need for a C/C++ compiler installed on a target system.   
 
`slang-llvm` contains the Clang C++ compiler, so it is possible to also compile and execute C/C++ code in the [host-callable](#host-callable) style. 

Limitations of using `slang-llvm`
 
* Can only currently be used for [shader style](#compile-style) 
  * Cannot produce object files, libraries, OS executables or binaries 
* Is *limited* because it is not possible to directly access libraries such as the C or C++ standard libraries (see [COM interface](#com-interface) for a work-around)
* It's not possible to source debug into `slang-llvm` compiled code running on the JIT (see [debugging](#debugging) for a work-around)
* Not currently possible to return as a ISlangBlob representation 

You can detect if `slang-llvm` is available via

```C++
    slang::IGlobalSession* slangSession = ...;
    const bool hasSlangLlvm = SLANG_SUCCEEDED(slangSession->checkPassThroughSupport(SLANG_PASS_THROUGH_LLVM)); 
```

## <a id="regular-cpp"/>Regular C/C++ compilers

Slang can work with regular C/C++ 'downstream' compilers. It has been tested to work with Visual Studio, Clang and G++/Gcc on Windows and Linux.

Under the covers when Slang is used to generate a binary via a C/C++ compiler, it must do so through the file system. Currently this means the source (say generated by Slang) and the binary (produced by the C/C++ compiler) must all be files. To make this work Slang uses temporary files. The reasoning for hiding this mechanism, other than simplicity, is that it allows using with [slang-llvm](#slang-llvm) without any changes. 

## <a id="visibility"/>Visibility

In a typical Slang [shader like](#compile-style) scenario, functionality is exposed via entry points. It can be convenient and desirable to be able to call Slang functions directly from application code, and not just via entry points. By default non entry point functions are *removed* if they are not reachable by the specified entry point. Additionally for non entry point functions Slang typically generates function names that differ from the original name. 

To work around these two issues the `public` and `__extern_cpp` modifiers can be used.

`public` makes the variable or function visible outside of the module even if it isn't used within the module. For the function to work it will also keep around any function or variable it accesses. 

Note! Some care is needed here around [context threading](#context-threading) - if a function or any function a function accesses requires state held in the context, the signature of the function will be altered to include the context as the first parameter.

Making a function or variable `public` does not mean that the name remains the same. To indicate that the name should not be altered use the `__extern_cpp` modifier. For example

```
// myGlobal will be visible to the application (note the __global modifier additionally means it has C++ global behavior)
__global public __extern_cpp int myGlobal;

// myFunc is available to the application
public __extern_cpp myFunc(int a) 
{
    return a * a;
}
```

## <a id="com-interface"/>COM interface support

Slang has preliminary support for [Component Object Model (COM)](https://en.wikipedia.org/wiki/Component_Object_Model) interfaces in CPU code. 

```
[COM]
interface IDoThings
{
    int doThing(int a, int b);
    int calcHash(NativeString in);
    void printMessage(NativeString nativeString);
}
```

This support provides a way for an application to provide access to functionality in the application runtime - essentially it allows Slang code to call into application code. To do this a COM interface can be created that exposes the desired functionality. The interface/s can be made available through any of the normal mechanisms - such as through a constant buffer variable. Additionally [`__global`](#actual-global) provides a way to make functions available to Slang code without the need for [context threading](#context-threading).

The example "examples/cpu-com-example" shows this at work.

## <a id="actual-global"/>Global support

The Slang language is based on the HLSL language. This heritage means that globals have slightly different meaning to typical C/C++ usage. 

```
int myGlobal;                           ///< A constant value stored in a constant buffer
static int staticMyGlobal;              ///< A global that cannot be seen by the application 
static const int staticConstMyGlobal;   ///< A fixed value
``` 

The variable `myGlobal` will be a member of a constant buffer, meaning it's value can only change via bindings and not during execution. For some uses having `myGlobal` in the constant buffer might be appropriate, for example

* It's use is reached from a [shader style](#compile-style) entry point 
* It's value is constant across the launch 

In Slang a variable can be declared as global in the C/C++ sense via the `__global` modifier. For example

```
__global int myGlobal;
```

Doing so means

* `myGlobal` will not be defined in the constant buffer
* It can be used in functions that do not have access to the [constant buffer](#context-threading)
* It can be modified in the kernel 
* Can only be used on CPU targets (currently `__global` is not supported on the GPU targets)

One disadvantage of using `__global` is in multi-threaded environments, with multiple launches on multiple CPU threads, there is only one global and will likely cause problems unless the global value is the same across all threads.

It may be useful to set a global directly via host code, without having to write a function to enable the access. This is possible by using [`public`](#visibility) and [`__extern_cpp`](#visibility) modifiers. For example 

```
__global public __extern_cpp int myGlobal;
```

The global can now be set from host code via

```C++
    slang::ICompileRequest = ...;

    // Get the 'shared library' (note that this doesn't necessarily have to be implemented as a shared library
    // it's just an interface to executable code).
    ComPtr<ISlangSharedLibrary> sharedLibrary;
    SLANG_RETURN_ON_FAIL(request->getTargetHostCallable(0, sharedLibrary.writeRef()));

    // Set myGlobal to 20
    {
        auto myGlobalPtr = (int*)sharedLibrary->findSymbolAddressByName("myGlobal");
        *myGlobalPtr = 20;
    }
```

In terms of reflection `__global` variables are not visible.

## NativeString

Slang supports a rich 'String' type when using the [host style](#compile-style), which for C++ targets is implemented as the `Slang::String` C++ type. The type is only available on CPU targets that support `slang-rt`. 

Some limited String-like support is available via `NativeString` type which for C/C++ CPU targets is equivalent to `const char*`. For GPU targets this will use the same hash mechanism as normally available. 

`NativeString` is supported by all [shader compilation styles](#compile-style) including [slang-llvm](#slang-llvm).

TODO(JS): What happens with String with shader compile style on CPU? Shouldn't it be the same as GPU (and reflected as such in reflection)?

## Debugging

It is currently not possible to step into LLVM-JIT code when using [slang-llvm](#slang-llvm). Fortunately it is possible to step into code compiled via a [regular C/C++ compiler](#regular-cpp). 

Below is a code snippet showing how to switch to a [regular C/C++ compiler](#regular-cpp) at runtime.

```C++
    SlangPassThrough findRegularCppCompiler(slang::IGlobalSession* slangSession)
    {
        // Current list of 'regular' C/C++ compilers
        const SlangPassThrough cppCompilers[] = 
        {
            SLANG_PASS_THROUGH_VISUAL_STUDIO,
            SLANG_PASS_THROUGH_GCC,
            SLANG_PASS_THROUGH_CLANG,
        };
        // Do we have a C++ compiler
        for (const auto compiler : cppCompilers)
        {
            if (SLANG_SUCCEEDED(slangSession->checkPassThroughSupport(compiler)))
            {
                return compile;
            }
        }
        return SLANG_PASS_THROUGH_NONE;
    }

    SlangResult useRegularCppCompiler(slang::IGlobalSession* session)
    {
        const auto regularCppCompiler = findRegularCppCompiler(session)
        if (regularCppCompiler != SLANG_PASS_THROUGH_NONE)
        {
            slangSession->setDownstreamCompilerForTransition(SLANG_CPP_SOURCE, SLANG_SHADER_HOST_CALLABLE, regularCppCompiler);
            slangSession->setDownstreamCompilerForTransition(SLANG_CPP_SOURCE, SLANG_HOST_HOST_CALLABLE, regularCppCompiler);
            return SLANG_OK;
        }
        return SLANG_FAIL;
    }
```        

It is generally recommended to use [slang-llvm](#slang-llvm) if that is appropriate, but to switch to using a [regular C/C++ compiler](#regular-cpp) when debugging is needed. This should be largely transparent to most code.

Executing CPU Code
==================

In typical Slang operation when code is compiled it produces either source or a binary that can then be loaded by another API such as a rendering API. With CPU code the binary produced could be saved to a file and then executed as an exe or a shared library/dll. In practice though it is common to want to be able to execute compiled code immediately. Having to save off to a file and then load again can be awkward. It is also not necessarily the case that code needs to be saved to a file to be executed. 

To handle being able call code directly, code can be compiled using the [host-callable](#host-callable).

For pass through compilation of C/C++ this mechanism allows any functions marked for export to be directly queried. Marking for export is a C/C++ compiler specific feature. Look at the definition of `SLANG_PRELUDE_EXPORT` in the [C++ prelude](#prelude).

For a complete example on how to execute CPU code using `spGetEntryPointHostCallable`/`getEntryPointHostCallable` look at code in `example/cpu-hello-world`. 

<a id="abi"/>Application Binary Interface (ABI)
===

Say we have some Slang source like the following:

```
struct Thing { int a; int b; }

Texture2D<float> tex;
SamplerState sampler;
RWStructuredBuffer<int> outputBuffer;        
ConstantBuffer<Thing> thing3;        
        
[numthreads(4, 1, 1)]
void computeMain(
    uint3 dispatchThreadID : SV_DispatchThreadID, 
    uniform Thing thing, 
    uniform Thing thing2)
{
   // ...
}
```

When compiled into a [shader compile style](#compile-style) shared library/dll/host-callable - how is it invoked? An entry point in the Slang source code produces several exported functions. The 'default' exported function has the same name as the entry point in the original source. It has the signature  

```
void computeMain(ComputeVaryingInput* varyingInput, UniformEntryPointParams* uniformParams, UniformState* uniformState);
```

NOTE! Using `main` as an entry point name should be avoided if CPU is a target because it typically causes compilation errors due it's normal C/C++ usage.

ComputeVaryingInput is defined in the prelude as 

```
struct ComputeVaryingInput
{
    uint3 startGroupID;
    uint3 endGroupID;
};
```

`ComputeVaryingInput` allows specifying a range of groupIDs to execute - all the ids in a grid from startGroup to endGroup, but not including the endGroupIDs. Most compute APIs allow specifying an x,y,z extent on 'dispatch'. This would be equivalent as having startGroupID = { 0, 0, 0} and endGroupID = { x, y, z }. The exported function allows setting a range of groupIDs such that client code could dispatch different parts of the work to different cores. This group range mechanism was chosen as the 'default' mechanism as it is most likely to achieve the best performance.

There are two other functions that consist of the entry point name postfixed with `_Thread` and `_Group`. For the entry point 'computeMain' these functions would be accessible from the shared library interface as `computeMain_Group` and `computeMain_Thread`. `_Group` has the same signature as the listed for computeMain, but it doesn't execute a range, only the single group specified by startGroupID (endGroupID is ignored). That is all of the threads within the group (as specified by `[numthreads]`) will be executed in a single call.

It may be desirable to have even finer control of how execution takes place down to the level of individual 'thread's and this can be achieved with the `_Thread` style. The signature looks as follows

```
struct ComputeThreadVaryingInput
{
    uint3 groupID;
    uint3 groupThreadID;
};

void computeMain_Thread(ComputeThreadVaryingInput* varyingInput, UniformEntryPointParams* uniformParams, UniformState* uniformState);
```

When invoking the kernel at the `thread` level it is a question of updating the groupID/groupThreadID, to specify which thread of the computation to execute. For the example above we have `[numthreads(4, 1, 1)]`. This means groupThreadID.x can vary from 0-3 and .y and .z must be 0. That groupID.x indicates which 'group of 4' to execute. So groupID.x = 1, with groupThreadID.x=0,1,2,3 runs the 4th, 5th, 6th and 7th 'thread'. Being able to invoke each thread in this way is flexible - in that any specific thread can specified and executed. It is not necessarily very efficient because there is the call overhead and a small amount of extra work that is performed inside the kernel. 

Note that the `_Thread` style signature is likely to change to support 'groupshared' variables in the near future.

In terms of performance the 'default' function is probably the most efficient for most common usages. The `_Group` style allows for slightly less loop overhead, but with many invocations this will likely be drowned out by the extra call/setup overhead. The `_Thread` style in most situations will be the slowest, with even more call overhead, and less options for the C/C++ compiler to use faster paths. 

The UniformState and UniformEntryPointParams struct typically vary by shader. UniformState holds 'normal' bindings, whereas UniformEntryPointParams hold the uniform entry point parameters. Where specific bindings or parameters are located can be determined by reflection. The structures for the example above would be something like the following... 

```
struct UniformEntryPointParams
{
    Thing thing;
    Thing thing2;
};

struct UniformState
{
    Texture2D<float > tex;
    SamplerState sampler;
    RWStructuredBuffer<int32_t> outputBuffer;
    Thing* thing3;
};   
```

Notice that of the entry point parameters `dispatchThreadID` is not part of UniformEntryPointParams and this is because it is not uniform.

`ConstantBuffer` and `ParameterBlock` will become pointers to the type they hold (as `thing3` is in the above structure).
 
`StructuredBuffer<T>`,`RWStructuredBuffer<T>` become

```
    T* data;
    size_t count;
```    

`ByteAddressBuffer`, `RWByteAddressBuffer` become 

```
    uint32_t* data;
    size_t sizeInBytes;
```    


Resource types become pointers to interfaces that implement their features. For example `Texture2D` become a pointer to a `ITexture2D` interface that has to be implemented in client side code. Similarly SamplerState and SamplerComparisonState become `ISamplerState` and `ISamplerComparisonState`.  
 
The actual definitions for the interfaces for resource types, and types are specified in 'slang-cpp-types.h' in the `prelude` directory.

## Unsized arrays

Unsized arrays can be used, which are indicated by an array with no size as in `[]`. For example 

```
    RWStructuredBuffer<int> arrayOfArrays[];
```

With normal 'sized' arrays, the elements are just stored contiguously within wherever they are defined. With an unsized array they map to `Array<T>` which is...

```
    T* data;
    size_t count;
```    

Note that there is no method in the shader source to get the `count`, even though on the CPU target it is stored and easily available. This is because of the behavior on GPU targets 

* That the count has to be stored elsewhere (unlike with CPU) 
* On some GPU targets there is no bounds checking - accessing outside the bound values can cause *undefined behavior*
* The elements may be laid out *contiguously* on GPU

In practice this means if you want to access the `count` in shader code it will need to be passed by another mechanism - such as within a constant buffer. It is possible in the future support may be added to allow direct access of `count` work across targets transparently. 

It is perhaps worth noting that the CPU allows us to have an indirection (a pointer to the unsized arrays contents) which has the potential for more flexibility than is possible on GPU targets. GPU target typically require the elements to be placed 'contiguously' from their location in their `container` - be that registers or in memory. This means on GPU targets there may be other restrictions on where unsized arrays can be placed in a structure for example, such as only at the end. If code needs to work across targets this means these restrictions will need to be followed across targets. 

## <a id="context-threading"/>Context Threading

The [shader compile style](#compile-style) brings some extra issues to bare. In the HLSL compute kernel launch model application visible variables and resource are bound. As described in the [ABI](#abi) section these bindings and additional information identifying a compute thread are passed into the launch as a context. Take for example the code snippet below

```
int myGlobal;

int myFunc(int v)
{
    return myGlobal + v;
}

int anotherFunc(int a, int b)
{
    return a + b;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    outputBuffer[dispatchThreadID.x] = myFunc(dispatchThreadID.x) + anotherFunc(1, dispatchThreadID.y);
}
```

The function `myFunc` accesses a variable `myGlobal` that is held within a constant buffer. The function cannot be meaningfully executed without access to the context, and the context is available as a parameter passed through `computeMain` entry point at launch. This means the *actual* signature of this function in output code will be something like

```
int32_t myFunc_0(KernelContext_0 * kernelContext_0)
{
    return *(&(*(&kernelContext_0->globalParams_0))->myGlobal_0) + int(1);
}
```

The context parameter has been *threaded* into this function. This *threading* will happen to any function that accesses any state that is held in the context. This behavior also happens transitively - if a function *could* call *any* another function that requires the context, the context will be threaded through to it also.

If application code assumed `myFunc` could be called with no parameters a crash would likely ensue. Note that `anotherFunc` does not have the issue because it doesn't perform an access that needs the context, and so no context threading is added.

If a global is desired in a function that wants to be called from the application, the [`__global`](#actual-global) modifier can be used.

## <a id="prelude"/>Prelude

For C++ targets, there is code to support the Slang generated source defined within the 'prelude'. The prelude is inserted text placed before the Slang generated C++ source. For the Slang command line tools as well as the test infrastructure, the prelude functionality is achieved through a `#include` in the prelude text of the `prelude/slang-cpp-prelude.h` specified with an absolute path. Doing so means other files the `slang-cpp-prelude.h` might need can be specified relatively, and include paths for the backend C/C++ compiler do not need to be modified. 

The prelude needs to define 

* 'Built in' types (vector, matrix, 'object'-like Texture, SamplerState etc) 
* Scalar intrinsic function implementations
* Compiler based definations/tweaks 

For the Slang prelude this is split into the following files...

* 'prelude/slang-cpp-prelude.h' - Header that includes all the other requirements & some compiler tweaks
* 'prelude/slang-cpp-scalar-intrinsics.h' - Scalar intrinsic implementations
* 'prelude/slang-cpp-types.h' - The 'built in types' 
* 'slang.h' - Slang header is used for majority of compiler based definitions

For a client application - as long as the requirements of the generated code are met, the prelude can be implemented by whatever mechanism is appropriate for the client. For example the implementation could be replaced with another implementation, or the prelude could contain all of the required text for compilation. Setting the prelude text can be achieved with the method on the global session...

```
/** Set the 'prelude' for generated code for a 'downstream compiler'.
@param passThrough The downstream compiler for generated code that will have the prelude applied to it. 
@param preludeText The text added pre-pended verbatim before the generated source

That for pass-through usage, prelude is not pre-pended, preludes are for code generation only. 
*/
virtual SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerPrelude(
SlangPassThrough passThrough,
const char* preludeText) = 0;
```

It may be useful to be able to include `slang-cpp-types.h` in C++ code to access the types that are used in the generated code. This introduces a problem in that the types used in the generated code might clash with types in client code. To work around this problem, you can wrap all of the types defined in the prelude with a namespace of your choosing. For example 

```
#define SLANG_PRELUDE_NAMESPACE CPPPrelude
#include "../../prelude/slang-cpp-types.h"
``` 

Would wrap all the Slang prelude types in the namespace `CPPPrelude`, such that say a `StructuredBuffer<int32_t>` could be specified in C++ source code as `CPPPrelude::StructuredBuffer<int32_t>`.

The code that sets up the prelude for the test infrastructure and command line usage can be found in ```TestToolUtil::setSessionDefaultPrelude```. Essentially this determines what the absolute path is to `slang-cpp-prelude.h` is and then just makes the prelude `#include "the absolute path"`.

The *default* prelude is set to the contents of the files for C++ held in the prelude directory and is held within the Slang shared library. It is therefore typically not necessary to distribute Slang with prelude files.

Language aspects
================

# Arrays passed by Value

Slang follows the HLSL convention that arrays are passed by value. This is in contrast the C/C++ where arrays are passed by reference. To make generated C/C++ follow this convention an array is turned into a 'FixedArray' struct type. Sinces classes by default in C/C++ are passed by reference the wrapped array is also. 

To get something similar to C/C++ operation the array can be marked `inout` to make it passed by reference. 

Limitations
===========

# <a id="out-of-bounds"/>Out of bounds access

In HLSL code if an access is made out of bounds of a StructuredBuffer, execution proceceeds. If an out of bounds read is performed, a zeroed value is returned. If an out of bounds write is performed it's effectively a noop, as the value is discarded. On the CPU target this behavior is *not* supported by default. 

For a debug CPU build an out of bounds access will assert, for a release build the behaviour is by default undefined. A limited Limited [zero index](#zero-index) out of bounds mechanism is supported, but must be enabled.

The reason for this is that such an access is difficult and/or slow to implement the identical GPU behavior on the CPU. The underlying problem is `operator[]` typically returns a reference to the contained value. If this is out of bounds - it's not clear what to return, in particular because the value may be read or written and moreover elements of the type might be written. In practice this means a global zeroed value cannot be returned. 

This could be somewhat supported if code gen worked as followed for say

```
RWStructuredBuffer<float4> values;
values[3].x = 10;
```

Produces

```
template <typename T>
struct RWStructuredBuffer
{
    T& at(size_t index, T& defValue) { return index < size ? values[index] : defValue; } 

    T* values;
    size_t size;
};

RWStructuredBuffer<float4> values;

// ...
Vector<float, 3> defValue = {};         // Zero initialize such that read access returns default values
values.at(3).x = 10;
```

Note that '[] 'would be turned into the `at` function, which takes the default value as a parameter provided by the caller. If this is then written to then only the defValue is corrupted.  Even this mechanism not be quite right, because if we write and then read again from the out of bounds reference in HLSL we may expect that 0 is returned, whereas here we get the value that was last written.

## <a id="zero-index"/>Zero index bound checking

If bounds checking is wanted in order to avoid undefined behavior and limit how memory is accessed `zero indexed` bounds checking might be appropriate. When enabled if an access is out of bounds the value at the zero index is returned. This is quite different behavior than the typical GPU behavior, but is fairly efficient and simple to implement. Importantly it means behavior is well defined and always 'in range' assuming there is an element.

To enable zero indexing bounds checking pass in the define `SLANG_ENABLE_BOUND_ZERO_INDEX` to a Slang compilation. This define is passed down to C++ and CUDA compilations, and the code in the CUDA and C++ preludes implement the feature. Note that zero indexed bounds checking will slow down accesses that are checked.

The C++ implementation of the feature can be seen by looking at the file "prelude/slang-cpp-types.h". For CUDA "prelude/slang-cuda-prelude.h".

The bounds checking macros are guarded such it is possible to replace the implementations, without directly altering the prelude.

TODO
====

# Main

* groupshared is not yet supported
* Output of header files 
* Output multiple entry points

# Internal Slang compiler features

These issues are more internal Slang features/improvements 

* Currently only generates C++ code, it would be fairly straight forward to support C (especially if we have 'intrinsic definitions')
* Have 'intrinsic definitions' in standard library - such that they can be generated where appropriate 
  + This will simplify the C/C++ code generation as means Slang language will generate must of the appropriate code
* Currently 'construct' IR inst is supported as is, we may want to split out to separate instructions for specific scenarios
* Refactoring around swizzle. Currently in emit it has to check for a variety of scenarios - could be simplified with an IR pass and perhaps more specific instructions. 
