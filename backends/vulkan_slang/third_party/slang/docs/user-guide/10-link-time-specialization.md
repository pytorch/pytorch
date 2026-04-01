---
layout: user-guide
permalink: /user-guide/link-time-specialization
---

# Link-time Specialization and Module Precompilation

Traditionally, graphics developers have been relying on preprocessor defines to specialize their shader code for high-performance GPU execution.
While functioning systems can be built around preprocessor macros, overusing them leads to many problems:
- Long compilation time. With preprocessor defines, specialization happens before parsing, which is a very early stage in the compilation flow.
  This means that the compiler must redo almost all work from scratch with every specialized variant, including parsing, type checking, IR generation
  and optimization, even when two specialized variants only differ in one constant value. The lack of reuse of compiler front-end work between
  different shader specializations contributes a significant portion to long shader compile times.
- Reduced code readability and maintainability. The compiler cannot enforce any structures on preprocessor macros and cannot offer static checks to
  guarantee that the preprocessor macros are used in an intended way. Macros don't blend well with the native language syntax, which leads to less
  readable code, mystic diagnostic messages when things go wrong, and suboptimal intellisense experience.
- Locked in with early specialization. Once the code is written using preprocessor macros for specialization, the application that uses the shader
  code has no choice but to provide the macro values during shader compilation and always opt-in to static specialization. If the developer changes
  their mind to move away from specialization, a lot of code needs to be rewritten. As a result, the application is locked out of opportunities to
  take advantage of different design decisions or future hardware features that allow more efficient execution of non-specialized code.

Slang approaches the problem of shader specialization by supporting generics as a first class feature that allow most specializable code to be
written in strongly typed code, and by allowing specialization to be triggered through link-time constants or types.

As discussed in the [Compiling code with Slang](compiling) chapter, Slang provides a three-step compilation model: precompiling, linking and target code generation.
Assuming the user shader is implemented as three Slang modules: `a.slang`, `b.slang`, and `c.slang`, the user can precompile all three modules to binary IR and store
them as `a.slang-module`, `b.slang-module`, and `c.slang-module` in a complete offline process that is independent to any specialization arguments.
Next, these three IR modules are linked together to form a self-contained program that will then go through a set of compiler optimizations for target code generation.
Slang's compilation model allows specialization arguments, in the form of constants or types to be provided during linking. This means that specialization happens at
a much later stage of compilation, reusing all the work done during module precompilation.

## Link-time Constants

The simplest form of link time specialization is done through link-time constants. See the following code for an example.
```c++
// main.slang

// Define a constant whose value will be provided in another module at link time.
extern static const int kSampleCount;

float sample(int index) {...}

RWStructuredBuffer<float> output;
void main(uint tid : SV_DispatchThreadID)
{
    [ForceUnroll]
    for (int i = 0; i < kSampleCount; i++)
        output[tid] += sample(i);
}
```
This code defines a compute shader that can be specialized with different constant values of `kSampleCount`. The `extern` modifier means that
`kSampleCount` is a constant whose value is not provided within the current module, but will be resolved during the linking step.
The `main.slang` file can be compiled offline into a binary IR module with the `slangc` tool:
```
slangc main.slang -o main.slang-module
```

To specialize the code with a value of `kSampleCount`, the user can create another module that defines it:

```c++
// sample-count.slang
export static const int kSampleCount = 2;
```

This file can also be compiled separately:
```
slangc sample-count.slang -o sample-count.slang-module
```

With these two modules precompiled, we can link them together to get our specialized code:
```
slangc sample-count.slang-module main.slang-module -target hlsl -entry main -profile cs_6_0 -o main.hlsl
```

This process can also be done with Slang's compilation API as in the following code snippet:

```c++

ComPtr<slang::ISession> slangSession = ...;
ComPtr<slang::IBlob> diagnosticsBlob;

// Load the main module from file.
slang::IModule* mainModule = slangSession->loadModule("main.slang", diagnosticsBlob.writeRef());

// Load the specialization constant module from string.
const char* sampleCountSrc = R"(export static const int kSampleCount = 2;)";
auto sampleCountModuleSrcBlob = UnownedRawBlob::create(sampleCountSrc, strlen(sampleCountSrc));
slang::IModule* sampleCountModule = slangSession->loadModuleFromSource(
    "sample-count",  // module name
    "sample-count.slang", // synthetic module path
    sampleCountModuleSrcBlob);  // module source content

// Compose the modules and entry points.
ComPtr<slang::IEntryPoint> computeEntryPoint;
SLANG_RETURN_ON_FAIL(
    module->findEntryPointByName(entryPointName, computeEntryPoint.writeRef()));

std::vector<slang::IComponentType*> componentTypes;
componentTypes.push_back(mainModule);
componentTypes.push_back(computeEntryPoint);
componentTypes.push_back(sampleCountModule);

ComPtr<slang::IComponentType> composedProgram;
SlangResult result = slangSession->createCompositeComponentType(
    componentTypes.data(),
    componentTypes.size(),
    composedProgram.writeRef(),
    diagnosticsBlob.writeRef());

// Link.
ComPtr<slang::IComponentType> linkedProgram;
composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());

// Get compiled code.
ComPtr<slang::IBlob> compiledCode;
linkedProgram->getEntryPointCode(0, 0, compiledCode.writeRef(), diagnosticBlob.writeRef());

```

## Link-time Types

In addition to constants, you can also define types that are specified at link-time. For example, given the following modules:

```csharp
// common.slang
interface ISampler
{
    int getSampleCount();
    float sample(int index);
}
struct FooSampler : ISampler
{
    int getSampleCount() { return 1; }
    float sample(int index) { return 0.0; }
}
struct BarSampler : ISampler
{
    int getSampleCount() { return 2; }
    float sample(int index) { return index * 0.5; }
}
```

```csharp
// main.slang
import common;
extern struct Sampler : ISampler;

RWStructuredBuffer<float> output;
void main(uint tid : SV_DispatchThreadID)
{
    Sampler sampler;
    [ForceUnroll]
    for (int i = 0; i < sampler.getSampleCount(); i++)
        output[tid] += sampler.sample(i);
}
```

Again, we can separately compile these modules into binary forms independently from how they will be specialized.
To specialize the shader, we can author a third module that provides a definition for the `extern Sampler` type:

```csharp
// sampler.slang
import common;
export struct Sampler : ISampler = FooSampler;
```

The `=` syntax is a syntactic sugar that expands to the following code:

```csharp
export struct Sampler : ISampler
{
    FooSampler inner;
    int getSampleCount() { return inner.getSampleCount(); }
    float sample(int index) { return inner.sample(index); }
}
```

When all these three modules are linked, we will produce a specialized shader that uses the `FooSampler`.

## Providing Default Settings

When defining an `extern` symbol as a link-time constant or type, it is allowed to provide a default value for that constant or type.
When no other modules exists to `export` the same-named symbol, the default value will be used in the linked program.

For example, the following code is considered complete at linking and can proceed to code generation without any issues:
```c++
// main.slang

// Provide a default value when no other modules are exporting the symbol.
extern static const int kSampleCount = 2;
// ... 
void main(uint tid : SV_DispatchThreadID)
{
    [ForceUnroll]
    for (int i = 0; i < kSampleCount; i++)
        output[tid] += sample(i);
}
```

## Restrictions

Unlike preprocessors, link-time constants and types can only be used in places where shader parameter layout cannot be
affected. This means that link-time constants and types are subject to the following restrictions:
- Link-time constants cannot be used to define array sizes.
- Link-time types are considered "incomplete" types. A struct or array type that has incomplete typed element is also an incomplete type.
  Incomplete types cannot be used as `ConstantBuffer` or `ParameterBlock` element type, and cannot be used directly as the type of
  a uniform variable.

However it is allowed to use incomplete types as the element type of `StructuredBuffer` or `GLSLStorageBuffer`.

## Using Precompiling Modules with the API

In addition to using `slangc` for precompiling Slang modules, the `IModule` class provides a method to serialize itself to disk:

```C++
/// Get a serialized representation of the checked module.
SlangResult IModule::serialize(ISlangBlob** outSerializedBlob);

/// Write the serialized representation of this module to a file.
SlangResult IModule::writeToFile(char const* fileName);
```

These functions will write only the module itself to a file, which excludes the modules that it includes. To write all imported
modules, you can use methods from the `ISession` class to enumerate all currently loaded modules (including transitively imported modules)
in the session:

```c++
SlangInt ISession::getLoadedModuleCount();
IModule* ISession::getLoadedModule(SlangInt index);
```

Additionally, the `ISession` class also provides a function to query if a previously compiled module is still up-to-date with the current
Slang version, the compiler options in the session and the current content of the source files used to compile the module:

```c++
bool ISession::isBinaryModuleUpToDate(
    const char* modulePath,
    slang::IBlob* binaryModuleBlob);
```

If the compiler options or source files have been changed since the module was last compiled, the `isBinaryModuleUpToDate` will return false.

The compiler can be setup to automatically use the precompiled modules when they exist and up-to-date. When loading a module,
either triggered via the `ISession::loadModule` call or via transitive `import`s in the modules being loaded, the compiler will look in the
search paths for a `.slang-module` file first. If it exists, it will load the precompiled module instead of compiling from the source.
If you wish the compiler to verify whether the `.slang-module` file is up-to-date before loading it, you can specify the `CompilerOptionName::UseUpToDateBinaryModule` to `1`
when creating the session. When this option is set, the compiler will verify the precompiled module is still update, and will recompile the module
from source if it is not up-to-date.


## Additional Remarks

Link-time specialization is Slang's answer to compile-time performance and modularity issues associated with preprocessor
based shader specialization. By representing specializable settings as link-time constants or link-time types, we are able
to defer shader specialization to link time, allowing reuse of all the front-end compilation work that includes tokenization,
parsing, type checking, IR generation and validation. As Slang evolves to support more language features and as the user code
is growing to be more complex, the cost of front-end compilation will only increase over time. By using link-time specialization
on precompiled modules, an application can be completely isolated from any front-end compilation cost.