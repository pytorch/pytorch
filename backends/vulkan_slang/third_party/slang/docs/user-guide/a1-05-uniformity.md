---
layout: user-guide
---

Uniformity Analysis
===========

On certain hardware, accessing resources with a non-uniform index may lead to significant performance degradation. Developers can often benefit from a compiler warning for unintentional non-uniform resource access.

Starting from v2024.1.0, Slang provides uniformity analysis that can warn users if a non-dynamically-uniform value is being used unintentionally. This feature is not enabled by default but can be turned on with the `-validate-uniformity` command-line option when using `slangc`, or the `CompilerOptionName::ValidateUniformity` compiler option when using the API.

In addition to specifying the compiler option, the source code must be augmented with the `dynamic_uniform` modifier to mark function parameters, struct fields or local variables as expecting a dynamic uniform value.

For example, the following code will trigger a warning:
```csharp
// Indicate that the `v` parameter needs to be dynamic uniform.
float f(dynamic_uniform float v)
{
    return v + 1.0;
}

[numthread(1,1,1)]
[shader("compute")]
void main(int tid : SV_DispatchThreadID)
{
    f(tid); // warning: tid is not dynamically uniform.
}
```

Currently, the analysis is being conservative for `struct` typed values, in that if any member of the `struct` is known to be non-uniform, the entire composite is
treated as non-uniform:
```csharp
struct MyType
{
    int a;
    int b;
}

void expectUniform(dynamic_uniform int a){}

void main(int tid : SV_DispatchThreadID)
{
    MyType t;
    t.a = tid;
    t.b = 0;

    // Generates a warning here despite t.b is non-uniform, because
    // t.a is non-uniform and that assignment makes `t` non-uniform.
    expectUniform(t.b);
}
```

To allow the compiler to provide more accurate analysis, you can use mark struct fields as
`dynamic_uniform`:

```csharp
struct MyType
{
    int a;
    dynamic_uniform int b;
}

void expectUniform(dynamic_uniform int a){}

void main(int tid : SV_DispatchThreadID)
{
    MyType t;
    t.a = tid;
    t.b = 0;

    // OK, because MyType::b is marked as dynamic_uniform.
    expectUniform(t.b);

    // Warning: trying to assign non-uniform value to dynamic_uniform location.
    t.b = tid;
}
```

## Treat Values as Uniform

In some cases, the compiler might not be able to deduce a value to be non-uniform. If you are certain that a value can
be treated as dynamic uniform, you can call `asDynamicUniform()` function to force the compiler to treat the value as
dynamic uniform. For example:
```csharp
void main(int tid: SV_DispatchThreadID)
{
    expectUniform(asDynamicUniform(tid)); // OK.
}
```

## Treat Function Return Values as Non-uniform

The uniformity analysis will automatically propagate uniformity to function return values. However, if you have
an intrinsic function that does not have a body, or you simply wish the return value of a function to be always
treated as non-uniform, you can mark the function with the `[NonUniformReturn]` attribute:
```csharp
[NonUniformReturn]
int f() { return 0; }
void expectUniform(dynamic_uniform int x) {}
void main()
{
    expectUniform(f()); // Warning.
}
```
