---
layout: user-guide
permalink: /user-guide/convenience-features
---

# Basic Convenience Features

This topic covers a series of nice-to-have language features in Slang. These features are not supported by HLSL but are introduced to Slang to simplify code development. Many of these features are added to Slang per request of our users. 

## Type Inference in Variable Definitions
Slang supports automatic variable type inference:
```csharp
var a = 1; // OK, `a` is an `int`.
var b = float3(0, 1, 2); // OK, `b` is a `float3`.
```
Automatic type inference requires an initialization expression to be present. Without an initial value, the compiler is not able to infer the type of the variable. The following code will result in a compiler error:
```csharp
var a; // Error, cannot infer the type of `a`.
```

You may use the `var` keyword to define a variable in a modern syntax:
```csharp
var a : int = 1; // OK.
var b : int; // OK.
```

## Immutable Values
The `var` syntax and the traditional C-style variable definition introduce a _mutable_ variable whose value can be changed after its definition. If you wish to introduce an immutable or constant value, you may use the `let` keyword:
```rust
let a = 5; // OK, `a` is `int`.
let b : int = 5; // OK.
```
Attempting to change an immutable value will result in a compiler error:
```rust
let a = 5;
a = 6; // Error, `a` is immutable.
```


## Namespaces

You can use the `namespace` syntax to define symbols in a namespace:
```csharp
namespace ns
{
    int f();
}
```

Slang also supports the abbreviated syntax for defining nested namespaces:
```csharp
namespace ns1.ns2
{
    int f();
}
// equivalent to:
namespace ns1::ns2
{
    int f();
}
// equivalent to:
namespace ns1
{
    namespace ns2
    {
        int f();
    }
}
```

To access symbols defined in a namespace, you can use their qualified name with namespace prefixes:
```csharp
void test()
{
    ns1.ns2.f();
    ns1::ns2::f(); // equivalent syntax.
}
```

Symbols defined in the same namespace can access each other without a qualified name, this is true even if the referenced symbol is defined in a different file or module:
```csharp
namespace ns
{
    int f();
    int g() { f(); } // OK.
}
```

You can also use the `using` keyword to pull symbols defined in a different namespace to
the current scope, removing the requirement for using fully qualified names.
```cpp
namespace ns1.ns2
{
    int f();
}

using ns1.ns2;
// or:
using namespace ns1.ns2; // alternative syntax.

void test() { f(); } // OK.
```

## Member functions

Slang supports defining member functions in `struct`s. For example, it is allowed to write:

```hlsl
struct Foo
{
    int compute(int a, int b)
    {
        return a + b;
    }
}
```

You can use the `.` syntax to invoke member functions:

```hlsl
Foo foo;
int rs = foo.compute(1,2);
```

Slang also supports static member functions, For example:
```
struct Foo
{
    static int staticMethod(int a, int b)
    {
        return a + b;
    }
}
```

Static member functions are accessed the same way as other static members, via either the type name or an instance of the type:

```hlsl
int rs = Foo.staticMethod(a, b);
```

or

```hlsl
Foo foo;
...
int rs = foo.staticMethod(a,b);
```

### Mutability of member function

For GPU performance considerations, the `this` argument in a member function is immutable by default. Attempting to modify `this` will result in a compile error. If you intend to define a member function that mutates the object, use `[mutating]` attribute on the member function as shown in the following example.

```hlsl
struct Foo
{
    int count;
    
    [mutating]
    void setCount(int x) { count = x; }

    // This would fail to compile.
    // void setCount2(int x) { count = x; }
}

void test()
{
    Foo f;
    f.setCount(1); // Compiles
}
```

## Properties

Properties provide a convenient way to access values exposed by a type, where the logic behind accessing the value is defined in `getter` and `setter` function pairs. Slang's `property` feature is similar to C# and Swift. 
```csharp
struct MyType
{
    uint flag;

    property uint highBits
    {
        get { return flag >> 16; }
        set { flag = (flag & 0xFF) + (newValue << 16); }
    }
};
```

Or equivalently in a "modern" syntax:

```csharp
struct MyType
{
    uint flag;

    property highBits : uint
    {
        get { return flag >> 16; }
        set { flag = (flag & 0xFF) + (newValue << 16); }
    }
};
```

You may also use an explicit parameter for the setter method:
```csharp
property uint highBits
{
    set(uint x) { flag = (flag & 0xFF) + (x << 16);  }
}
```

> #### Note ####
> Slang currently does not support automatically synthesized `getter` and `setter` methods. For example,
> the following code is not supported:
> ```
> property uint highBits {get;set;} // Not supported yet.
> ```

## Initializers

### Constructors
> #### Note ####
> The syntax for defining constructors is subject to future change.


Slang supports defining constructors in `struct` types. You can write:
```csharp
struct MyType
{
    int myVal;
    __init(int a, int b)
    {
        myVal = a + b;
    }
}
```

You can use a constructor to construct a new instance by using the type name in a function call expression:
```csharp
MyType instance = MyType(1,2);  // instance.myVal is 3.
```

You may also use C++ style initializer list to invoke a constructor:
```csharp
MyType instance = {1, 2};
```

If a constructor does not define any parameters, it will be recognized as *default* constructor that will be automatically called at the definition of a variable:

```csharp
struct MyType
{
    int myVal;
    __init()
    {
        myVal = 10;
    }
};

int test()
{
    MyType test;
    return test.myVal; // returns 10.
}
```

Slang will also implicitly call a *default* constructor of all parents of a derived struct (same as C++):
```csharp
struct MyType_Base
{
    int myVal1;
    __init() {myVal1 = 22;}
}

struct MyType1 : MyType_Base
{
    int myVal2;
    __init()
    {
        // implicitly calls `MyType_Base::__init()`
        myVal2 = 15;
    }
}
testMyType1()
{
    MyType1 a;
    // a.myVal1 == 22
    // a.myVal2 == 15
}

struct MyType2 : MyType_Base
{
}
testMyType2()
{
    MyType2 b; // implicitly calls `MyType_Base::__init()`
    // b.myVal1 == 22
}
```

### Member Init Expressions

Slang supports member init expressions:
```csharp
struct MyType
{
    int myVal = 5;
}
```

## Operator Overloading

Slang allows defining operator overloads as global methods:
```csharp
struct MyType
{
    int val;
    __init(int x) { val = x; }
}

MyType operator+(MyType a, MyType b)
{
    return MyType(a.val + b.val);
}

int test()
{
    MyType rs = MyType(1) + MyType(2);
    return rs.val; // returns 3.
}
```
Slang currently supports overloading the following operators: `+`, `-`, `*`, `/`, `%`, `&`, `|`, `<`, `>`, `<=`, `>=`, `==`, `!=`, unary `-`, `~` and `!`. Please note that the `&&` and `||` operators are not supported.

In addition, you can overload operator `()` as a member method:
```csharp
struct MyFunctor
{
    int operator()(float v)
    {
        // ...
    }
}
void test()
{
    MyFunctor f;
    int x = f(1.0f); // calls MyFunctor::operator().
    int y = f.operator()(1.0f); // explicitly calling operator().
}
```

## Subscript Operator

Slang allows overriding `operator[]` with `__subscript` syntax:
```csharp
struct MyType
{
    int val[12];
    __subscript(int x, int y) -> int
    {
        get { return val[x*3 + y]; }
        set { val[x*3+y] = newValue; }
    }
}
int test()
{
    MyType rs;
    rs[0, 0] = 1;
    rs[1, 0] = rs[0, 0] + 1;
    return rs[1, 0]; // returns 2.
}
```

## Tuple Types

Tuple types can hold collection of values of different types.
Tuples types are defined in Slang with the `Tuple<...>` syntax, and
constructed with either a constructor or the `makeTuple` function:
```csharp
Tuple<int, float, bool> t0 = Tuple<int, float, bool>(5, 2.0f, false);
Tuple<int, float, bool> t1 = makeTuple(3, 1.0f, true);
```

Tuple elements can be accessed with `_0`, `_1` member names:
```csharp
int i = t0._0; // 5
bool b = t1._2; // true
```

You can use the swizzle syntax similar to vectors and matrices to form new
tuples:

```csharp
t0._0_0_1 // evaluates to (5, 5, 2.0f)
```

You can concatenate two tuples:

```csharp
concat(t0, t1) // evaluates to (5, 2.0f, false, 3, 1.0f, true)
```

If all element types of a tuple conforms to `IComparable`, then the tuple itself
will conform to `IComparable`, and you can use comparison operators on the tuples
to compare them:

```csharp
let cmp = t0 < t1; // false
```

You can use `countof()` on a tuple type or a tuple value to obtain the number of
elements in a tuple. This is considered a compile-time constant.
```csharp
int n = countof(Tuple<int, float>); // 2
int n1 = countof(makeTuple(1,2,3)); // 3
```

All tuple types will be translated to `struct` types, and receive the same layout
as `struct` types.

## `Optional<T>` type

Slang supports the `Optional<T>` type to represent a value that may not exist.
The dedicated `none` value can be used for any `Optional<T>` to represent no value.
`Optional<T>::value` property can be used to retrieve the value.

```csharp
struct MyType
{
    int val;
}

int useVal(Optional<MyType> p)
{
    if (p == none)        // Equivalent to `!p.hasValue`
        return 0;
    return p.value.val;
}

int caller()
{
    MyType v;
    v.val = 0;
    useVal(v);  // OK to pass `MyType` to `Optional<MyType>`.
    useVal(none);  // OK to pass `none` to `Optional<MyType>`.
    return 0;
}
```

## `if_let` syntax
Slang supports `if (let name = expr)` syntax to simplify the code when working with `Optional<T>` value. The syntax is similar to Rust's
`if let` syntax, the value expression must be an `Optional<T>` type, for example:

```csharp
Optional<int> getOptInt() { ... }

void test()
{
    if (let x = getOptInt())
    {
        // if we are here, `getOptInt` returns a value `int`.
        // and `x` represents the `int` value.
    }
}
```

## `reinterpret<T>` operation

Sometimes it is useful to reinterpret the bits of one type as another type, for example:
```csharp
struct MyType
{
    int a;
    float2 b;
    uint c;
}

MyType myVal;
float4 myPackedVector = packMyTypeToFloat4(myVal);
```

The `packMyTypeToFloat4` function is usually implemented by bit casting each field in the source type and assigning it into the corresponding field in the target type,
by calling `intAsFloat`, `floatAsInt` and using bit operations to shift things in the right place.
Instead of writing `packMyTypeToFloat4` function yourself, you can use Slang's built-in `reinterpret<T>` to do just that for you:
```
float4 myPackedVector = reinterpret<float4>(myVal);
```

`reinterpret` can pack any type into any other type as long as the target type is no smaller than the source type.

## Pointers (limited)

Slang supports pointers when generating code for SPIRV, C++ and CUDA targets. The syntax for pointers is similar to C, with the exception that operator `.` can also be used to dereference a member from a pointer. For example:
```csharp
struct MyType
{
    int a;
};

int test(MyType* pObj)
{
    MyType* pNext = pObj + 1;
    MyType* pNext2 = &pNext[1];
    return pNext.a + pNext->a + (*pNext2).a + pNext2[0].a;
}

cbuffer Constants
{
    MyType *ptr;
};

int validTest()
{
    return test(ptr);
}

int invalidTest()
{
    // cannot produce a pointer from a local variable 
    MyType obj;
    return test(&obj); // !! ERROR !!
}
```

Pointer types can also be specified using the generic syntax: `Ptr<MyType>` is equivalent to `MyType*`.

### Limitations

- Slang supports pointers to global memory, but not shared or local memory. For example, it is invalid to define a pointer to a local variable.

- Slang supports pointers that are defined as shader parameters (e.g. as a constant buffer field).

- Slang can produce pointers using the & operator from data in global memory.

- Slang doesn't support forming pointers to opaque handle types, e.g. `Texture2D`. For handle pointers, use `DescriptorHandle<T>` instead.

- Slang doesn't support coherent load/stores.

- Slang doesn't support custom alignment specification.

- Slang currently does not support pointers to immutable values, i.e. `const T*`.

## `DescriptorHandle` for Bindless Descriptor Access

Slang supports the `DescriptorHandle<T>` type that represents a bindless handle to a resource. This feature provides a portable way of implementing
the bindless resource idiom. When targeting HLSL, GLSL and SPIRV where descriptor types (e.g. textures, samplers and buffers) are opaque handles,
`DescriptorHandle<T>` will translate into a `uint2` so it can be defined in any memory location. The underlying `uint2` value is treated as an index
to access the global descriptor heap or resource array in order to obtain the actual resource handle. On targets with where resource handles
are not opaque handles, `DescriptorHandle<T>` maps to `T` and will have the same size and alignment defined by the target.

`DescriptorHandle<T>` is declared as:
```slang
struct DescriptorHandle<T> where T:IOpaqueDescriptor {}
```
where `IOpaqueDescriptor` is an interface implemented by all resource types, including textures,
`ConstantBuffer`, `RaytracingAccelerationStructure`, `SamplerState`, `SamplerComparisonState` and all types of `StructuredBuffer`.

You may also write `Texture2D.Handle` as a short-hand of `DescriptorHandle<Texture2D>`.

`DescriptorHandle<T>` supports `operator *`, `operator ->`, and can implicitly convert to `T`, for example:

```slang
uniform StructuredBuffer<DescriptorHandle<Texture2D>> textures;
uniform int textureIndex;

// define a descriptor handle using builtin convenience typealias:
uniform StructuredBuffer<float4>.Handle output;

[numthreads(1,1,1)]
void main()
{
    output[0] = textures[textureIndex].Load(int3(0));

    // Alternatively, this syntax is also valid:
    (*output)[0] = textures[textureIndex]->Load(int3(0));
}
```

By default, when targeting HLSL, `DescriptorHandle<T>` translates to uses of `ResourceDescriptorHeap[index]` and `SamplerDescriptorHeap[index]`.
In particular, when combined with combined texture sampler types (e.g. `Sampler2D`), Slang will fetch the texture using the first
component of the handle, and the sampler state from the second component of the handle. For example:

```
uniform DescriptorHandle<Sampler2D> s;
void test()
{
    s.Sample(uv);
}
```

translates to:

```hlsl
uniform uint2 s;
void test()
{
    Texture2D(ResourceDescriptorHeap[s.x]).Sample(
        SamplerState(SamplerDescriptorHeap[s.y]),
        uv
    );
}
```

When targeting SPIRV, Slang will introduce a global array of descriptors and fetch from the global array.
The descriptor set ID of the global descriptor array can be configured with the `-bindless-space-index`
(or `CompilerOptionName::BindlessSpaceIndex` when using the API) option.

> #### Note
> The default implementation for SPIRV may change in the future if SPIRV is extended to provide what is
> equivalent to D3D's `ResourceDescriptorHeap` construct.

Users can override the default behavior of convering from bindless handle to resource handle, by providing a
`getDescriptorFromHandle` in user code. For example:

```slang
// All texture and buffer handles are defined in descriptor set 100.
[vk::binding(0, 100)]
__DynamicResource<__DynamicResourceKind.General> resourceHandles[];

// All sampler handles are defined in descriptor set 101.
[vk::binding(0, 101)]
__DynamicResource<__DynamicResourceKind.Sampler> samplerHandles[];

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    __target_switch
    {
    case spirv:
        if (T.kind == ResourceKind.Sampler)
            return samplerHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
        else
            return resourceHandles[((uint2)handle).x].asOpaqueDescriptor<T>();
    default:
        return defaultGetDescriptorFromHandle(handle);
    }
}
```

Note that the `getDescriptorFromHandle` is not supposed to be called from the user code directly,
it will be automatically called by the compiler to dereference a `DescriptorHandle<T>` to get `T`.
Think about providing `getDescriptorFromHandle` as a way to override `operator->` for `DescriptorHandle<T>`. 

The `IOpaqueDescriptor` interface is defined as:

```slang
interface IOpaqueDescriptor
{
    /// The kind of the descriptor.
    static const DescriptorKind kind;
    static const DescriptorAccess descriptorAccess;
}
```

The user can call `defaultGetDescriptorFromHandle` function from their implementation of
`getDescriptorFromHandle` to dispatch to the default behavior.

The `kind` and `descriptorAccess` constants allows user code to fetch from different locations
depending on the type and access of the resource being requested. The `DescriptorKind` and
`DescriptorAccess` enums are defined as:

```slang
enum DescriptorKind
{
    Unknown, /// Unknown descriptor kind.
    Texture, /// A texture descriptor.
    CombinedTextureSampler, /// A combined texture and sampler state descriptor.
    Buffer, /// A buffer descriptor.
    Sampler, /// A sampler state descriptor.
    AccelerationStructure, /// A ray tracing acceleration structure descriptor.
}

enum DescriptorAccess
{
    Unknown = -1,
    Read = 0,
    Write = 1,
    ReadWrite = 2,
    RasterizerOrdered = 3,
    Feedback = 4,
}
```

By default, the value of a `DescriptorHandle<T>` object is assumed to be dynamically uniform across all
execution threads. If this is not the case, the user is required to mark the `DescriptorHandle` as `nonuniform`
*immediately* before dereferencing it:
```slang
void test(DescriptorHandle<Texture2D> t)
{
    nonuniform(t)->Sample(...);
}
```

If the resource pointer value is not uniform and `nonuniform` is not called, the result may be
undefined.

Extensions
--------------------
Slang allows defining additional methods for a type outside its initial definition. For example, suppose we already have a type defined:

```csharp
struct MyType
{
    int field;
    int get() { return field; }
}
```

You can extend `MyType` with new method members:
```csharp
extension MyType
{
    float getNewField() { return newField; }
}
```

All locations that sees the definition of the `extension` can access the new members:

```csharp
void test()
{
    MyType t;
    float val = t.getNewField();
}
```

This feature is similar to extensions in Swift and extension methods in C#.

> #### Note:
> You can only extend a type with additional methods. Extending with additional data fields is not allowed.

Multi-level break
-------------------

Slang allows `break` statements with a label to jump into any ancestor control flow break points, and not just the immediate parent.
Example:
```
outer:
for (int i = 0; i < 5; i++)
{
    inner:
    for (int j = 0; j < 10; j++)
    {
        if (someCondition)
            break outer;
    }
}
```

Force inlining
-----------------
Most of the downstream shader compilers will inline all the function calls. However you can instruct Slang compiler to do the inlining
by using the `[ForceInline]` decoration:
```
[ForceInline]
int f(int x) { return x + 1; }
```


Special Scoping Syntax
-------------------
Slang supports three special scoping syntax to allow users to mix in custom decorators and content in the shader code. These constructs allow a rendering engine to define custom meta-data in the shader, or map engine-specific block syntax to a meaningful block that is understood by the compiler via proper `#define`s.

### `__ignored_block`
An ignored block will be parsed and ignored by the compiler:
```
__ignored_block
{
    arbitrary content in the source file,
    will be ignored by the compiler as if it is a comment.
    Can have nested {} here.
}
```

### `__transparent_block`
Symbols defined in a transparent block will be treated as if they are defined
in the parent scope:
```csharp
struct MyType
{
    __transparent_block
    {
        int myFunc() { return 0; }
    }
}
```
Is equivalent to:
```csharp
struct MyType
{
    int myFunc() { return 0; }
}
```

### `__file_decl`
Symbols defined in a `__file_decl` will be treated as if they are defined in
the global scope. However, symbols defined in different `__file_decl`s is not visible
to each other. For example:
```csharp
__file_decl
{
    void f1()
    {
    }
}
__file_decl
{
    void f2()
    {
        f1(); // error: f1 is not visible from here.
    }
}
```

User Defined Attributes (Experimental)
-------------------

In addition to many system defined attributes, users can define their own custom attribute types to be used in the `[UserDefinedAttribute(args...)]` syntax. The following example shows how to define a custom attribute type.

```csharp
[__AttributeUsage(_AttributeTargets.Var)]
struct MaxValueAttribute
{
    int value;
    string description;
};

[MaxValue(12, "the scale factor")]
uniform int scaleFactor;
```

In the above code, the `MaxValueAttribute` struct type is decorated with the `[__AttributeUsage]` attribute, which informs that `MaxValueAttribute` type should be interpreted as a definition for a user-defined attribute, `[MaxValue]`, that can be used to decorate all variables or fields. The members of the struct defines the argument list for the attribute.

The `scaleFactor` uniform parameter is declared with the user defined `[MaxValue]` attribute, providing two arguments for `value` and `description`.

The `_AttributeTargets` enum is used to restrict the type of decls the attribute can apply. Possible values of `_AttributeTargets` can be `Function`, `Param`, `Struct` or `Var`.

The usage of user-defined attributes can be queried via Slang's reflection API through `TypeReflection` or `VariableReflection`'s `getUserAttributeCount`, `getUserAttributeByIndex` and `findUserAttributeByName` methods. 
