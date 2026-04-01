---
layout: user-guide
permalink: /user-guide/reflection
---

Using the Reflection API
=========================

This chapter provides an introduction to the Slang reflection API.
Our goals in this chapter are to:

* Demonstrate the recommended types and operations to use for the most common reflection scenarios

* Provide an underlying mental model for how Slang's reflection information represents the structure of a program

We will describe the structure of a program that traverses all of the parameters of a shader program and prints information (including binding locations) for them.
The code shown here is derived from the [reflection-api](https://github.com/shader-slang/slang/tree/master/examples/reflection-api) example that is included in the Slang repository.
Readers may find it helpful to follow along with that code, to see a more complete picture of what is presented here.

Compiling a Program
-------------------

The first step in reflecting a shader program is, unsurprisingly, to compile it.
Currently reflection information cannot be queried from code compiled via the command-line `slangc` tool, so applications that want to perform reflection on Slang shader code should use the [compilation API](./compiling#using-the-compilation-api) to compile a program, and then use `getLayout()` to extract reflection information:

```c++
slang::IComponentType* program = ...;
slang::ProgramLayout* programLayout = program->getLayout(targetIndex);
```

For more information, see the [relevant section](./compiling#layout-and-reflection) of the chapter on compilation.

Types and Variables
-------------------

We start our discussion of the reflection API with two of the fundamental building blocks used to represent the structure of a program: types and variables.

A key property of GPU shader programming is that the same type may be laid out differently, depending on how it is used.
For example, a user-defined `struct` type `Stuff` will often be laid out differently if it is used in a `ConstantBuffer<Stuff>` than in a `StructuredBuffer<Stuff>`.

Because the same thing can be laid out in multiple ways (even within the same program), the Slang reflection API represents types and variables as distinct things from the *layouts* applied to them.
This section focuses only on the underlying types/variables, while later sections will build on these concepts to show how layouts can be reflected.

### Variables

A `VariableReflection` represents a variable declaration in the input program.
Variables include global shader parameters, fields of `struct` types, and entry-point parameters.

Because a `VariableReflection` does not include layout information, the main things that can be queried on it are just its name and type:

```c++
void printVariable(
    slang::VariableReflection* variable)
{
    const char* name = variable->getName();
    slang::TypeReflection* type = variable->getType();

    print("name: ");    printQuotedString(name);
    print("type: ");    printType(type);
}
```

### Types

A `TypeReflection` represents some type in the input program.
There are various different *kinds* of types, such as arrays, user-defined `struct` types, and built-in types like `int`.
The reflection API represents these different cases with the `TypeReflection::Kind` enumeration.

On its own, a `TypeReflection` does not include layout information.

We will now start building a function for printing information about types:

```c++
void printType(slang::TypeReflection* type)
{
    const char* name = type->getName();
    slang::TypeReflection::Kind kind = type->getKind();

    print("name: ");    printQuotedString(name);
    print("kind: ");    printTypeKind(kind);

    // ...
}
```

Given what has been presented so far, if we have a Slang variable declaration like the following:

```hlsl
float x;
```

then applying `printVariable()` to a `VariableReflection` for `x` would yield:

```
name: "x"
type:
  name: "float"
  kind: Scalar
```

Additional information can be queried from a `TypeReflection`, depending on its kind:

```c++
void printType(slang::TypeReflection* type)
{
    // ...

    switch(type->getKind())
    {
    default:
        break;

    // ...
    }
}
```

The following subsections will show examples of what can be queried for various kinds of types.

#### Scalar Types

Scalar types store an additional enumerant to indicate which of the built-in scalar types is being represented:

```c++
case slang::TypeReflection::Kind::Scalar:
    {
        print("scalar type: ");
        printScalarType(type->getScalarType());
    }
    break;
```

The `slang::ScalarType` enumeration includes cases for the built-in integer and floating-point types (for example, `slang::ScalarType::UInt64` and `slang::ScalarType::Float16`), as well as the basic `bool` type (`slang::ScalarType::Bool`).
The `void` type is also considered a scalar type (`slang::ScalarType::Void`);

#### Structure Types

A structure type may have zero or more *fields*.
Each field is represented as a `VariableReflection`.
A `TypeReflection` allows the fields to be enumerated using `getFieldCount()` and `getFieldByIndex()`.

```c++
case slang::TypeReflection::Kind::Struct:
    {
        print("fields:");
        int fieldCount = type->getFieldCount();
        for (int f = 0; f < fieldCount; f++)
        {
            print("- ");
            slang::VariableReflection* field =
                type->getFieldByIndex(f);
            printVariable(field);
        }
    }
    break;
```

For the purposes of the reflection API, the fields of a `struct` type are its non-static members (both `public` and non-`public`).

Given Slang code like the following:

```hlsl
struct S
{
    int a;
    float b;
}
```

Reflection on type `S` would yield:

```
name: "S"
kind: Struct
fields:
  - name: "a"
    type:
      name: "int"
      kind: Scalar
  - name: "b"
    type:
      name: "float"
      kind: Scalar
```

#### Arrays

An array type like `int[3]` is defined by the number and type of elements in the array, which can be queried with `getElementCount()` and `getElementType`, respectively:

```c++
case slang::TypeReflection::Kind::Array:
    {
        print("element count: ");
        printPossiblyUnbounded(type->getElementCount());

        print("element type: ");
        printType(type->getElementType());
    }
    break;
```

Some array types, like `Stuff[]`, have *unbounded* size.
The Slang reflection API represents this case using the maximum value possible for the `size_t` result from `getElementCount()`:

```c++
void printPossiblyUnbounded(size_t value)
{
    if (value == ~size_t(0))
    {
        printf("unbounded");
    }
    else
    {
        printf("%u", unsigned(value));
    }
}
```

#### Vectors

Vector types like `int3` are similar to arrays, in that they are defined by their element type and number of elements:

```c++
case slang::TypeReflection::Kind::Vector:
    {
        print("element count: ");
        printCount(type->getElementCount());

        print("element type: ");
        printType(type->getElementType());
    }
    break;
```

#### Matrices

Matrix types like `float3x4` are defined by the number of rows, the number of columns, and the element type:

```c++
case slang::TypeReflection::Kind::Matrix:
    {
        print("row count: ");
        printCount(type->getRowCount());

        print("column count: ");
        printCount(type->getColumnCount());

        print("element type: ");
        printType(type->getElementType());
    }
    break;
```

#### Resources

There are a wide range of resource types, including simple cases like `TextureCube` and `StructuredBuffer<int>`, as well as quite complicated ones like `RasterizerOrderedTexture2DArray<int4>` and `AppendStructuredBuffer<Stuff>`.

The Slang reflection API breaks down the properties of a resource type into its shape, access, and result type:

```c++
case slang::TypeReflection::Kind::Resource:
    {
        key("shape");
        printResourceShape(type->getResourceShape());

        key("access");
        printResourceAccess(type->getResourceAccess());

        key("result type");
        printType(type->getResourceResultType());
    }
    break;
```

The *result type* of a resource is simply whatever would be returned by a basic read operation on that resource.
For resource types in Slang code, the result type is typically written as a generic type parameter after the type name.
For a `StructuredBuffer<Thing>` the result type is `Thing`, while for a `Texture2D<int3>` it is `int3`.
A texture type like `Texture2D` that does not give an explicit result type has a default result type of `float4`.

The *access* of a resource (`SlangResourceAccess`) represents how the elements of the resource may be accessed by shader code.
For Slang resource types, access is typically encoded as a prefix on the type name.
For example, an unprefixed `Texture2D` has read-only access (`SLANG_RESOURCE_ACCESS_READ`), while a `RWTexture2D` has read-write access (`SLANG_RESOURCE_ACCESS_READ_WRITE`).

The *shape* of a resource (`SlangResourceShape`) represents the conceptual rank/dimensionality of the resource and how it is indexed.
For Slang resource type names, everything after the access prefix is typically part of the shape.

A resource shape breaks down into a *base shape* along with a few possible suffixes like array-ness:

```c++
void printResourceShape(SlangResourceShape shape)
{
    print("base shape:");
    switch(shape & SLANG_BASE_SHAPE_MASK)
    {
    case SLANG_TEXTURE1D: printf("TEXTURE1D"); break;
    case SLANG_TEXTURE2D: printf("TEXTURE2D"); break;
    // ...
    }

    if(shape & SLANG_TEXTURE_ARRAY_FLAG) printf("ARRAY");
    if(shape & SLANG_TEXTURE_MULTISAMPLE_FLAG) printf("MULTISAMPLE");
    // ...
}
```

#### Single-Element Containers

Types like `ConstantBuffer<T>` and `ParameterBlock<T>` represent a grouping of parameter data, and behave like an array or structured buffer with only a single element:

```c++
case slang::TypeReflection::Kind::ConstantBuffer:
case slang::TypeReflection::Kind::ParameterBlock:
case slang::TypeReflection::Kind::TextureBuffer:
case slang::TypeReflection::Kind::ShaderStorageBuffer:
    {
        key("element type");
        printType(type->getElementType());
    }
    break;
```

Layout for Types and Variables
------------------------------

The Slang reflection API provides `VariableLayoutReflection` and `TypeLayoutReflection` to represent a *layout* of a given variable or type.
As discussed earlier, the same type might have multiple different layouts used for it in the same program.

### Layout Units

A key challenge that the Slang reflection API has to address is how to represent the offset of a variable (or struct field, etc.) or the size of a type when `struct` types are allowed to mix various kinds of data together.

For example, consider the following Slang code:

```hlsl
struct Material
{
    Texture2D albedoMap;
    SamplerState sampler;
    float2 uvScale;
    float2 uvBias;
}
struct Uniforms
{
    TextureCube environmentMap;
    SamplerState environmentSampler;
    float3 sunLightDirection;
    float3 sunLightIntensity;
    Material material;
    // ...
}
ParameterBlock<Uniforms> uniforms;
```

When laid out in the given parameter block, what is the offset of the field `Uniforms::material`? What is the size of the `Material` type?

The key insight is that layout is multi-dimensional: the same type can have a size in multiple distinct units.
For example, when compiling the above code for D3D12/DXIL, the answer is that the `Uniforms::material` has an offset of one `t` register, one `s` register, and 32 bytes.
Similarly, the size of the `Material` type is one `t` register, one `s` register, and 16 bytes.

We refer to these distinct units of measure used in layouts (including bytes, `t` registers, and `s` registers) as *layout units*.
Layout units are represented in the Slang reflection API with the `slang::ParameterCategory` enumeration.
(We will avoid the term "parameter category," despite that being the name currently exposed in the public API; that name has turned out to be a less-than-ideal choice).

### Variable Layouts

A `VariableLayoutReflection` represents a layout computed for a given variable (itself a `VariableReflection`).
The underlying variable can be accessed with `getVariable()`, but the variable layout also provides accessors for the most important properties.

A variable layout stores the offsets of that variable (possibly in multiple layout units), and also a type layout for the data stored in the variable.

```c++
void printVarLayout(slang::VariableLayoutReflection* varLayout)
{
    print("name"); printQuotedString(varLayout->getName());

    printRelativeOffsets(varLayout);

    key("type layout");
    printTypeLayout(varLayout->getTypeLayout());
}
```

#### Offsets

The offsets stored by a `VariableLayoutReflection` are always *relative* to the enclosing `struct` type, scope, or other context that surrounds the variable.

The `VariableLayoutReflection::getOffset` method can be used to query the relative offset of a variable for any given layout unit:

```c++
void printOffset(
    slang::VariableLayoutReflection* varLayout,
    slang::ParameterCategory layoutUnit)
{
    size_t offset = varLayout->getOffset(layoutUnit);

    print("value: "); print(offset);
    print("unit: "); printLayoutUnit(layoutUnit);

    // ...
}
```

If an application knows what unit(s) it expects a variable to be laid out in, it can directly query those.
However, in a case like our systematic traversal of all shader parameters, it is not always possible to know what units a given variable uses.

The Slang reflection API can be used to query layout units used by a given variable layout with `getCategoryCount()` and `getCategoryByIndex()`:

```c++
void printRelativeOffsets(
    slang::VariableLayoutReflection* varLayout)
{
    print("relative offset: ");
    int usedLayoutUnitCount = varLayout->getCategoryCount();
    for (int i = 0; i < usedLayoutUnitCount; ++i)
    {
        auto layoutUnit = varLayout->getCategoryByIndex(i);
        printOffset(varLayout, layoutUnit);
    }
}
```

#### Spaces / Sets

For certain target platforms and layout units, the offset of a variable for that unit might include an additional dimension that represents a Vulkan/SPIR-V descriptor set, D3D12/DXIL register space, or a WebGPU/WGSL binding group.
In this chapter, we will uniformly refer to all of these concepts as *spaces*.

The relative space offset of a variable layout for a given layout unit can be queried with `getBindingSpace()`:

```c++
void printOffset(
    slang::VariableLayoutReflection* varLayout,
    slang::ParameterCategory layoutUnit)
{
    // ...

    size_t spaceOffset = varLayout->getBindingSpace(layoutUnit);

    switch(layoutUnit)
    {
    default:
        break;

    case slang::ParameterCategory::ConstantBuffer:
    case slang::ParameterCategory::ShaderResource:
    case slang::ParameterCategory::UnorderedAccess:
    case slang::ParameterCategory::SamplerState:
    case slang::ParameterCategory::DescriptorTableSlot:
        print("space: "); print(spaceOffset);    
    }
}
```

The code above only prints the space offset for the layout units where a space is semantically possible and meaningful.

### Type Layouts

A `TypeLayoutReflection` represents a layout computed for a type.
The underlying type that layout was computed for can be accessed using `TypeLayoutReflection::getType()`, but accessors are provided so that the most common properties of types can be queried on type layouts.

The main thing that a type layout stores is the size of the type:

```c++
void printTypeLayout(slang::TypeLayoutReflection* typeLayout)
{
    print("name: "); printQuotedString(typeLayout->getName());
    print("kind: "); printTypeKind(typeLayout->getKind());

    printSizes(typeLayout);

    // ...
}
```

#### Size

Similarly to variable layouts, the size of a type layout can be queried given a chosen layout unit:

```c++
void printSize(
    slang::TypeLayoutReflection* typeLayout,
    slang::ParameterCategory layoutUnit)
{
    size_t size = typeLayout->getSize(layoutUnit);

    key("value"); printPossiblyUnbounded(size);
    key("unit"); writeLayoutUnit(layoutUnit);
}
```

Note that the size of a type may be *unbounded* for a particular layout unit; this case is encoded just like the unbounded case for the element count of an array type (`~size_t(0)`).

The layout units used by a particular type layout can be iterated over using `getCategoryCount()` and `getCategoryByIndex()`:

```c++
void printSizes(slang::TypeLayoutReflection* typeLayout)
{
    print("size: ");
    int usedLayoutUnitCount = typeLayout->getCategoryCount();
    for (int i = 0; i < usedLayoutUnitCount; ++i)
    {
        auto layoutUnit = typeLayout->getCategoryByIndex(i);
        print("- "); printSize(typeLayout, layoutUnit);
    }

    // ...
}
```

#### Alignment and Stride

For any given layout unit, a type layout can also reflect the alignment of the type for that unit with `TypeLayoutReflection::getAlignment()`.
Alignment is typically only interesting when the layout unit is bytes (`slang::ParameterCategory::Uniform`).

Note that, unlike in C/C++, a type layout in Slang may have a size that is not a multiple of its alignment.
The *stride* of a type layout (for a given layout unit) is its size rounded up to its alignment, and is used as the distance between consecutive elements in arrays.
The stride of a type layout can be queried for any chosen layout unit with `TypeLayoutReflection::getStride()`.

Note that all of the `TypeLayoutReflection` methods `getSize()`, `getAlignment()`, and `getStride()` default to returning information in bytes, if a layout unit is not specified.
The same is true of the `VariableLayoutReflection::getOffset()` method.

The alignment and stride of a type layout can be reflected when it is relevant with code like:

```c++
void printTypeLayout(slang::TypeLayoutReflection* typeLayout)
{
    // ...

    if(typeLayout->getSize() != 0)
    {
        print("alignment in bytes: ");
        print(typeLayout->getAlignment());

        print("stride in bytes: ");
        print(typeLayout->getStride());
    }

    // ...
}
```

#### Kind-Specific Information

Just as with the underlying types, a type layout may store additional information depending on the kind of type:

```c++
void printTypeLayout(slang::TypeLayoutReflection* typeLayout)
{
    // ...

    switch(typeLayout->getKind())
    {
    default:
        break;
    
        // ...
    }
}
```

The following subsections will cover the important kinds to handle when reflecting type layouts.

#### Structure Type Layouts

A type layout for a `struct` type provides access to the fields of the `struct`, with each field represented as a variable layout:

```c++
case slang::TypeReflection::Kind::Struct:
    {
        print("fields: ");

        int fieldCount = typeLayout->getFieldCount();
        for (int f = 0; f < fieldCount; f++)
        {
            auto field = typeLayout->getFieldByIndex(f);
            printVarLayout(field);
        }
    }
    break;
```

The offset information stored on the type layout for each field will always be relative to the start of the `struct` type.

#### Array Type Layouts

Array type layouts store a layout for the element type of the array, which can be accessed with `getElementTypeLayout()`:

```c++
case slang::TypeReflection::Kind::Array:
    {
        print("element count: ");
        printPossiblyUnbounded(typeLayout->getElementCount());

        print("element type layout: ");
        printTypeLayout(typeLayout->getElementTypeLayout());
    }
    break;
```

#### Matrix Type Layouts

A layout for a matrix type stores a matrix layout *mode* (`SlangMatrixLayoutMode`) to record whether the type was laid out in row-major or column-major layout:

```c++
case slang::TypeReflection::Kind::Matrix:
    {
        // ...

        print("matrix layout mode: ");
        printMatrixLayoutMode(typeLayout->getMatrixLayoutMode());
    }
    break;
```

Note that the concepts of "row" and "column" as employed by Slang are the opposite of how Vulkan, SPIR-V, GLSL, and OpenGL use those terms.
When Slang reflects a matrix as using row-major layout, the corresponding matrix in generated SPIR-V will have a `ColMajor` decoration.
For an explanation of why these conventions differ, please see the relevant [appendix](./a1-01-matrix-layout.md).

#### Single-Element Containers

Constant buffers, parameter blocks, and other types representing grouping of parameters are the most subtle cases to handle for reflection.
The Slang reflection API aspires to provide complete and accurate information for these cases, but understanding *why* the provided data is what it is requires an appropriate mental model.

##### Simple Cases

In simple cases, a constant buffer has only ordinary data in it (things where the only used layout unit is bytes):

```
struct DirectionalLight
{
    float3 direction;
    float3 intensity;
}
ConstantBuffer<DirectionalLight> light;
```

When this case is laid out for D3D12, the `DirectionalLight` type will consume 28 bytes, but the `light` parameter will instead consume one `b` register.
We thus see that the `ConstantBuffer<>` type effectively "hides" the number of bytes used by its element.

Similarly, when a parameter block only has opaque types in it:

```
struct Material
{
    Texture2D albedoMap;
    Texture2D glossMap;
    SamplerState sampler;
}
ParameterBlock<Material> material;
```

When this is laid out for Vulkan, the `Material` type will consume 3 bindings, but the `material` parameter will instead consume one space.
A `ParameterBLock<>` type hides the bindings/registers/slots used by its element.

##### When Things Leak

If the element type of a constant buffer includes any data that isn't just measured in bytes, that usage will "leak" into the size of the constant buffer.
For example:

```
struct ViewParams
{
    float3 cameraPos;
    float3 cameraDir;
    TextureCube envMap;
}
ConstantBuffer<ViewParams> view;
```

If this example is laid out for D3D12, the `ViewParams` type will have a size of 28 bytes (according to D3D constant buffer layout rules) and one `t` register.
The size of the `view` parameter will be one `b` register and one `t` register.
The `ConstantBuffer<>` type can hide the bytes used by `ViewParams`, but the used `t` register leaks out and becomes part of the size of `view`.

If the same example is laid out for Vulkan, the `ViewParams` type will have a size of 28 bytes (according to `std140` layout rules) and one `binding`.
The size of the `view` parameter will be two `binding`s.

An important question a user might have in the Vulkan case, is whether the `binding` for `view` comes before that for `view.envMap`, or the other way around.
The answer is that the Slang compiler always lays out the "container" part of a parameter like `view` (the constant buffer) before the element, but a client of the reflection API shouldn't have to know such things to understand the information that gets reflected.

Note that in the Vulkan case, the offset of the `envMap` field within `ViewParams` is zero `binding`s, but the offset of `view.envMap` field relative to `view` is one `binding`.
Computing the cumulative offset of `view.envMap` requires more information than just that available on the variable layouts for `view` and `view.envMap`.

Similar cases of usage leaking can occur for parameter blocks, when one parameter block is nested within another.

##### A `ConstantBuffer<>` Without a Constant Buffer

While it is an uncommon case, it is possible to use a `ConstantBuffer<>` with an element type that contains no ordinary data (nothing with a layout unit of bytes):

```
struct Material
{
    Texture2D albedoMap;
    Texture2D glossMap;
    SamplerState sampler;
}
ConstantBuffer<Material> material;
```

If this case is compiled for Vulkan, the `material` parameter will consume 3 `binding`s, but none of those will be for a constant buffer.
In this case, unlike in the preceding example with `view.envMap`, the offset of `material.albedoMap` relative to `material` will be zero `binding`s.

##### Implicitly-Allocated Constant Buffers

A common use case for parameter blocks is to wrap up all of the parameters of a shader, or of some subsystem.
In such cases, there are likely to be both ordinary-type and opaque-type fields:

```
struct PointLight
{
    float3 position;
    float3 intensity;
}
struct LightingEnvironment
{
    TextureCube envMap;
    PointLight pointLights[10];
}
ParameterBlock<LightingEnvironment> lightEnv;
```

If this example is compiled for Vulkan, the `LightingEnvironment` type uses 316 bytes and one `binding` (ParameterCategory::DescriptorTableSlot), while `lightEnv` uses one descriptor `set`  (ParameterCategory::SubElementRegisterSpace).

What is not clear in the above description, however, is that because `LightingEnvironment` uses ordinary bytes, the Slang compiler will have to implicitly allocate a `binding` for a constant buffer to hold those bytes.
Conceptually, the layout is similar to what would be produced for `ParameterBlock<ConstantBuffer<LightingEnvironment>>`.

Furthermore, that constant buffer `binding` will be the first binding within the descriptor `set` for `lightEnv`, so that the cumulative `binding` offset for `lightEnv.envMap` will be one `binding` (even though `LightingEnvironment::envMap` has a relative offset of zero `binding`s).

##### Container and Element

In order to properly handle all of the nuances described here, the layout for a type like `ConstantBuffer<Thing>` or `ParameterBlock<Thing>` includes both layout information for the element of the container (a `Thing`) as well as layout information for the *container* itself.
Furthermore, the layout information for both the element and container need to support storing offset information (not just size), relative to the overall `ConstantBuffer<>` or `ParameterBlock<>`.

The breakdown is thus:

* The size information for the complete container type layout reflects whatever usage "leaks" out, such that it would need to be accounted for when further aggregating the overall type.

* Information about the allocated container is stored as a variable layout, queried with `getContainerVarLayout()`

  * The type layout for that variable layout shows what was allocated to represent the container itself, including any implicitly-allocated constant buffer

  * The offsets of that variable layout show where the container is situated relative to the overall type.
  With the current layout strategies used by the Slang compiler, all of these offsets will be zero.

* Information about the element is stored as a variable layout, queried with `getElementVarLayout()`

  * The type layout of that variable layout shows how the element type is laid out inside container.

  * The offsets on that variable layout show where the element is situated relative to the overall type.
  These offsets will be non-zero in cases where there is some layout unit used by both the element type and the container itself.

Given this understanding, we can now look at the logic to reflect a type layout for a constant buffer, parameter block, or similar type.

```c++
case slang::TypeReflection::Kind::ConstantBuffer:
case slang::TypeReflection::Kind::ParameterBlock:
case slang::TypeReflection::Kind::TextureBuffer:
case slang::TypeReflection::Kind::ShaderStorageBuffer:
    {
        print("container: ");
        printOffsets(typeLayout->getContainerVarLayout());
    
        auto elementVarLayout = typeLayout->getElementVarLayout();
        print("element: ");
        printOffsets(elementVarLayout);

        print("type layout: ");
        printTypeLayout(
            elementVarLayout->getTypeLayout();
    }
    break;
```

Note that the application logic here does not simply make use of `printVarLayout()` on the results of both `getContainerVarLayout()` and `getElementVarLayout()`, even though it technically could.
While these sub-parts of the overall type layout are each represented as a `VariableLayoutReflection`, many of the properties of those variable layouts are uninteresting or null; they primarily exist to convey offset information.

##### Example

Given input code like the following:

```hlsl
struct Material
{
    Texture2D albedoMap;
    SamplerState sampler;
    float2 uvScale;
    float2 uvBias;
}

struct FrameParams
{
    ConstantBuffer<Material> material;

    float3 cameraPos;
    float3 cameraDir;

    TextureCube envMap;
    float3 sunLightDir;
    float3 sunLightIntensity;

    Texture2D shadowMap;
    SamplerComparisonState shadowMapSampler;
}

ParameterBlock<FrameParams> params;
```

We will look at the kind of output our example application prints for `params` when compiling for Vulkan.
The basic information for the variable and its type layout looks like:

```
- name: "params"
  offset:
    relative:
    - value: 1
      unit: SubElementRegisterSpace # register spaces / descriptor sets
  type layout:
    name: "ParameterBlock"
    kind: ParameterBlock
    size:
      - value: 1
        unit: SubElementRegisterSpace # register spaces / descriptor sets
```

As we would expect, the size of the parameter block is one register space (aka Vulkan descriptor `set`).
In this case, the Slang compiler has assigned `params` to have a space offset of 1 (`set=1` in GLSL terms).

The offset information for the container part of `params` is the following:

```
container:
offset:
  relative:
    - value: 0
      unit: DescriptorTableSlot # bindings
      space: 0
    - value: 0
      unit: SubElementRegisterSpace # register spaces / descriptor sets
```

We can see from this information that the `ParameterBlock<>` container had two things allocated to it: a descriptor set (`ParameterCategory::SubElementRegisterSpace`), and a binding within that descriptor set (`ParameterCategory::DescriptorTableSlot`) for the automatically-introduced constant buffer.
That automatically-introduced buffer has an offset of 0 bindings from the start of the descriptor set.

The layout for the element part of the parameter block is as follows:

```
element:
  offset:
    relative:
      - value: 1
        unit: DescriptorTableSlot # bindings
        space: 0
      - value: 0
        unit: Uniform # bytes
  type layout:
    name: "FrameParams"
    kind: Struct
    size:
      - value: 6
        unit: DescriptorTableSlot # bindings
      - value: 64
        unit: Uniform # bytes
    alignment in bytes: 16
    stride in bytes: 64
    fields:                  
      - name: "material"
        offset:
          relative:
            - value: 0
              unit: DescriptorTableSlot # bindings
              space: 0
      ...
```

We see here that the type layout for the element is as expected of a layout for the `FrameParams` type.
In particular, note how the `material` field has a relative offset of zero bindings from the start of the `struct`, as is expected for the first field.
In order to account for the automatically-introduced constant buffer that is used by the container part of the layout, the element variable layout includes a relative offset of one binding (`ParameterCategory::DescriptorTableSlot`).

In a later section we will discuss how to easily sum up the various relative offsets shown in an example like this, when an application wants to compute a *cumulative* offset for a field like `params.material.sampler`.


##### Pitfalls to Avoid

It is a common mistake for users to apply `getElementTypeLayout()` on a single-element container, instead of using `getElementVarLayout()` as we advise here.
The implementation of the reflection API makes an effort to ensure that the type layout returned by `getElementTypeLayout()` automatically bakes in the additional offsets that are needed, but the results can still be unintuitive.

Programs and Scopes
-------------------

So far, our presentation has largely been bottom-up: we have shown how to recursively perform reflection on types, variables, and their layouts, but we have not yet shown how how to get this recursive traversal started.
We will now proceed top-down for a bit, and look at how to reflect the top-level parameters of a program.

A `ProgramLayout` is typically obtained using `IComponentType::getLayout()` after compiling and linking a Slang program.
A program layout primarily comprises the global scope, and zero or more entry points:

```c++
void printProgramLayout(
    slang::ProgramLayout* programLayout)
{
    print("global scope: ");
    printScope(programLayout->getGlobalParamsVarLayout());

    print("entry points: ");
    int entryPointCount = programLayout->getEntryPointCount();
    for (int i = 0; i < entryPointCount; ++i)
    {
        print("- ");
        printEntryPointLayout(
            programLayout->getEntryPointByIndex(i));
    }
}
```

The global scope and entry points are each an example of a *scope* where top-level shader parameters can be declared.
Scopes are represented in the reflection API using `VariableLayoutReflection`s.
We will now discuss the details of reflection for scopes, starting with the global scope as an example.

### Global Scope

In order to understand how the Slang reflection API exposes the global scope, it is valuable to think of the steps (some of them optional) that the Slang compiler applies to global-scope shader parameter declarations as part of compilation.

#### Parameters are Grouped Into a Structure

If a shader program declares global-scope parameters like the following:

```hlsl
Texture2D diffuseMap;
TextureCube envMap;
SamplerState sampler;
```

The Slang compiler will conceptually group all of those distinct global-scope parameter declarations into a `struct` type and then have only a single global-scope parameter of that type:

```hlsl
struct Globals
{
    Texture2D diffuseMap;
    TextureCube envMap;
    SamplerState sampler;
}
uniform Globals globals;
```

In this simple kind of case, the scope will be reflected as a variable layout with a `struct` type layout, with one field for each parameter declared in that scope:

```c++
void printScope(
    slang::VariableLayoutReflection*    scopeVarLayout)
{
    auto scopeTypeLayout = scopeVarLayout->getTypeLayout();
    switch (scopeTypeLayout->getKind())
    {
    case slang::TypeReflection::Kind::Struct:
        {
            print("parameters: ");

            int paramCount = scopeTypeLayout->getFieldCount();
            for (int i = 0; i < paramCount; i++)
            {
                print("- ");

                auto param = scopeTypeLayout->getFieldByIndex(i);
                printVarLayout(param, &scopeOffsets);
            }
        }
        break;

        // ...
    }
}
```

#### Wrapped in a Constant Buffer If Needed

In existing shader code that was originally authored for older APIs (such as D3D9) it is common to find a mixture of opaque and ordinary types appearing as global-scope shader parameters:

```hlsl
Texture2D diffuseMap;
TextureCube envMap;
SamplerState sampler;

uniform float3 cameraPos;
uniform float3 cameraDir;
```

In these cases, when the Slang compiler groups the parameters into a single `struct`:

```hlsl
struct Globals
{
    Texture2D diffuseMap;
    TextureCube envMap;
    SamplerState sampler;

    float3 cameraPos;
    float3 cameraDir;
}
```

it finds that the resulting `struct` consumes a non-zero number of bytes and, for most compilation targets, it will automatically wrap that structure in a `ConstantBuffer<>` before declaring the single shader parameter that represents the global scope:

```hlsl
ConstantBuffer<Globals> globals
```

This case shows up in the Slang reflection API as the scope having a type layout with the constant-buffer kind:

```c++
case slang::TypeReflection::Kind::ConstantBuffer:
    print("automatically-introduced constant buffer: ");

    printOffsets(scopeTypeLayout->getContainerVarLayout());

    printScope(scopeTypeLayout->getElementVarLayout());
    break;
```

In this case, the container variable layout reflects the relative offsets for where the automatically-introduced constant buffer is bound, and the element variable layout reflects the global scope parameters that were wrapped in this way.

#### Wrapped in a Parameter Block If Needed

For targets like D3D12/DXIL, Vulkan/SPIR-V, and WebGPU/WGSL, most shader parameters must be bound via the target-specific grouping mechanism (descriptor tables, descriptor sets, or binding groups, respectively).
If the Slang compiler is compiling for such a target and detects that there are global-scope parameters that do not specify an explicit space, then it will conceptually wrap the global-scope declarations in a `ParameterBlock<>` that provides a default space.

For example, if compiling this code to Vulkan:

```hlsl
Texture2D diffuseMap;
[[vk::binding(1,0)]] TextureCube envMap;
SamplerState sampler;
```

the Slang compiler will detect that `envMap` is explicitly bound to `binding` 1 in space (aka descriptor `set`) 0, and that neither `diffuseMap` nor `sampler` has been explicitly bound.
Both of the unbound parameters need to be passed inside of some space, so the compiler will allocate space 1 for that purpose (as space 0 was already claimed by explicit bindings).
In simplistic terms, the compiler will behave *as if* the global-scope parameters are wrapped up in a `struct` and then further wrapped up into a `ParameterBlock<>`.

This case shows up in the Slang reflection API as the scope having a type layout with the parameter-block kind:

```c++
case slang::TypeReflection::Kind::ParameterBlock:
    print("automatically-introduced parameter block: ");

    printOffsets(scopeTypeLayout->getContainerVarLayout());

    printScope(scopeTypeLayout->getElementVarLayout());
    break;
```

In cases where the parameters in a scope require *both* a constant buffer and a parameter block to be automatically introduced, the scope is reflected as if things were wrapped with `ParameterBlock<...>` and not `ParameterBlock<ConstantBuffer<...>>`.
That is, the binding information for the implicit constant buffer will be found as part of the container variable layout for the parameter block.

#### Pitfalls to Avoid

The `ProgramLayout` type has the appealingly-named `getParameterCount` and `getParameterByIndex()` methods, which seem to be the obvious way to navigate the global-scope parameters of a shader.
However, we recommend *against* using these functions in applications that want to be able to systematically and robustly reflect any possible input shader code.

While the reflection API implementation makes an effort to ensure that the information returned by `getParameterByIndex()` is not incorrect, it is very difficult when using those functions to account for how global-scope parameters might have been grouped into an automatically-introduced constant buffer or parameter block.
The `getGlobalConstantBufferBinding()` and `getGlobalConstantBufferSize()` methods can be used in some scenarios, but aren't the best way to get the relevant information.

While it would only matter in corner cases, we still recommend that applications use `getGlobalParamsVarLayout()` instead of `getGlobalParamsTypeLayout()`, to account for cases where the global-scope might have offsets applied to it (and also to handle the global scope and entry-point scopes more uniformly).

### Entry Points

An `EntryPointReflection` provides information on an entry point.
This includes the stage that the entry point was compiled for:

```c++
void printEntryPointLayout(slang::EntryPointReflection* entryPointLayout)
{
    print("stage: "); printStage(entryPointLayout->getStage());

    // ...
}
```

#### Entry Point Parameters

An entry point acts as a scope for top-level shader parameters, much like the global scope.
Entry-point parameters are grouped into a `struct`, and then automatically wrapped in a constant buffer or parameter block if needed.
The main additional consideration, compared to the global scope, is that an entry-point function may also declare a result type.
When present, the function result acts more or less as an additional `out` parameter.

The parameter scope and result of an entry point can be reflected with logic like:

```c++
void printEntryPointLayout(slang::EntryPointReflection* entryPointLayout)
{
    // ...
    printScope(entryPointLayout->getVarLayout());

    auto resultVarLayout = entryPointLayout->getResultVarLayout();
    if (resultVarLayout->getTypeLayout()->getKind() != slang::TypeReflection::Kind::None)
    {
        key("result"); printVarLayout(resultVarLayout);
    }
}
```

##### Pitfalls to Avoid

Similarly to the case for the global scope, we recommend against using the `getParameterCount()` and `getParameterByIndex()` methods on `EntryPointReflection`, since they make it harder to handle cases where the entry-point scope might have been allocated as a constant buffer (although the `hasDefaultConstantBuffer()` method is provided to try to support older applications that still use `getParameterByIndex()`).
Applications are also recommended to use `EntryPointReflection::getVarLayout()` instead of `::getTypeLayout()`, to more properly reflect the way that offsets are computed and applied to the parameters of an entry point.

#### Stage-Specific Information

Depending on the stage that an entry point was compiled for, it may provide additional information that an application can query:

```c++
void printEntryPointLayout(slang::EntryPointReflection* entryPointLayout)
{
    // ...
    switch (entryPointLayout->getStage())
    {
    default:
        break;

        // ...
    }
    // ...
}
```

For example, compute entry points store the thread-group dimensions:

```c++
case SLANG_STAGE_COMPUTE:
    {
        SlangUInt sizes[3];
        entryPointLayout->getComputeThreadGroupSize(3, sizes);

        print("thread group size: ");
        print("x: "); print(sizes[0]);
        print("y: "); print(sizes[1]);
        print("z: "); print(sizes[2]);
    }
    break;
```

#### Varying Parameters

So far we have primarily been talking about the *uniform* shader parameters of a program: those that can be passed in from application code to shader code.
Slang's reflection API also reflects the *varying* shader parameters that appear are passed between stages of a pipeline.

Variable and type layouts for varying shader parameters will typically show usage of:

* Varying input slots (`slang::ParameterCategory::VaryingInput`) for stage inputs
* Varying output slots (`slang::ParameterCategory::VaryingOutput`) for `out` parameters and the entry-point result
* Both (`slang::ParameterCategory::VaryingInput` *and* `::VaryingOutput`) for `inout` parameters
* Nothing (no usage for any unit) for *system value* parameters (typically using an `SV_*` semantic)

For user-defined varying parameters, some GPU APIs care about the *semantic* that has been applied to the parameter.
For example, given this shader code:

```hlsl
[shader("vertex")]
float4 vertexMain(
    float3 position : POSITION,
    float3 normal : NORMAL,
    float3 uv : TEXCOORD,
    // ...
    )
    : SV_Position
{
    // ...
}
```

the shader parameter `normal` of `vertexMain` has a semantic of `NORMAL`.

Semantics are only relevant for shader parameters that became part of the varying input/output interface of an entry point for some stage, in which case the `VariableLayoutReflection::getStage()` method will return that stage.
A semantic is decomposed into both a name and an index (e.g., `TEXCOORD5` has a name of `"TEXCOORD"` and an index of `5`).
This information can be reflected with `getSemanticName()` and `getSemanticIndex()`:


```c++
void printVarLayout(slang::VariableLayoutReflection* varLayout)
{
    // ...
    if (varLayout->getStage() != SLANG_STAGE_NONE)
    {
        print("semantic: ");
        print("name: "); printQuotedString(varLayout->getSemanticName());
        print("index: "); print(varLayout->getSemanticIndex());
    }
    // ...
}
```

Calculating Cumulative Offsets
------------------------------

All of the code so far has only extracted the *relative* offsets of variable layouts.
Offsets for fields have been relative to the `struct` that contains them.
Offsets for top-level parameters have been relative to the scope that contains them, or even to a constant buffer or parameter block that was introduced for that scope.

There are many cases where an application needs to calculate a *cumulative* offset (or even an absolute offset) for some parameter, even down to the granularity of individual `struct` fields.
As a notable example, allocation of D3D root signatures and Vulkan pipeline layouts for a program requires being able to enumerate the absolute offsets of all bindings in all descriptor tables/sets.

Because offsets for certain layout units include an additional dimension for a space, our example application will define a simple `struct` to represent a cumulative offset:

```c++
struct CumulativeOffset
{
    int value; // the actual offset
    int space; // the associated space
};
```

### Access Paths

There are multiple ways to track and calculate cumulative offsets.
Here we will present a solution that is both simple and reasonably efficient, while still yielding correct results even in complicated scenarios.

If all we had to do was calculate the byte offsets of things, a single `size_t` would be enough to represent a cumulative offset.
However, we have already seen that in the context of a GPU language like Slang, we can have offsets measured in multiple different layout units.
A naive implementation might try to represent a cumulative offset as a vector or dictionary of scalar offsets, with (up to) one for each layout unit.
The sheer number of layout units (the cases of the `slang::ParameterCategory` enumeration) makes such an approach unwieldy.

Instead we focus on the intuition that the cumulative offset of a variable layout, for any given layout unit, can be computed by summing up all the relative offsets along the *access path* to that variable.
For example, given code like:

```hlsl
struct Material
{
    Texture2D albedoMap;
    Texture2D glossMap;
    SamplerState sampler;
}
struct LightingEnvironment
{
    TextureCube environmentMap;
    float3 sunLightDir;
    float3 sunLightIntensity;
}
struct Params
{
    LightingEnvironment lights;
    Material material;
}
uniform Params params;
```

we expect that the cumulative offset of `params.material.glossMap` in units of Vulkan `binding`s can be computed by summing up the offsets in that unit of `params` (0), `material` (1), and `glossMap` (1).

When recursively traversing the parameters of a shader, out example application will track an access path as a singly-linked list of variable layouts that points up the stack, from the deepest variable to the shallowest:

```c++
struct AccessPathNode
{
    slang::VariableLayoutReflection* varLayout;
    AccessPathNode* outer;
};

struct AccessPath
{
    AccessPathNode* leafNode = nullptr;
};
```

For the example code above, if our recursive traversal is at `params.material.glossMap`, then the access path will start with a node for `glossMap` which points to a node for `material`, which points to a node for `glossMap`.

For many layout units, we can calculate a cumulative offset simply by summing up contributions along the entire access path, with logic like the following:

```c++
CumulativeOffset calculateCumulativeOffset(slang::ParameterCategory layoutUnit, AccessPath accessPath)
{
    // ...
    for(auto node = accessPath.leafNode; node != nullptr; node = node->outer)
    {
        result.value += node->varLayout->getOffset(layoutUnit);
        result.space += node->varLayout->getBindingSpace(layoutUnit);
    }
    // ...
}
```

Once our example application is properly tracking access paths, we will be able to use them to calculate and print the cumulative offsets of variable layouts:

```c++
void printOffsets(
    slang::VariableLayoutReflection* varLayout,
    AccessPath accessPath)
{
    // ...

    print("cumulative:");
    for (int i = 0; i < usedLayoutUnitCount; ++i)
    {
        print("- ");
        auto layoutUnit = varLayout->getCategoryByIndex(i);
        printCumulativeOffset(varLayout, layoutUnit, accessPath);
    }
}
```

Printing the cumulative offset of a variable layout requires adding the offset information for the variable itself to the offset calculated from its access path:

```c++
void printCumulativeOffset(
    slang::VariableLayoutReflection* varLayout,
    slang::ParameterCategory layoutUnit,
    AccessPath accessPath)
{
    CumulativeOffset cumulativeOffset = calculateCumulativeOffset(layoutUnit, accessPath);

    cumulativeOffset.offset += varLayout->getOffset(layoutUnit);
    cumulativeOffset.space += varLayout->getBindingSpace(layoutUnit);

    printOffset(layoutUnit, cumulativeOffset.offset, cumulativeOffset.space);
}
```

### Tracking Access Paths

In order to support calculation of cumulative offsets, the various functions we've presented so far like `printVarLayout()` and `printTypeLayout()` need to be extended with an additional parameter for an `AccessPath`.
For example, the signature of `printTypeLayout()` becomes:

```c++
void printTypeLayout(slang::TypeLayoutReflection* typeLayout, AccessPath accessPath)
{
    // ...
}
```

#### Variable Layouts

When traversing a variable layout, we then need to extend the access path to include the additional variable layout, before traversing down into its type layout:

```c++
void printVarLayout(slang::VariableLayoutReflection* typeLayout, AccessPath accessPath)
{
    // ...

    ExtendedAccessPath varAccessPath(accessPath, varLayout);

    print("type layout: ");
    printTypeLayout(varLayout->getTypeLayout(), varAccessPath);
}
```

#### Scopes

Similar logic is needed within `printScope()` in our example program:

```c++
void printScope(
    slang::VariableLayoutReflection* scopeVarLayout,
    AccessPath                       accessPath)
{
    ExtendedAccessPath scopeAccessPath(accessPath, scopeVarLayout);

    // ...
}
```

The calls to `printOffsets()`, `printTypeLayout()`, etc. inside of `printScope()` will then pass along the extended access path.

#### Array-Like Types

When the traversing an array, matrix, or vector type, it is impossible to compute a single cumulative offset that is applicable to all elements of the type.
The recursive calls to `printTypeLayout()` in these cases will simply pass in an empty `AccessPath`.
For example:

```c++
case slang::TypeReflection::Kind::Array:
    {
        // ...

        print("element type layout: ");
        printTypeLayout(
            typeLayout->getElementTypeLayout(),
            AccessPath());
    }
    break;
```

### Handling Single-Element Containers

Types like constant buffers and parameter blocks add complexity that requires additions to our representation and handling of access paths.

First, when calculating the cumulative byte offset of variables inside a constant buffer (or any of these single-element container types), it is important not to sum contributions too far up the access path.
Consider this example:

```c++
struct A
{
    float4 x;
    Texture2D t;
}
struct B
{
    float4 y;
    ConstantBuffer<Inner> a;
}
struct C
{
    float4 z;
    Texture2D t;
    B b;
}
uniform C c;
```

When compiling for D3D12, the cumulative byte offset of `c.b` is 16, but the cumulative byte offset of `c.b.a.x` needs to be zero, because its byte offset should be measured relative to the enclosing constant buffer `c.b.a`.
In contrast, the cumulative of offset of `c.b` in `t` registers is one, and the cumulative offset of `c.b.a.t` needs to be two.

Similarly, when calculating the cumulative offsets of variables inside a parameter block (for targets that can allocate each parameter block its own space), it is important not to sum contributions past an enclosing parameter block.

We can account for these subtleties by extending the representation of access paths in our example application to record the node corresponding to the deepest constant buffer or parameter block along the path:

```c++
struct AccessPath
{
    AccessPathNode* leaf = nullptr;
    AccessPathNode* deepestConstantBuffer = nullptr;
    AccessPathNode* deepestParameterBlock = nullptr;
};
```

Now when traversing a single-element container type in `printTypeLayout`, we can make a copy of the current access path and modify its `deepestConstantBuffer` to account for the container:

```c++
case slang::TypeReflection::Kind::ConstantBuffer:
case slang::TypeReflection::Kind::ParameterBlock:
case slang::TypeReflection::Kind::TextureBuffer:
case slang::TypeReflection::Kind::ShaderStorageBuffer:
    {
        // ...

        AccumulatedOffsets innerAccessPath = accessPath;
        innerAccessPath.deepestConstantBuffer = innerAccessPath.leaf;

        // ...
    }
    break;
```

Further, if the container had a full space allocated to it, then we also update the `deepestParameterBlock`:

```c++
// ...
if (containerVarLayout->getTypeLayout()->getSize(
    slang::ParameterCategory::SubElementRegisterSpace) != 0)
{
    innerAccessPath.deepestParameterBlock = innerAccessPath.leaf;
}
// ...
```

Finally, when traversing the element of the container, we need to use this new `innerAccessPath`, and also extend the access path when traversing into the type layout of the element:

```c++
print("element: ");
printOffsets(elementVarLayout, innerAccessPath);

ExtendedAccessPath elementAccessPath(innerAccessPath, elementVarLayout);

print("type layout: ");
printTypeLayout(
    elementVarLayout->getTypeLayout(),
    elementAccessPath);
```

### Accumulating Offsets Along An Access Path

We now understand that the proper way to calculate a cumulative offset depends on the layout unit:

```c++
CumulativeOffset calculateCumulativeOffset(
    slang::ParameterCategory layoutUnit,
    AccessPath               accessPath)
{
    switch(layoutUnit)
    {
    // ...
    }
}
```

#### Layout Units That Don't Require Special Handling

By default, relative offsets will be summed for all nodes along the access path:

```c++
default:
    for (auto node = accessPath.leaf; node != nullptr; node = node->outer)
    {
        result.offset += node->varLayout->getOffset(layoutUnit);
    }
    break;
```

#### Bytes

When a byte offset is being computed, relative offsets will only be summed up to the deepest enclosing constant buffer, if any:

```c++
case slang::ParameterCategory::Uniform:
    for (auto node = accessPath.leaf; node != accessPath.deepestConstantBuffer; node = node->outer)
    {
        result.offset += node->varLayout->getOffset(layoutUnit);
    }
    break;
```

#### Layout Units That Care About Spaces

Finally, we need to handle the layout units that care about spaces:

```c++
case slang::ParameterCategory::ConstantBuffer:
case slang::ParameterCategory::ShaderResource:
case slang::ParameterCategory::UnorderedAccess:
case slang::ParameterCategory::SamplerState:
case slang::ParameterCategory::DescriptorTableSlot:
    // ...
    break;
```

Relative offsets, including space offsets, need to be summed along the access path up to the deepest enclosing parameter block, if any:

```c++
for (auto node = accessPath.leaf; node != accessPath.deepestParameterBlock; node = node->outer)
{
    result.offset += node->varLayout->getOffset(layoutUnit);
    result.space += node->varLayout->getBindingSpace(layoutUnit);
}
```

Additionally, the offset of the enclosing parameter block in spaces needs to be added to the space of the cumulative offset:

```c++
for (auto node = accessPath.deepestParameterBlock; node != nullptr; node = node->outer)
{
    result.space += node->varLayout->getOffset(slang::ParameterCategory::SubElementRegisterSpace);
}
```

Determining Whether Parameters Are Used
---------------------------------------

Some application architectures make use of shader code that declares a large number of shader parameters at global scope, but only uses a small fraction of those parameters at runtime.
Similarly, shader parameters may be declared at global scope even if they are only used by a single entry point in a pipeline.
These kinds of architectures are not ideal, but they are pervasive.

Slang's base reflection API *intentionally* does not provide information about which shader parameters are or are not used by a program, or specific entry points.
This choice ensures that applications using the reflection API can robustly re-use data structures built from reflection data across hot reloads of shaders, or switches between variants of a program.

Applications that need to know which parameters are used (and by which entry points or stages) need to query for additional metadata connected to the entry points of their compiled program using `IComponentType::getEntryPointMetadata()`:

```c++
slang::IComponentType* program = ...;
slang::IMetadata* entryPointMetadata;
program->getEntryPointMetadata(
        entryPointIndex,
        0, // target index
        &entryPointMetadata);
```

When traversal of reflection data reaches a leaf parameter, the application can use `IMetadata::isParameterLocationUsed()` with the absolute location of that parameter for a given layout unit:

```c++
unsigned calculateParameterStageMask(
    slang::ParameterCategory layoutUnit,
    CumulativeOffset offset)
{
    unsigned mask = 0;
    for(int i = 0; i < entryPointCount; ++i)
    {
        bool isUsed = false;
        entryPoints[i].metadata->isParameterLocationUsed(
            layoutUnit, offset.space, offset.value, isUsed);
        if(isUsed)
        {
            mask |= 1 << unsigned(entryPoints[i].stage);
        }
    }
    return mask;
}
```

The application can then incorporate this logic into a loop over the layout units consumed by a parameter:

```c++
unsigned calculateParameterStageMask(
    slang::VariableLayoutReflection* varLayout,
    AccessPath accessPath)
{
    unsigned mask = 0;

    int usedLayoutUnitCount = varLayout->getCategoryCount();
    for (int i = 0; i < usedLayoutUnitCount; ++i)
    {
        auto layoutUnit = varLayout->getCategoryByIndex(i);
        auto offset = calculateCumulativeOffset(
            varLayout, layoutUnit, accessPath);
        
        mask |= calculateStageMask(
            layoutUnit, offset);
    }

    return mask;
}
```

Finally, we can wrap all this up into logic to print which stage(s) use a given parameter, based on the information in the per-entry-point metadata:

```c++
void printVarLayout(
    slang::VariableLayoutReflection* varLayout,
    AccessPath accessPath)
{
    //...
    unsigned stageMask = calculateStageMask(
        varLayout, accessPath);

    print("used by stages: ");
    for(int i = 0; i < SLANG_STAGE_COUNT; i++)
    {
        if(stageMask & (1 << i))
        {
            print("- ");
            printStage(SlangStage(i));
        }
    }
    // ...
}
```

Conclusion
----------

At this point we have provided a comprehensive example of how to robustly traverse the information provided by the Slang reflection API to get a complete picture of the shader parameters of a program, and what target-specific locations they were bound to.
We hope that along the way we have also imparted some key parts of the mental model that exists behind the reflection API and its representations.
