> Note: This document is a work in progress. It is both incomplete and, in many cases, inaccurate.

Types
=====

This section defines the kinds of types supported by Slang.

Types in Slang do not necessarily prescribe a single _layout_ in memory.
The discussion of each type will specify any guarantees about layout it provides; any details of layout not specified here may depend on the target platform, compiler options, and context in which a type is used.

Void Type
---------

The type `void` contains no data and has a single, unnamed, value.

A `void` value takes up no space, and thus does not affect the layout of types.
Formally, a `void` value behaves as if it has a size of zero bytes, and one-byte alignment.

Scalar Types
------------

### Boolean Type

The type `bool` is used to represent Boolean truth values: `true` and `false`.

The size of a `bool` varies across target platforms; programs that need to ensure a matching in-memory layout between targets should not use `bool` for in-memory data structures.
On all platforms, the `bool` type must be _naturally aligned_ (its alignment is its size).

### Integer Types

The following integer types are defined:

| Name          | Description |
|---------------|-------------|
| `int8_t`      | 8-bit signed integer |
| `int16_t`     | 16-bit signed integer |
| `int`         | 32-bit signed integer |
| `int64_t`     | 64-bit signed integer |
| `uint8_t`     | 8-bit unsigned integer |
| `uint16_t`    | 16-bit unsigned integer |
| `uint`        | 32-bit unsigned integer |
| `uint64_t`    | 64-bit unsigned integer |

All signed integers used two's complement representation.
All arithmetic operations on integers (both signed and unsigned) wrap on overflow/underflow.

All target platforms must support the `int` and `uint` types.
Specific [target platforms](../target-compatibility.md) may not support the other integer types.

All integer types are stored in memory with their natural size and alignment on all targets that support them.

### Floating-Point Types

The following floating-point type are defined:

| Name          | Description                   |
|---------------|-------------------------------|
| `half`        | 16-bit floating-point number (1 sign bit, 5 exponent bits, 10 fraction bits) |
| `float`       | 32-bit floating-point number (1 sign bit, 8 exponent bits, 23 fraction bits) |
| `double`      | 64-bit floating-point number (1 sign bit, 11 exponent bits, 52 fraction bits) |

All floating-point types are laid out in memory using the matching IEEE 754 standard format (`binary16`, `binary32`, `binary64`).
Target platforms may define their own rules for rounding, precision, denormals, infinities, and not-a-number values.

All target platforms must support the `float` type.
Specific [targets](../target-compatibility.md) may not support the other floating-point types.

All floating-point types are stored in memory with their natural size and alignment on all targets that support them.

Vector Types
------------

A vector type is written as `vector<T, N>` and represents an `N`-element vector with elements of type `T`.
The _element type_ `T` must be one of the built-in scalar types, and the _element count_ `N` must be a specialization-time constant integer.
The element count must be between 2 and 4, inclusive.

A vector type allows subscripting of its elements like an array, but also supports element-wise arithmetic on its elements.
_Element-wise arithmetic_ means mapping unary and binary operators over the elements of a vector to produce a vector of results:

```hlsl
vector<int,4> a = { 1, 2, 30, 40 };
vector<int,4> b = { 10, 20, 3, 4 };

-a; // yields { -1, -2, -30, -40 }
a + b; // yields { 11, 22, 33, 44 }
b / a; // yields { 10, 10, 0, 0 }
a > b; // yields { false, false, true, true }
```

A vector type is laid out in memory as `N` contiguous values of type `T` with no padding.
The alignment of a vector type may vary by target platforms.
The alignment of `vector<T,N>` will be at least the alignment of `T` and may be at most `N` times the alignment of `T`.

As a convenience, Slang defines built-in type aliases for vectors of the built-in scalar types.
E.g., declarations equivalent to the following are provided by the Slang core module:

```hlsl
typealias float4 = vector<float, 4>;
typealias int8_t3 = vector<int8_t, 3>;
```

### Legacy Syntax

For compatibility with older codebases, the generic `vector` type includes default values for `T` and `N`, being declared as:

```hlsl
struct vector<T = float, let N : int = 4> { ... }
```

This means that the bare name `vector` may be used as a type equivalent to `float4`:

```hlsl
// All of these variables have the same type
vector a;
float4 b;
vector<float> c;
vector<float, 4> d;
```

Matrix Types
------------

A matrix type is written as `matrix<T, R, C>` and represents a matrix of `R` rows and `C` columns, with elements of type `T`.
The element type `T` must be one of the built-in scalar types.
The _row count_ `R` and _column count_ `C` must be specialization-time constant integers.
The row count and column count must each be between 2 and 4, respectively.

A matrix type allows subscripting of its rows, similar to an `R`-element array of `vector<T,C>` elements.
A matrix type also supports element-wise arithmetic.

Matrix types support both _row-major_ and _column-major_ memory layout.
Implementations may support command-line flags or API options to control the default layout to use for matrices.

> Note: Slang currently does *not* support the HLSL `row_major` and `column_major` modifiers to set the layout used for specific declarations.

Under row-major layout, a matrix is laid out in memory equivalently to an `R`-element array of `vector<T,C>` elements.

Under column-major layout, a matrix is laid out in memory equivalent to the row-major layout of its transpose.
This means it will be laid out equivalently to a `C`-element array of `vector<T,R>` elements.

As a convenience, Slang defines built-in type aliases for matrices of the built-in scalar types.
E.g., declarations equivalent to the following are provided by the Slang core module:

```hlsl
typealias float3x4 = matrix<float, 3, 4>;
typealias int64_t4x2 = matrix<int64_t, 4, 2>;
```

> Note: For programmers using OpenGL or Vulkan as their graphics API, and/or who are used to the GLSL language,
> it is important to recognize that the equivalent of a GLSL `mat3x4` is a Slang `float3x4`.
> This is despite the fact that GLSL defines a `mat3x4` as having 3 *columns* and 4 *rows*, while a Slang `float3x4` is defined as having 3 rows and 4 columns.
> This convention means that wherever Slang refers to "rows" or "columns" of a matrix, the equivalent terms in the GLSL, SPIR-V, OpenGL, and Vulkan specifications are "column" and "row" respectively (*including* in the compound terms of "row-major" and "column-major")
> While it may seem that this choice of convention is confusing, it is necessary to ensure that subscripting with `[]` can be efficiently implemented on all target platforms.
> This decision in the Slang language is consistent with the compilation of HLSL to SPIR-V performed by other compilers.


### Matrix Operations

Matrix types support several operations:

* Element-wise operations (addition, subtraction, multiplication) using the standard operators (`+`, `-`, `*`). These operations require matrices of the same dimensions.
* Algebraic matrix-matrix multiplication using the `mul()` function, which supports matrices of compatible dimensions (e.g., `float2x3 * float3x4`).
* Matrix-vector multiplication using `mul()`, where the vector can be interpreted as either a row or column vector depending on the parameter order:
  * `mul(v, m)` - v is interpreted as a row vector
  * `mul(m, v)` - v is interpreted as a column vector

For proper matrix multiplication, always use the `mul()` function. The `operator*` performs element-wise multiplication and should only be used when you want to multiply corresponding elements of same-sized matrices.

> Note: This differs from GLSL, where the `*` operator performs matrix multiplication. When porting code from GLSL or CUDA to Slang, you'll need to replace matrix multiplications using `*` with calls to `mul()`.

### Legacy Syntax

For compatibility with older codebases, the generic `matrix` type includes default values for `T`, `R`, and `C`, being declared as:

```hlsl
struct matrix<T = float, let R : int = 4, let C : int = 4> { ... }
```

This means that the bare name `matrix` may be used as a type equivalent to `float4x4`:

```hlsl
// All of these variables have the same type
matrix a;
float4x4 b;
matrix<float, 4, 4> c;
```

Structure Types
---------------

Structure types are introduced with `struct` declarations, and consist of an ordered sequence of named and typed fields:

```hlsl
struct S
{
    float2 f;
    int3 i;
}
```

### Standard Layout

The _standard layout_ for a structure type uses the following algorithm:

* Initialize variables `size` and `alignment` to zero and one, respectively
* For each field `f` of the structure type:
  * Update `alignment` to be the maximum of `alignment` and the alignment of `f`
  * Set `size` to the smallest multiple of `alignment` not less than `size`
  * Set the offset of field `f` to `size`
  * Add the size of `f` to `size`

When this algorithm completes, `size` and `alignment` will be the size and alignment of the structure type.

Most target platforms do not use the standard layout directly, but it provides a baseline for defining other layout algorithms.
Any layout for structure types must guarantee an alignment at least as large as the standard layout.

### C-Style Layout

C-style layout for structure types differs from standard layout by adding an additional final step:

* Set `size` the smallest multiple of `alignment` not less than `size`

This mirrors the layout rules used by typical C/C++ compilers.

### D3D Constant Buffer Layout

D3D constant buffer layout is similar to standard layout with two differences:

* The initial alignment is 16 instead of one

* If a field would have _improper straddle_, where the interval `(fieldOffset, fieldOffset+fieldSize)` (exclusive on both sides) contains any multiple of 16, *and* the field offset is not already a multiple of 16, then the offset of the field is adjusted to the next multiple of 16

Array Types
-----------

An _array type_ is either a statically-sized or dynamically-sized array type.

A known-size array type is written `T[N]` where `T` is a type and `N` is a specialization-time constant integer.
This type represents an array of exactly `N` values of type `T`.

An unknown-size array type is written `T[]` where `T` is a type.
This type represents an array of some fixed, but statically unknown, size.

> Note: Unlike in C and C++, arrays in Slang are always value types, meaning that assignment and parameter passing of arrays copies their elements.

### Declaration Syntax

For variable and parameter declarations using traditional syntax, a variable of array type may be declared by using the element type `T` as a type specifier (before the variable name) and the `[N]` to specify the element count after the variable name:

```hlsl
int a[10];
```

Alternatively, the array type itself may be used as the type specifier:

```hlsl
int[10] a;
```

When using the `var` or `let` keyword to declare a variable, the array type must not be split:

```hlsl
var a : int[10];
```

> Note: when declaring arrays of arrays (often thought of as "multidimensional arrays") a programmer must be careful about the difference between the two declaration syntaxes.
> The following two declarations are equivalent:
>
> ```hlsl
> int[3][5] a;
> int a[5][3];
> ```
>
> In each case, `a` is a five-element array of three-element arrays of `int`s.
> However, one declaration orders the element counts as `[3][5]` and the other as `[5][3]`.

### Element Count Inference

When a variable is declared with an unknown-size array type, and also includes an initial-value expression:

```hlsl
int a[] = { 0xA, 0xB, 0xC, 0xD };
```

The compiler will attempt to infer an element count based on the type and/or structure of the initial-value expression.
In the above case, the compiler will infer an element count of 4 from the structure of the initializer-list expression.
Thus the preceding declaration is equivalent to:

```hlsl
int a[4] = { 0xA, 0xB, 0xC, 0xD };
```

A variable declared in this fashion semantically has a known-size array type and not an unknown-size array type; the use of an unknown-size array type for the declaration is just a convenience feature.

### Standard Layout

The _stride_ of a type is the smallest multiple of its alignment not less than its size.

Using the standard layout for an array type `T[]` or `T[N]`:

* The _element stride_ of the array type is the stride of its element type `T`
* Element `i` of the array starts at an offset that is `i` times the element stride of the array
* The alignment of the array type is the alignment of `T`
* The size of an unknown-size array type is unknown
* The size of a known-size array with zero elements is zero
* The size of a known-size array with a nonzero number `N` of elements is the size of `T` plus `N - 1` times the element stride of the array

### C-Style Layout

The C-style layout of an array type differs from the standard layout in that the size of a known-size array with a nonzero number `N` of elements is `N` times the element stride of the array.

### D3D Constant Buffer Layout

The D3D constant buffer layout of an array differs from the standard layout in that the element stride of the array is set to the smallest multiple of the alignment of `T` that is not less than the stride of `T`

This Type
---------

Within the body of a structure or interface declaration, the keyword `This` may be used to refer to the enclosing type.
Inside of a structure type declaration, `This` refers to the structure type itself.
Inside of an interface declaration, `This` refers to the concrete type that is conforming to the interface (that is, the type of `this`).

Opaque Types
------------

_Opaque_ types are built-in types that (depending on the target platform) may not have a well-defined size or representation in memory.
Similar languages may refer to these as "resource types" or "object types."

The full list of opaque types supported by Slang can be found in the core module reference, but important examples are:

* Texture types such as `Texture2D<T>`, `TextureCubeArray<T>`, and `RWTexture2DMS<T>`
* Sampler state types: `SamplerState` and `SamplerComparisonState`
* Buffer types like `ConstantBuffer<T>` and  `StructuredBuffer<T>`
* Parameter blocks: `ParameterBlock<T>`

Layout for opaque types depends on the target platform, and no specific guarantees can be made about layout rules across platforms.

Known and Unknown Size
----------------------

Every type has either known or unknown size.
Types with unknown size arise in a few ways:

* An unknown-size array type has unknown size

* A structure type has unknown size if any field type has unknown size

The use of types with unknown size is restricted as follows:

* A type with unknown size cannot be used as the element type of an array

* A type with unknown size can only be used as the last field of a structure type

* A type with unknown size cannot be used as a generic argument to specialize a user-defined type, function, etc. Specific built-in generic types/functions may support unknown-size types, and this will be documented on the specific type/function.
