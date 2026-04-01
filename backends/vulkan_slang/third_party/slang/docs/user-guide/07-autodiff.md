---
layout: user-guide
permalink: /user-guide/autodiff
---

# Automatic Differentiation

To support differentiable graphics systems such as Gaussian splatters, neural radiance fields, differentiable path tracers, and more,
Slang provides first class support for differentiable programming. 
An overview: 
- Slang supports the `fwd_diff` and `bwd_diff` operators that can generate the forward and backward-mode derivative propagation functions for any valid Slang function annotated with the `[Differentiable]` attribute. 
- The `DifferentialPair<T>` built-in generic type is used to pass derivatives associated with each function input. 
- The `IDifferentiable`, and the experimental `IDifferentiablePtrType`, interfaces denote differentiable value and pointer types respectively, and allow finer control over how types behave under differentiation.
- Further, Slang allows for user-defined derivative functions through the `[ForwardDerivative(custom_fn)]` and `[BackwardDerivative(custom_fn)]`
- All Slang features, such as control-flow, generics, interfaces, extensions, and more are compatible with automatic differentiation, though the bottom of this chapter documents some sharp edges & known issues.

## Auto-diff operations `fwd_diff` and `bwd_diff`

In Slang, `fwd_diff` and `bwd_diff` are higher-order functions used to transform Slang functions into their forward or backward derivative methods. To better understand what these methods do, here is a small refresher on differentiable calculus:
### Mathematical overview: Jacobian and its vector products
Forward and backward derivative methods are two different ways of computing a dot product with the Jacobian of a given function.
Parts of this overview are based on JAX's excellent auto-diff cookbook [here](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#how-it-s-made-two-foundational-autodiff-functions). The relevant [wikipedia article](https://en.wikipedia.org/wiki/Automatic_differentiation) is also a great resource for understanding auto-diff.
 
The [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) (also called the total derivative) of a function $$\mathbf{f}(\mathbf{x})$$ is represented by $$D\mathbf{f}(\mathbf{x})$$. 

For a general function with multiple scalar inputs and multiple scalar outputs, the Jacobian is a _matrix_ where $$D\mathbf{f}_{ij}$$ represents the [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative) of the $$i^{th}$$ output element w.r.t the $$j^{th}$$ input element $$\frac{\partial f_i}{\partial x_j}$$

As an example, consider a polynomial function

$$ f(x, y) = x^3 + x^2 - y $$

Here, $$f$$ here has 1 output and 2 inputs. $$Df$$ is therefore the row matrix:

$$ Df(x, y) = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [3x^2 + 2x, -1] $$

Another, more complex example with a function that has multiple outputs (for clarity, denoted by $$f_1$$, $$f_2$$, etc..)

$$ \mathbf{f}(x, y) = \begin{bmatrix} f_0(x, y) & f_1(x, y) & f_2(x, y) \end{bmatrix} = \begin{bmatrix} x^3 & y^2x & y^3 \end{bmatrix} $$

Here, $$D\mathbf{f}$$ is a 3x2 matrix with each element containing a partial derivative:

$$ D\mathbf{f}(x, y) = \begin{bmatrix} 
\partial f_0 / \partial x & \partial f_0 / \partial y \\  
\partial f_1 / \partial x & \partial f_1 / \partial y \\
\partial f_2 / \partial x & \partial f_2 / \partial y
\end{bmatrix} = 
\begin{bmatrix} 
3x^2  & 0   \\  
y^2   & 2yx \\
0     & 3y^2
\end{bmatrix} $$

Computing full Jacobians is often unnecessary and expensive. Instead, auto-diff offers ways to compute _products_ of the Jacobian with a vector, which is a much faster operation.
There are two basic ways to compute this product: 
 1. the Jacobian-vector product $$ \langle D\mathbf{f}(\mathbf{x}), \mathbf{v} \rangle $$, also called forward-mode autodiff, and can be computed using `fwd_diff` operator in Slang, and
 2. the vector-Jacobian product $$ \langle \mathbf{v}^T, D\mathbf{f}(\mathbf{x}) \rangle $$, also called reverse-mode autodiff, and can be computed using `bwd_diff` operator in Slang. From a linear algebra perspective, this is the transpose of the forward-mode operator. 

#### Propagating derivatives with forward-mode auto-diff
The products described above allow the _propagation_ of derivatives forward and backward through the function $f$

The forward-mode derivative (Jacobian-vector product) can convert a derivative of the inputs to a derivative of the outputs. 
For example, let's say inputs $$\mathbf{x}$$ depend on some scalar $$\theta$$, and $$\frac{\partial \mathbf{x}}{\partial \theta}$$ is a vector of partial derivatives describing that dependency.

Invoking forward-mode auto-diff with $$\mathbf{v} = \frac{\partial \mathbf{x}}{\partial \theta}$$ converts this into a derivative of the outputs w.r.t the same scalar $$\theta$$.
This can be verified by expanding the Jacobian and applying the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) of derivatives:

$$\langle D\mathbf{f}(\mathbf{x}), \frac{\partial \mathbf{x}}{\partial \theta} \rangle = \langle \begin{bmatrix} \frac{\partial f_0}{\partial x_0} & \frac{\partial f_0}{\partial x_1} & \cdots \\ \frac{\partial f_1}{\partial x_0} & \frac{\partial f_1}{\partial x_1} & \cdots \\ \cdots & \cdots & \cdots \end{bmatrix}, \begin{bmatrix} \frac{\partial x_0}{\partial \theta} \\ \frac{\partial x_1}{\partial \theta} \\ \cdots \end{bmatrix} \rangle = \begin{bmatrix} \frac{\partial f_0}{\partial \theta} \\ \frac{\partial f_1}{\partial \theta} \\ \cdots \end{bmatrix} = \frac{\partial \mathbf{f}}{\partial \theta}$$

#### Propagating derivatives with reverse-mode auto-diff
The reverse-mode derivative (vector-Jacobian product) can convert a derivative w.r.t outputs into a derivative w.r.t inputs.
For example, let's say we have some scalar $$\mathcal{L}$$ that depends on the outputs $$\mathbf{f}$$, and $$\frac{\partial \mathcal{L}}{\partial \mathbf{f}}$$ is a vector of partial derivatives describing that dependency.

Invoking forward-mode auto-diff with $$\mathbf{v} = \frac{\partial \mathcal{L}}{\partial \mathbf{f}}$$ converts this into a derivative of the same scalar $$\mathcal{L}$$ w.r.t the inputs $$\mathbf{x}$$.
To provide more intuition for this, we can expand the Jacobian in a same way we did above:

$$\langle \frac{\partial \mathcal{L}}{\partial \mathbf{f}}^T, D\mathbf{f}(\mathbf{x}) \rangle = \langle \begin{bmatrix}\frac{\partial \mathcal{L}}{\partial f_0} & \frac{\partial \mathcal{L}}{\partial f_1} & \cdots \end{bmatrix}, \begin{bmatrix} \frac{\partial f_0}{\partial x_0} & \frac{\partial f_0}{\partial x_1} & \cdots \\ \frac{\partial f_1}{\partial x_0} & \frac{\partial f_1}{\partial x_1} & \cdots \\ \cdots & \cdots & \cdots \end{bmatrix} \rangle = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial x_0} & \frac{\partial \mathcal{L}}{\partial x_1} & \cdots \end{bmatrix} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}}^T$$

This mode is the most popular, since machine learning systems often construct their differentiable pipeline with multiple inputs (which can number in the millions or billions), and a single scalar output often referred to as the 'loss' denoted by $$\mathcal{L}$$. The desired derivative can be constructed with a single reverse-mode invocation.

### Invoking auto-diff in Slang
With the mathematical foundations established, we can describe concretely how to compute derivatives using Slang.

In Slang derivatives are computed using `fwd_diff`/`bwd_diff` which each correspond to Jacobian-vector and vector-Jacobian products.
For forward-diff, to pass the vector $$\mathbf{v}$$ and receive the outputs, we use the `DifferentialPair<T>` type. We use pairs of inputs because every input element $$x_i$$ has a corresponding element $$v_i$$ in the vector, and each original output element has a corresponding output element in the product.

Example of `fwd_diff`:
```csharp
[Differentiable] // Auto-diff requires that functions are marked differentiable
float2 foo(float a, float b) 
{ 
    return float2(a * b * b, a * a);
}

void main()
{
    DifferentialPair<float> dp_a = diffPair(
        1.0, // input 'a'
        1.0  // vector 'v' for vector-Jacobian product input (for 'a')
    );

    DifferentialPair<float> dp_b = diffPair(2.4, 0.0);

    // fwd_diff to compute output and d_output w.r.t 'a'.
    // Our output is also a differential pair.
    //
    DifferentialPair<float2> dp_output = fwd_diff(foo)(dp_a, dp_b);

    // Extract output's primal part, which is just the standard output when foo is called normally.
    // Can also use `.getPrimal()`
    //
    float2 output_p = dp_output.p;

    // Extract output's derivative part. Can also use `.getDifferential()`
    float2 output_d = dp_output.d;

    printf("foo(1.0, 2.4) = (%f %f)\n", output_p.x, output_p.y);
    printf("d(foo)/d(a) at (1.0, 2.4) = (%f, %f)\n", output_d.x, output_d.y);
}
```

Note that all the inputs and outputs to our function become 'paired'. This only applies to differentiable types, such as `float`, `float2`, etc. See the section on differentiable types for more info.

`diffPair<T>(primal_val, diff_val)` is a built-in utility function that constructs the pair from the primal and differential values.  

Additionally, invoking forward-mode also computes the regular (or 'primal') output value (can be obtained from `output.getPrimal()` or `output.p`). The same is _not_ true for reverse-mode.

For reverse-mode, the example proceeds in a similar way, and we still use `DifferentialPair<T>` type. However, note that each input gets a corresponding _output_ and each output gets a corresponding _input_. Thus, all inputs become `inout` differential pairs, to allow the function to write into the derivative part (the primal part is still accepted as an input in the same pair data-structure).
The one extra rule is that the derivative corresponding to the return value of the function is accepted as the last argument (an extra input). This value does not need to be a pair.

Example:
```csharp
[Differentiable] // Auto-diff requires that functions are marked differentiable
float2 foo(float a, float b) 
{ 
    return float2(a * b * b, a * a);
}

void main()
{
    DifferentialPair<float> dp_a = diffPair(
        1.0 // input 'a'
    ); // Calling diffPair without a derivative part initializes to 0.

    DifferentialPair<float> dp_b = diffPair(2.4);

    // Derivatives of scalar L w.r.t output.
    float2 dL_doutput = float2(1.0, 0.0);

    // bwd_diff to compute dL_da and dL_db
    // The derivative of the output is provided as an additional _input_ to the call
    // Derivatives w.r.t inputs are written into dp_a.d and dp_b.d
    //
    bwd_diff(foo)(dp_a, dp_b, dL_doutput);

    // Extract the derivatives of L w.r.t input
    float dL_da = dp_a.d;
    float dL_db = dp_b.d;

    printf("If dL/dOutput = (1.0, 0.0), then (dL/da, dL/db) at (1.0, 2.4) = (%f, %f)", dL_da, dL_db);
}
```

## Differentiable Type System

Slang will only generate differentiation code for values that has a *differentiable* type. 
Differentiable types are defining through conformance to one of two built-in interfaces:
1. `IDifferentiable`: For value types (e.g. `float`, structs of value types, etc..)
2. `IDifferentiablePtrType`: For buffer, pointer & reference types that represent locations rather than values.

### Differentiable Value Types
All basic types (`float`, `int`, `double`, etc..) and all aggregate types (i.e. `struct`) that use any combination of these are considered value types in Slang.

Slang uses the `IDifferentiable` interface to define differentiable types. Basic types that describe a continuous value (`float`, `double` and `half`) and their vector/matrix versions (`float3`, `half2x2`, etc..) are defined as differentiable by the standard library. For all basic types, the type used for the differential (can be obtained with `T.Differential`) is the same as the primal.

#### Builtin Differentiable Value Types
The following built-in types are differentiable: 
- Scalars: `float`, `double` and `half`.
- Vector/Matrix: `vector` and `matrix` of `float`, `double` and `half` types.
- Arrays: `T[n]` is differentiable if `T` is differentiable.
- Tuples: `Tuple<each T>` is differentiable if `T` is differentiable. 


#### User-defined Differentiable Value Types

However, it is easy to define your own differentiable types.
Typically, all you need is to implement the `IDifferentiable` interface. 

```csharp
struct MyType : IDifferentiable
{
    float x;
    float y;
};
```

The main requirement of a type implementing `IDifferentiable` is the `Differential` associated type that the compiler uses to carry the corresponding derivative.
In most cases the `Differential` of a type can be itself, though it can be different if necessary.
You can access the differential of any differentiable type through `Type.Differential`

Example:
```csharp
MyType obj;
obj.x = 1.f;

MyType.Differential d_obj;
// Differentiable fields will have a corresponding field in the diff type
d_obj.x = 1.f;
```

Slang can automatically derive the `Differential` type in the majority of cases.
For instance, for `MyType`, Slang can infer the differential trivially:
```csharp
struct MyType : IDifferentiable
{
    // Automatically inserted by Slang from the fact that 
    // MyType has 2 floats which are both differentiable
    //
    typealias Differential = MyType;
    // ...
}
```

For more complex types that aren't fully differentiable, a new type is synthesized automatically:

```csharp
struct MyPartialDiffType : IDifferentiable
{
    // Automatically inserted by Slang based on which fields are differentiable.
    typealias MyPartialDiffType = syn_MyPartialDiffType_Differential;
    
    float x;
    uint y;
};

// Synthesized
struct syn_MyPartialDiffType_Differential
{
    // Only one field since 'y' does not conform to IDifferentiable
    float x;
};
```

You can make existing types differentiable through Slang's extension mechanism.
For instance, `extension MyType : IDifferentiable { }` will make `MyType` differentiable retroactively.

See the `IDifferentiable` [reference documentation](https://shader-slang.org/stdlib-reference/interfaces/idifferentiable-01/index) for more information on how to override the default behavior.

#### DifferentialPair<T>: Pairs of differentiable value types

The `DifferentialPair<T>` type is used to pass derivatives to a derivative call by representing a pair of values of type `T` and `T.Differential`. Note that `T` must conform to `IDifferentiable`.

`DifferentialPair<T>` can either be created via constructor calls or the `diffPair` utility method.

Example:

```csharp
MyType obj = {1.f, 2.f};

MyType.Differential d_obj = {0.4f, 3.f};

// The differential part of a differentiable-pair is of the diff type.
DifferentialPair<MyType> dp_obj = diffPair(obj, d_obj);

// Use .p to extract the primal part
MyType new_p_obj = dp_obj.p;

// Use .d to extract the differential part
MyType.Differential new_d_obj = dp_obj.d;
```

### Differentiable Ptr types
Pointer types are any type that represents a location or reference to a value rather than the value itself.
Examples include resource types (`RWStructuredBuffer`, `Texture2D`), pointer types (`Ptr<float>`) and references.

The `IDifferentiablePtrType` interface can be used to denote types that need to transform into pairs during auto-diff. However, unlike
an `IDifferentiable` type whose derivative portion is an _output_ under `bwd_diff`, the derivative part of `IDifferentiablePtrType` remains an input. This is because only the value is returned as an output, while the location where it needs to be written to, is still effectively an input to the derivative methods.

> #### Note ####
> Support for `IDifferentiablePtrType` is still experimental. There are no built-in types conforming to this interface, though we plan to add stdlib support in the near future.

`IDifferentiablePtrType` only requires a `Differential` associated type to be specified.

#### DifferentialPtrPair<T>: Pairs of differentiable ptr types
For types conforming to `IDifferentiablePtrType`, the corresponding pair to use for passing the derivative counterpart is `DifferentialPtrPair<T>`, which represents a pair of `T` and `T.Differential`. Objects of this type can be created using a constructor.

#### Example of defining and using an `IDifferentiablePtrType` object.
Here is an example of create a differentiable buffer pointer type, and using it within a differentiable function.
You can find an interactive sample on the Slang playground [here](https://shader-slang.org/slang-playground/?target=WGSL&code=eJy1VF1v2kAQfPevWEWKYhfkmFdMkBrRSpHKhyBSpdIIHfgcTjFn9z4gEeK_d-_ONsZp1L6UF8MxOzszuz62K3KhoMjI27PINU9iTyqhNwrGb_c6TamY5YwrKqAPDyNmDihXjKwzOlPi8a2g3tED_NzewuOWQnKGZFAQpGYSCM_VFikYl4rwDYU8bdOHlkQhH8kYkTBq8ty10bFn4fPvC6tVC5q4_wdplhM1hLVOYwvRiMd2qaQq9k5Yhzq_Mf4CBDZaqnwHCRVsTxTbU295TzYvByKSUX3mI1-yWh-S4Mmz3GAO_HY4Rdd1Yjyhr0EZiaCojEMRopplEToV0HGgJ5Rj1UxyRUFtkRkzgnWpoCELJHvmxJg0WUrFsgwThRvGb9pxM8xxn7MEKtV-M0cc2Awhg5b44aX6LjifyVSryknbrmm7KpTAyRRh4pKuzqzb-kfLNHTuLLE1v7zcpypgqXfTdPFLE0HlIMPiCe4e9h2-T73SVxbmEgVFYVqux3JMXh8QJ_0JTs_icgG-s2qQMT4GMMFHpxNYgKM7U-5JtjJQO3SMiQVxjTDt0I6DfHJP9-_Ja84fcc7u-SXr9-efJyO_F6Guj5eY8UIrrL2s_PFlPl38rdRujym121AItDwmjPsfDdTN8ug6diE6OSO20L_6yoRU0IuMR01lH37yqzKIPybaixqRNniukz5cp1iMQXbLTJUwqQblyErgQu_MHSHdEpivaVtCyXOxLL1oaAhrtndra1Ip9_boInJeqxtsRiTeVvZFMk0LV4dH0g3D3VL4Xq3Mgvvt5oFfO_6X986Zr0UF3bq6F0Zlvs1UzreSEYc9dabgEIrQXR3_ZT61unJKp98JDfhi).
```csharp
struct MyBufferPointer : IDifferentiablePtrType
{
    // The differential part is another instance of MyBufferPointer.
    typealias Differential = MyBufferPointer;

    RWStructuredBuffer<float> buf;
    uint offset;
};

// Link a custom derivative
[BackwardDerivative(load_bwd)]
float load(MyBufferPointer p, uint index)
{
    return p.buf[p.offset + index];
}

// Note that the backward derivative signature is still an 'in' differential pair.
void load_bwd(DifferentialPtrPair<MyBufferPointer> p, uint index, float dOut)
{
    MyBufferPointer diff_ptr = p.d;
    diff_ptr.buf[diff_ptr.offset + index] += dOut;
}

[Differentiable]
float sumOfSquares<let N : int>(MyBufferPointer p)
{
    float sos = 0.f;

    [MaxIters(N)]
    for (uint i = 0; i < N; i++)
    {
        float val_i = load(p, i);
        sos += val_i * val_i;
    }

    return sos;
}

RWStructuredBuffer<float> inputs;
RWStructuredBuffer<float> derivs;

void main()
{
    MyBufferPointer ptr = {inputs, 0};
    print("Sum of squares of first 10 values: ", sumOfSquares<10>(ptr));

    MyBufferPointer deriv_ptr = {derivs, 0};

    // Pass a pair of pointers as input.
    bwd_diff(sumOfSquares<10>)(
        DifferentialPtrPair<MyBufferPointer>(ptr, deriv_ptr),
        1.0);
    
    print("Derivative of result w.r.t the 10 values: \n");
    for (uint i = 0; i < 10; i++)
        print("%d: %f\n", i, load(deriv_ptr, i));
}
```

## User-Defined Derivative Functions

As an alternative to compiler-generated derivatives, you can choose to provide an implementation for the derivative, which the compiler will use instead of attempting to generate one. 

This can be performed on a per-function basis by using the decorators `[ForwardDerivative(fwd_deriv_func)]` and `[BackwardDerivative(bwd_deriv_func)]` to reference the derivative from the primal function.

For instance, it often makes little sense to differentiate the body of a `sin(x)` implementation, when we know that the derivative is `cos(x) * dx`. In Slang, this can be represented in the following way:
```csharp
DifferentialPair<float> sin_fwd(DifferentialPair<float> dpx)
{
    float x = dpx.p;
    float dx = dpx.d;
    return DifferentialPair<float>(dpx.p, cos(x) * dx);
}

// sin() is now considered differentiable (atleast for forward-mode) since it provides
// a derivative implementation.
//
[ForwardDerivative(sin_fwd)]
float sin(float x)
{
    // Calc sin(X) using Taylor series..
}

// Any uses of sin() in a `[Differentiable]` will automaticaly use the sin_fwd implementation when differentiated.
```

A similar example for a backward derivative.
```csharp
void sin_bwd(inout DifferentialPair<float> dpx, float dresult)
{
    float x = dpx.p;

    // Write-back the derivative to each input (the primal part must be copied over as-is)
    dpx = DifferentialPair<float>(x, cos(x) * dresult);
}

[BackwardDerivative(sin_bwd)]
float sin(float x)
{
    // Calc sin(X) using Taylor series..
}
```

> Note that the signature of the provided forward or backward derivative function must match the expected signature from invoking `fwd_diff(fn)`/`bwd_diff(fn)`
> For a full list of signature rules, see the reference section for the [auto-diff operators](#fwd_difff--slang_function---slang_function).

### Back-referencing User Derivative Attributes.
Sometimes, the original function's definition might be inaccessible, so it can be tricky to add an attribute to create the association.

For such cases, Slang provides the `[ForwardDerivativeOf(primal_fn)]` and `[BackwardDerivativeOf(primal_fn)]` attributes that can be used
on the derivative function and contain a reference to the function for which they are providing a derivative implementation.
As long as both the derivative function is in scope, the primal function will be considered differentiable.

Example:
```csharp
// Module A
float sin(float x) { /* ... */ } 

// Module B
import A;
[BackwardDerivativeOf(sin)] // Add a derivative implementation for sin() in module A.
void sin_bwd(inout DifferentialPair<float> dpx, float dresult) { /* ... */ }
```

User-defined derivatives also work for generic functions, member functions, accessors, and more. 
See the reference section for the [`[ForwardDerivative(fn)]`](https://shader-slang.org/stdlib-reference/attributes/forwardderivative-07.html) and [`[BackwardDerivative(fn)]`](https://shader-slang.org/stdlib-reference/attributes/backwardderivative-08) attributes for more. 

## Using Auto-diff with Generics
Automatic differentiation works seamlessly with generically-defined types and methods.
For generic methods, differentiability of a type is defined either through an explicit `IDifferentiable` constraint or any other
interface that extends `IDifferentiable`.

Example for generic methods:
```csharp
[Differentiable]
T calcFoo<T : IDifferentiable>(T x) { /* ... */ }

[Differentiable]
T calcBar<T : __BuiltinFloatingPointType>(T x) { /* ... */ }

[Differentiable]
void main()
{
    DifferentialPair<float4> dpa = /* ... */;

    // Can call with any type that is IDifferentiable. Generic parameters
    // are inferred like any other call.
    //
    bwd_diff(calcFoo)(dpa, float4(1.f));

    // But you can also be explicit with < >
    bwd_diff(calcFoo<float4>)(dpa, float4(1.f));

    // x is differentiable for calcBar because 
    // __BuiltinFloatingPointType : IDifferentiable
    //
    DifferentialPair<double> dpb = /* .. */;
    bwd_diff(calcBar)(dpb, 1.0);
}
```

You can implement `IDifferentiable` on a generic type. Automatic synthesis still applies and will use
generic constraints to resolve whether a field is differentiable or not.
```csharp
struct Foo<T : IDifferentiable, U> : IDifferentiable
{
    T t;
    U u;
};

// The synthesized Foo<T, U>.Differential will contain a field for
// 't' but not 'U'
//
```

## Using Auto-diff with Interface Requirements and Interface Types
For interface requirements, using `[Differentiable]` attribute enforces that any implementation of that method must also be
differentiable. You can, of course, provide a manual derivative implementation to satisfy the requirement.

The following is a sample snippet. You can run the full sample on the playground [here](https://shader-slang.org/slang-playground/?target=HLSL&code=eJyVVMtu2zAQvOsrFgEKy4Wq1C7QQ1330AYBcujjnhbBWiRjphQpUJQjI8i_d0mRquLYASLYtLwc7sySs5R1Y6yDRuH-1ppOs1UmteNWYMXh6tKY7CEDeq4vpBDccu0kbhT_E4JCGXRQoary4bWfr7LHLGud7SoHtPqqbhR8miYagJTeGbvKQuj8HDyO15QdnTQadhIBO2dq-lsB-0_tZ8tXCQrxgdo_lrvOaujhLXwoBY1RCQTE43P1y5fk-8gLJdSoO1RqD401O8mkvgXGrdwRYseh5m5rWBvLuTT2Hi27GOdzX8aNuGfzobbrr1j9PQbZjJBXlb-clh-rDz-TjVW_UNrPIdcXSHryU4BTbP78PC7vy2YgLiJ7X7JRw_yJiJ2RDFJ5udSmcyeF9UWsnFnedsodyuhhfWqtl1SkdQebMgoiSd4BcMvdz81d3lGDgNnc3Ug2j66QAvIhAus1LA4FpEaQfljDw6L8KB5Xh9vkZxOlH7lq-fFEyzET6X05EWl_1inRJqZuOseTUwo4UtfxZv1mOToOSEy6dajppjACuHRbbpPEBZjxfRkWhi0U9F2njYxcsfdS9ou9xjp0fdugq7bgDGBDHdRY6Wln3hWzMsLTqh-GptzWqyUK6VquBMgWtNHv2JP6CxLORrYtesz0ilHA0GEBG3LcrJ8F9GwwyCytupdKkTut3U8aui3bqagdWoi-WntRZehLf0Nmk7MaEOHWDJanIrX7jlLn6QxOuZ41SInH3lqW76NhqWFufDiPJzzPCVrAgj6lSPSBJz97I37rs8LnKpm_u_8BU5nW2Q). 
```csharp
interface IFoo
{
    [Differentiable]
    float calc(float x);
}

struct FooImpl : IFoo
{
    // Implementation via automatic differentiation.
    [Differentiable]
    float calc(float x)
    { /* ... */ }
}

struct FooImpl2 : IFoo
{
    // Implementation via manually providing derivative methods.
    [ForwardDerivative(calc_fwd)]
    [BackwardDerivative(calc_bwd)]
    float calc(float x)
    { /* ... */ }

    DifferentialPair<float> calc_fwd(DifferentialPair<float> x)
    { /* ... */ }

    void calc_bwd(inout DifferentialPair<float> x, float dresult)
    { /* ... */ }
}

[Differentiable]
float compute(float x, uint obj_id)
{
    // Create an instance of either FooImpl1 or FooImpl2
    IFoo foo = createDynamicObject<IFoo>(obj_id); 
    
    // Dynamic dispatch to appropriate 'calc'.
    //
    // Note that foo itself is non-differentiable, and 
    // has no differential data, but 'x' and 'result'
    // will carry derivatives.s
    //
    var result = foo.calc(x);
    return result;
}
```

### Differentiable Interface (and Associated) Types
> Note: This is an advanced use-case and support is currently experimental.

You can have an interface or an interface associated type extend `IDifferentiable` and use that in differentiable interface requirement functions. This is often important in large code-bases with modular components that are all differentiable (one example is the material system in large production renderers)

Here is a snippet of how to make an interface and associated type (and by consequence all its implementations) differentiable. 
For a full working sample, check out the Slang playground [here](https://shader-slang.org/slang-playground/?target=WGSL&code=eJylVVFvmzAQfudXnCpVgoXRhK4vpdnLuodIq9pqe9umygGzuXPANaYjqvLfd8bgmIR0a-eHcHfcnT9_38WwlSilAsHJ-ocs6yJLPI8VisqcpBQWHwhPa04UKws4h8Uly3MqaaEYWXLqPXmA6-sw-r0N5rwkCogQfO0buw4SbzPs_kWSospLuTrYm1RVmTKiaKbWgsIOHMfFxgexuFUr8ov6HZJKyTpV8IkVlMibkq-LcsUI3-ncIekOlDjO0jgvIaEJ4AkkVbUsgMAbaGCCbWDjbSyc25pkEndOX4_IOOlznDwPzbfYgs5IhyANZwstpSi3glhBO4haNMIZqQYazPcod2ELNRu68cu0bcNme7321CUpAnjy1U9WRdgc3kJnzoLQmpvENujVSk1oVKpXEzEitjNUhwnZuiuW3Qn1XxSNTdxDy5JN0SvGUdCETeBda82QOm0ZBOEgdxvHpDObjuXDPAxbf5_zB5fzvbM514c-1vXy3q_xcgGWhR03j4TPHDsOOjVYDj7LYD6H6fi4DPXkTHNhmuk2-0A564HqX8oreojiYeeHnc4h-JplHUCaW8hwAqdRPsINe46b7gZA5Q0nev56JoTlRMS91fTUOKSWy3tE11NrOuhaEQfduA0-D3pgsCTqb1gHaxqZi6bRF6_nPZZIvpCI64qwwu-3boFeXV9-_Iyd-hkvJXSqYnCa4OPC5KA5meyqZw7TfmHEDU4clkRxcuB1jK9P3dctJP9oKNFVmVE4zsBv5tPoLDiH4_xbcRQCCw29-LT7bU0kVmf3ROnlSMRvCJMXLZr3kIkGgWT4Vkd9XZb8Q9HCOaQttkhe1CIeaxE7LZa_szud4OsTB_rIzv6uE2unCWEWTd2jd8RmvqRVzVVwkuEoWCaxIsqCPRncbH05O_l277_XxWN1sa3Df88fIn-viQ)

```csharp
interface IFoo : IDifferentiable
{
    associatedtype BaseType : IDifferentiable;

    [Differentiable]
    BaseType foo(BaseType x);
};

[Differentiable]
float calc(float x)
{
    // Note that since IFoo is differentiable, 
    // any data in the IFoo implementation is differentiable
    // and will carry derivatives.
    //
    IFoo obj = makeObj(/* ... */);
    return obj.foo(x);
}
```

Under the hood, Slang will automatically construct an anonymous abstract type to represent the differentials. 
However, on targets that don't support true dynamic dispatch, these are lowered into tagged unions. 
While we are working to improve the implementation, this union can currently include all active differential 
types, rather than just the relevant ones. This can lead to increased memory use.

## Primal Substitute Functions

Sometimes it is desirable to replace a function with another when generating derivative code. 
Most often, this is because a lot of shader operations may just not have a function body, such hardware intrinsics for
texture sampling. In such cases, Slang provides a `[PrimalSubstitute(fn)]` attribute that can be used to provide
a reference implementation that Slang can differentiate to generate the derivative function.

The following is a small snippet with bilinear texture sampling. For a full example application that uses this concept, see the [texture differentiation sample](https://github.com/shader-slang/slang/tree/master/examples/autodiff-texture) in the Slang repository.

```csharp
[PrimalSubstitute(sampleTextureBiliear_reference)]
float4 sampleTextureBilinear(Texture2D<float4> x, float2 loc) 
{ 
    // HW-accelerated sampling intrinsics. 
    // Slang does not have access to body, so cannot differentiate.
    //
    x.Sample(/*...*/)
}

// Since the substitute is differentiable, so is `sampleTextureBilinear`.
[Differentiable]
float4 sampleTextureBilinear_reference(Texture2D<float4> x, float2 loc)
{
    // Reference SW interpolation, that is differentiable.
}

[Differentiable]
float computePixel(Texture2D<float> x, float a, float b)
{
    // Slang will use HW-accelerated sampleTextureBilinear for standard function
    // call, but differentiate the SW reference interpolation during backprop.
    // 
    float4 sample1 = sampleTextureBilinear(x, float2(a, 1));
}
```

Similar to `[ForwardDerivativeOf(fn)]` and `[BackwardDerivativeOf(fn)]` attributes, Slang provides a `[PrimalSubstituteOf(fn)]` attribute that can be used on the substitute function to reference the primal one.

## Working with Mixed Differentiable and Non-Differentiable Code

Introducing differentiability to an existing system often involves dealing with code that mixes differentiable and non-differentiable logic.
Slang provides type checking and code analysis features to allow users to clarify the intention and guard against unexpected behaviors involving when to propagate derivatives through operations.

### Excluding Parameters from Differentiation

Sometimes we do not wish a parameter to be considered differentiable despite it has a differentiable type. We can use the `no_diff` modifier on the parameter to inform the compiler to treat the parameter as non-differentiable and skip generating differentiation code for the parameter. The syntax is:

```csharp
// Only differentiate this function with regard to `x`.
float myFunc(no_diff float a, float x);
```

The forward derivative and backward propagation functions of `myFunc` should have the following signature:
```csharp
DifferentialPair<float> fwd_derivative(float a, DifferentialPair<float> x);
void back_prop(float a, inout DifferentialPair<float> x, float dResult);
```

In addition, the `no_diff` modifier can also be used on the return type to indicate the return value should be considered non-differentiable. For example, the function
```csharp
no_diff float myFunc(no_diff float a, float x, out float y);
```
Will have the following forward derivative and backward propagation function signatures:

```csharp
float fwd_derivative(float a, DifferentialPair<float> x);
void back_prop(float a, inout DifferentialPair<float> x, float d_y);
```

By default, the implicit `this` parameter will be treated as differentiable if the enclosing type of the member method is differentiable. If you wish to exclude `this` parameter from differentiation, use `[NoDiffThis]` attribute on the method:
```csharp
struct MyDifferentiableType : IDifferentiable
{
    [NoDiffThis]   // Make `this` parameter `no_diff`.
    float compute(float x) { ... }
}
```

### Excluding Struct Members from Differentiation

When using automatic `IDifferentiable` conformance synthesis for a `struct` type, Slang will by-default treat all struct members that have a differentiable type as differentiable, and thus include a corresponding field in the generated `Differential` type for the struct.
For example, given the following definition
```csharp
struct MyType : IDifferentiable
{
    float member1;
    float2 member2;
}
```
Slang will generate:
```csharp
struct MyType.Differential : IDifferentiable
{
    float member1;  // derivative for MyType.member1
    float2 member2; // derivative for MyType.member2
}
```
If the user does not want a certain member to be treated as differentiable despite it has a differentiable type, a `no_diff` modifier can be used on the struct member to exclude it from differentiation.
For example, the following code excludes `member1` from differentiation:
```csharp
struct MyType : IDifferentiable
{
    no_diff float member1;  // excluded from differentiation
    float2 member2;
}
```
The generated `Differential` in this case will be:
```csharp
struct MyType.Differential : IDifferentiable
{
    float2 member2;
}
```

### Assigning Differentiable Values into a Non-Differentiable Location

When a value with derivatives is being assigned to a location that is not differentiable, such as a struct member that is marked as `no_diff`, the derivative info is discarded and any derivative propagation is stopped at the assignment site.
This may lead to unexpected results. For example:
```csharp
struct MyType : IDifferentiable
{
    no_diff float member;
    float someOtherMember;
}
[Differentiable]
float f(float x)
{
    MyType t;
    t.member = x * x; // Error: assigning value with derivative into a non-differentiable location.
    return t.member;
}
```
In this case, we are assigning the value `x*x`, which carries a derivative, into a non-differentiable location `MyType.member`, thus throwing away any derivative info. When `f` returns `t.member`, there will be no derivative associated with it, so the function will not propagate the derivative through. This code is most likely not intending to discard the derivative through the assignment. To help avoid this kind of unintentional behavior, Slang will treat any assignments of a value with derivative info into a non-differentiable location as a compile-time error. To eliminate this error, the user should either make `t.member` differentiable, or to force the assignment by clarifying the intention to discard any derivatives using the built-in `detach` method.
The following code will compile, and the derivatives will be discarded:
```csharp
[Differentiable]
float f(float x)
{
    MyType t;
    // OK: the code has expressed clearly the intention to discard the derivative and perform the assignment.
    t.member = detach(x * x);
    return t.member;
}
```

### Calling Non-Differentiable Functions from a Differentiable Function
Calling non-differentiable function from a differentiable function is allowed. However, derivatives will not be propagated through the call. The user is required to clarify the intention by prefixing the call with the `no_diff` keyword. An un-clarified call to non-differentiable function will result in a compile-time error.

For example, consider the following code:
```csharp
float g(float x)
{
    return 2*x;
}

[Differentiable]
float f(float x)
{
    // Error: implicit call to non-differentiable function g.
    return g(x) + x * x;
}
```
The derivative will not propagate through the call to `g` in `f`. As a result, `fwd_diff(f)(diffPair(1.0, 1.0))` will return
`{3.0, 2.0}` instead of `{3.0, 4.0}` as the derivative from `2*x` is lost through the non-differentiable call. To prevent unintended error, it is treated as a compile-time error to call `g` from `f`. If such a non-differentiable call is intended, a `no_diff` prefix is required in the call:
```csharp
[Differentiable]
float f(float x)
{
    // OK. The intention to call a non-differentiable function is clarified.
    return no_diff g(x) + x * x;
}
```

However, the `no_diff` keyword is not required in a call if a non-differentiable function does not take any differentiable parameters, or if the result of the differentiable function is not dependent on the derivative being propagated through the call.

### Treat Non-Differentiable Functions as Differentiable
Slang allows functions to be marked with a `[TreatAsDifferentiable]` attribute for them to be considered as differentiable functions by the type-system. When a function is marked as `[TreatAsDifferentiable]`, the compiler will not generate derivative propagation code from the original function body or perform any additional checking on the function definition. Instead, it will generate trivial forward and backward propagation functions that returns 0.

This feature can be useful if the user marked an `interface` method as forward or backward differentiable, but only wish to provide non-trivial derivative propagation functions for a subset of types that implement the interface. For other types that does not actually need differentiation, the user can simply put `[TreatAsDifferentiable]` on the method implementations for them to satisfy the interface requirement.

See the following code for an example of `[TreatAsDifferentiable]`:
```csharp
interface IFoo
{
    [Differentiable]
    float f(float v);
}

struct B : IFoo
{
    [TreatAsDifferentiable]
    float f(float v)
    {
        return v * v;
    }
}

[Differentiable]
float use(IFoo o, float x)
{
    return o.f(x);
}

// Test:
B obj;
float result = fwd_diff(use)(obj, diffPair(2.0, 1.0)).d;
// result == 0.0, since `[TreatAsDifferentiable]` causes a trivial derivative implementation
// being generated regardless of the original code.
```

## Higher-Order Differentiation

Slang supports generating higher order forward and backward derivative propagation functions. It is allowed to use `fwd_diff` and `bwd_diff` operators inside a forward or backward differentiable function, or to nest `fwd_diff` and `bwd_diff` operators. For example, `fwd_diff(fwd_diff(sin))` will have the following signature:

```csharp
DifferentialPair<DifferentialPair<float>> sin_diff2(DifferentialPair<DifferentialPair<float>> x);
```

The input parameter `x` contains four fields: `x.p.p`, `x.p.d,`, `x.d.p`, `x.d.d`, where `x.p.p` specifies the original input value, both `x.p.d` and `x.d.p` store the first order derivative if `x`, and `x.d.d` stores the second order derivative of `x`. Calling `fwd_diff(fwd_diff(sin))` with `diffPair(diffPair(pi/2, 1.0), DiffPair(1.0, 0.0))` will result `{ { 1.0, 0.0 }, { 0.0, -1.0 } }`.

User defined higher-order derivative functions can be specified by using `[ForwardDerivative]` or `[BackwardDerivative]` attribute on the derivative function, or by using `[ForwardDerivativeOf]` or `[BackwardDerivativeOf]` attribute on the higher-order derivative function.

## Restrictions and Known Issues

The compiler can generate forward derivative and backward propagation implementations for most uses of array and struct types, including arbitrary read and write access at dynamic array indices, and supports uses of all types of control flows, mutable parameters, generics and interfaces. This covers the set of operations that is sufficient for a lot of functions. However, the user needs to be aware of the following restrictions when using automatic differentiation:

- All operations to global resources, global variables and shader parameters, including texture reads or atomic writes, are treated as a non-differentiable operation. Slang provides support for special data-structures (such as `Tensor`) through libraries such as `SlangPy`, which come with custom derivative implementations
- If a differentiable function contains calls that cause side-effects such as updates to global memory, there is currently no guarantee on how many times side-effects will occur during the resulting derivative function or back-propagation function.
- Loops: Loops must have a bounded number of iterations. If this cannot be inferred statically from the loop structure, the attribute `[MaxIters(<count>)]` can be used specify a maximum number of iterations. This will be used by compiler to allocate space to store intermediate data. If the actual number of iterations exceeds the provided maximum, the behavior is undefined. You can always mark a loop with the `[ForceUnroll]` attribute to instruct the Slang compiler to unroll the loop before generating derivative propagation functions. Unrolled loops will be treated the same way as ordinary code and are not subject to any additional restrictions.
- Double backward derivatives (higher-order differentiation): The compiler does not currently support multiple backward derivative calls such as `bwd_diff(bwd_diff(fn))`. The vast majority of higher-order derivative applications can be acheived more efficiently via multiple forward-derivative calls or a single layer of `bwd_diff` on functions that use one or more `fwd_diff` passes.

The above restrictions do not apply if a user-defined derivative or backward propagation function is provided.

## Reference

This section contains some additional information for operators that are not currently included in the [standard library reference](https://shader-slang.org/stdlib-reference/)

### `fwd_diff(f : slang_function) -> slang_function`
The `fwd_diff` operator can be used on a differentiable function to obtain the forward derivative propagation function.

A forward derivative propagation function computes the derivative of the result value with regard to a specific set of input parameters. 
Given an original function, the signature of its forward propagation function is determined using the following rules:
- If the return type `R` implements `IDifferentiable` the forward propagation function will return a corresponding `DifferentialPair<R>` that consists of both the computed original result value and the (partial) derivative of the result value. Otherwise, the return type is kept unmodified as `R`.
- If a parameter has type `T` that implements `IDifferentiable`, it will be translated into a `DifferentialPair<T>` parameter in the derivative function, where the differential component of the `DifferentialPair` holds the initial derivatives of each parameter with regard to their upstream parameters.
- If a parameter has type `T` that implements `IDifferentiablePtrType`, it will be translated into a `DifferentialPtrPair<T>` parameter where the differential component references the differential component.
- All parameter directions are unchanged. For example, an `out` parameter in the original function will remain an `out` parameter in the derivative function.
- Differentiable methods cannot have a type implementing `IDifferentiablePtrType` as an `out` or `inout` parameter, or a return type. Types implementing `IDifferentiablePtrType` can only be used for input parameters to a differentiable method. Marking such a method as `[Differentiable]` will result in a compile-time diagnostic error.

For example, given original function:
```csharp
[Differentiable]
R original(T0 p0, inout T1 p1, T2 p2, T3 p3);
```
Where `R`, `T0`, `T1 : IDifferentiable`, `T2` is non-differentiable, and `T3 : IDifferentiablePtrType`, the forward derivative function will have the following signature:
```csharp
DifferentialPair<R> derivative(DifferentialPair<T0> p0, inout DifferentialPair<T1> p1, T2 p2, DifferentialPtrPair<T3> p3);
```

This forward propagation function takes the initial primal value of `p0` in `p0.p`, and the partial derivative of `p0` with regard to some upstream parameter in `p0.d`. It takes the initial primal and derivative values of `p1` and updates `p1` to hold the newly computed value and propagated derivative. Since `p2` is not differentiable, it remains unchanged.

### `bwd_diff(f : slang_function) -> slang_function`

A backward derivative propagation function propagates the derivative of the function output to all the input parameters simultaneously.

Given an original function `f`, the general rule for determining the signature of its backward propagation function is that a differentiable output `o` becomes an input parameter holding the partial derivative of a downstream output with regard to the differentiable output, i.e. $$\partial y/\partial o$$; an input differentiable parameter `i` in the original function will become an output in the backward propagation function, holding the propagated partial derivative $$\partial y/\partial i$$; and any non-differentiable outputs are dropped from the backward propagation function. This means that the backward propagation function never returns any values computed in the original function.

More specifically, the signature of its backward propagation function is determined using the following rules:
- A backward propagation function always returns `void`.
- A differentiable `in` parameter of type `T : IDifferentiable` will become an `inout DifferentialPair<T>` parameter, where the original value part of the differential pair contains the original value of the parameter to pass into the back-prop function. The original value will not be overwritten by the backward propagation function. The propagated derivative will be written to the derivative part of the differential pair after the backward propagation function returns. The initial derivative value of the pair is ignored as input.
- A differentiable `out` parameter of type `T : IDifferentiable` will become an `in T.Differential` parameter, carrying the partial derivative of some downstream term with regard to the return value.
- A differentiable `inout` parameter of type `T : IDifferentiable` will become an `inout DifferentialPair<T>` parameter, where the original value of the argument, along with the downstream partial derivative with regard to the argument is passed as input to the backward propagation function as the original and derivative part of the pair. The propagated derivative with regard to this input parameter will be written back and replace the derivative part of the pair. The primal value part of the parameter will *not* be updated.
- A differentiable return value of type `R` will become an additional `in R.Differential` parameter at the end of the backward propagation function parameter list, carrying the result derivative of a downstream term with regard to the return value of the original function.
- A non-differentiable return value of type `NDR` will be dropped.
- A non-differentiable `in` parameter of type `ND` will remain unchanged in the backward propagation function.
- A non-differentiable `out` parameter of type `ND` will be removed from the parameter list of the backward propagation function.
- A non-differentiable `inout` parameter of type `ND` will become an `in ND` parameter.
- Types implemented `IDifferentiablePtrType` work the same was as the forward-mode case. They can only be used with `in` parameters, and are converted into `DifferentialPtrPair` types. Their directions are **not** affected.

For example consider the following original function:
```csharp
struct T : IDifferentiable {...}
struct R : IDifferentiable {...}
struct P : IDifferentiablePtrType {...}
struct ND {} // Non differentiable

[Differentiable]
R original(T p0, out T p1, inout T p2, ND p3, out ND p4, inout ND p5, P p6);
```
The signature of its backward propagation function is:
```csharp
void back_prop(
    inout DifferentialPair<T> p0,
    T.Differential p1,
    inout DifferentialPair<T> p2,
    ND p3,
    ND p5,
    DifferentialPtrPair<P> p6,
    R.Differential dResult);
```
Note that although `p2` is still `inout` in the backward propagation function, the backward propagation function will only write propagated derivative to `p2.d` and will not modify `p2.p`.

### Built-in Differentiable Functions

The following built-in functions are differentiable and both their forward and backward derivative functions are already defined in the standard library's core module:

- Arithmetic functions: `abs`, `max`, `min`, `sqrt`, `rcp`, `rsqrt`, `fma`, `mad`, `fmod`, `frac`, `radians`, `degrees`
- Interpolation and clamping functions: `lerp`, `smoothstep`, `clamp`, `saturate`
- Trigonometric functions: `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- Hyperbolic functions: `sinh`, `cosh`, `tanh`
- Exponential and logarithmic functions: `exp`, `exp2`, `pow`, `log`, `log2`, `log10`
- Vector functions: `dot`, `cross`, `length`, `distance`, `normalize`, `reflect`, `refract`
- Matrix transforms: `mul(matrix, vector)`, `mul(vector, matrix)`, `mul(matrix, matrix)`
- Matrix operations: `transpose`, `determinant`
- Legacy blending and lighting intrinsics: `dst`, `lit`