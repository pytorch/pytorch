public module core;

// Slang `core` library

// Aliases for base types
/// @category scalar_types Scalar types
typedef half float16_t;
/// @category scalar_types
typedef float float32_t;
/// @category scalar_types
typedef double float64_t;

/// @category scalar_types
typedef int int32_t;

/// @category scalar_types
typedef uint uint32_t;

/// @category scalar_types
typedef uintptr_t size_t;
/// @category scalar_types
typedef uintptr_t usize_t;
/// @category scalar_types
typedef intptr_t ssize_t;

// Modifier for variables that must resolve to compile-time constants
// as part of translation.
syntax constexpr : ConstExprModifier;

// Modifier for variables that should have writes be made
// visible at the global-memory scope
syntax globallycoherent : GloballyCoherentModifier;

/// Modifier to disable inteprolation and force per-vertex passing of a varying attribute.
///
/// When a varying attribute passed to the fragment shader is marked `pervertex`, it will
/// not be interpolated during rasterization (similar to `nointerpolate` attributes).
/// Unlike a plain `nointerpolate` attribute, this modifier indicates that the attribute
/// should *only* be acccessed through the `GetAttributeAtVertex()` operation, so access its
/// distinct per-vertex values.
///
syntax pervertex : PerVertexModifier;

/// Modifier to indicate a buffer or texture element type is
/// backed by data in an unsigned normalized format.
///
/// The `unorm` modifier is only valid on `float` and `vector`s
/// with `float` elements.
///
/// This modifier does not affect the semantics of any variable,
/// parameter, or field that uses it. The semantics of a `float`
/// or vector are the same with or without `unorm`.
///
/// The `unorm` modifier can be used for the element type of a
/// buffer or texture, to indicate that the data that is bound
/// to that buffer or texture is in a matching normalized format.
/// Some platforms may require a `unorm` qualifier for such buffers
/// and textures, and others may operate correctly without it.
///
syntax unorm : UNormModifier;

/// Modifier to indicate a buffer or texture element type is
/// backed by data in an signed normalized format.
///
/// The `snorm` modifier is only valid on `float` and `vector`s
/// with `float` elements.
///
/// This modifier does not affect the semantics of any variable,
/// parameter, or field that uses it. The semantics of a `float`
/// or vector are the same with or without `snorm`.
///
/// The `snorm` modifier can be used for the element type of a
/// buffer or texture, to indicate that the data that is bound
/// to that buffer or texture is in a matching normalized format.
/// Some platforms may require a `unorm` qualifier for such buffers
/// and textures, and others may operate correctly without it.
///
syntax snorm : SNormModifier;

/// Modifier to indicate that a function name should not be mangled
/// by the Slang compiler.
///
/// The `__extern_cpp` modifier makes a symbol to have unmangled
/// name in source/output C++ code.
///
syntax __extern_cpp : ExternCppModifier;

/// Represents types that can be initialized with a default constructor.
__magic_type(DefaultInitializableType)
interface IDefaultInitializable
{
    __builtin_requirement($( (int)BuiltinRequirementKind::DefaultInitializableConstructor))
    __init();
}


/// Represents types that can be compared.
/// Implemented by all builtin integer and floating point scalar or vector types.
interface IComparable
{
    /// Returns true if two values of the conforming type are equal.
    __builtin_requirement($( (int)BuiltinRequirementKind::Equals))
    bool equals(This other);

    /// Returns true if `this` is less than `other`.
    __builtin_requirement($( (int)BuiltinRequirementKind::LessThan))
    bool lessThan(This other);

    /// Returns true if `this` is less than or equal to `other`.
    __builtin_requirement($( (int)BuiltinRequirementKind::LessThanOrEquals))
    bool lessThanOrEquals(This other);
}

/// Represents types that has a defined range of values.
/// Implemented by all builtin integer and floating point types.
interface IRangedValue
{
    /// The maximum value that an instance of the type can hold.
    /// This is a constant value specific to the type.
    static const This maxValue;

    /// The minimum value that an instance of the type can hold.
    /// This is a constant value specific to the type.
    static const This minValue;
}

__attributeTarget(DeclBase)
attribute_syntax [TreatAsDifferentiable] : TreatAsDifferentiableAttribute;

/// Represents types that provide arithmetic operations.
interface IArithmetic : IComparable
{
    /// Adds two values of the conforming type.
    /// @param other The value to add to `this`.
    /// @return The sum of `this` and `other`.
    This add(This other);

    /// Subtracts one value of the conforming type from another.
    /// @param other The value to subtract from `this`.
    /// @return The difference of `this` and `other`.
    This sub(This other);

    /// Multiplies two values of the conforming type.
    /// @param other The value to multiply with `this`.
    /// @return The product of `this` and `other`.
    This mul(This other);

    /// Divides one value of the conforming type by another.
    /// @param other The value by which to divide `this`.
    /// @return The quotient of `this` divided by `other`.
    This div(This other);

    /// Computes the remainder of division of one value of the conforming type by another.
    /// @param other The divisor used to divide `this`.
    /// @return The remainder of `this` divided by `other`.
    This mod(This other);

    /// Negates a value of the conforming type.
    /// @return The negation of `this`.
    This neg();

    __init(int val);

        /// Initialize from the same type.
    __init(This value);
}

/// Represents types that provide logical operations.
interface ILogical : IComparable
{
    /// Shifts the bits of this value to the left by the specified number of positions.
    __builtin_requirement($( (int)BuiltinRequirementKind::Shl) )
    This shl(int value);

    /// Shifts the bits of this value to the right by the specified number of positions.
    __builtin_requirement($( (int)BuiltinRequirementKind::Shr) )
    This shr(int value);

    /// Performs a bitwise AND operation on this value with another value of the same type.
    __builtin_requirement($( (int)BuiltinRequirementKind::BitAnd) )
    This bitAnd(This other);

    /// Performs a bitwise OR operation on this value with another value of the same type.
    __builtin_requirement($( (int)BuiltinRequirementKind::BitOr) )
    This bitOr(This other);

    /// Performs a bitwise XOR operation on this value with another value of the same type.
    __builtin_requirement($( (int)BuiltinRequirementKind::BitXor) )
    This bitXor(This other);

    /// Performs a bitwise NOT operation on this value, flipping all bits.
    __builtin_requirement($( (int)BuiltinRequirementKind::BitNot) )
    This bitNot();

    /// Performs a logical AND operation on this value with another value of the same type.
    __builtin_requirement($( (int)BuiltinRequirementKind::And) )
    This and(This other);

    /// Performs a logical OR operation on this value with another value of the same type.
    __builtin_requirement($( (int)BuiltinRequirementKind::Or) )
    This or(This other);

    /// Performs a logical NOT operation on this value, returning the logical negation.
    __builtin_requirement($( (int)BuiltinRequirementKind::Not) )
    This not();

    __builtin_requirement($( (int)BuiltinRequirementKind::InitLogicalFromInt) )
    __init(int val);
}

/// Represents a type that can be used for integer arithmetic operations.
/// 
/// Implemented by builtin scalar types: `int`, `uint`, `int64_t`, `uint64_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`.
///
/// Also implemented by `vector<T, N>` where `T` is one of the above scalar types.
///
/// @remarks This interface can be used to define generic functions that work with integer-like types. See example below.
/// @example The following code defines a generic function that computes `a*b+1`, where `a`, `b` can be any integer scalar or vector types.
/// ```csharp
/// T compute<T:IInteger>(T a, T b)
/// {
///     return a * b + T(1);
/// }
/// 
/// RWStructuredBuffer<int> outputBuffer;
/// 
/// [numthreads(1,1,1)]
/// void test()
/// {
///     int a = 2;
///     int b = 3;
///     outputBuffer[0] = compute(a, b); // result = 2*3 + 1 = 7
/// 
///     int16_t2 a2 = int16_t2(2, 3);
///     int16_t2 b2 = int16_t2(4, 5);
///     // result2 = int16_t2(2*4 + 1, 3*5 + 1) = int16_t2(9, 16)
///     int16_t2 result2 = compute<int16_t2>(a2, b2);
///     outputBuffer[1] = result2.x;
/// }
/// ```
///
interface IInteger : IArithmetic, ILogical
{
    /// Converts the value of the current instance to an `int`.
    /// @return The converted `int` value.
    int toInt();

    /// Converts the value of the current instance to an `int64_t`.
    /// @return The converted `int64_t` value.
    int64_t toInt64();

    /// Converts the value of the current instance to an `uint`.
    /// @return The converted `uint` value.
    uint toUInt();

    /// Converts the value of the current instance to an `uint64_t`.
    /// @return The converted `uint64_t` value.
    uint64_t toUInt64();

    __init(int val);
    __init(int64_t val);
}

/// Represents a type that can be used for floating point arithmetic operations.
/// 
/// Implemented by builtin scalar types: `float`, `half`, `double`.
///
/// Also implemented by `vector<T, N>` where `T` is one of the above scalar types.
///
/// @remarks This interface can be used to define generic functions that work with floating-point-like types. See example below.
/// @example The following code defines a generic function that computes `a*b+1`, where `a`, `b` can be any floating point scalar or vector types.
/// ```csharp
/// T compute<T:IFloat>(T a, T b)
/// {
///     return a * b + T(1.0f);
/// }
/// 
/// RWStructuredBuffer<float> outputBuffer;
/// 
/// [numthreads(1,1,1)]
/// void test()
/// {
///     float a = 2.0;
///     float b = 3.0;
///     outputBuffer[0] = compute(a, b); // result = 2.0*3.0 + 1.0 = 7.0
/// 
///     half2 a2 = half2(2.0h, 3.0h);
///     half2 b2 = half2(4.0h, 5.0h);
///     // result2 = half2(2*4 + 1, 3*5 + 1) = half2(9, 16)
///     half2 result2 = compute(a2, b2);
///     outputBuffer[1] = result2.x;
/// }
/// ```
///
interface IFloat : IArithmetic, IDifferentiable
{
    /// Initializes a value of the conforming type form a `float`.
    [TreatAsDifferentiable]
    __init(float value);

    /// Converts a value of the conforming type into a `float`.
    [TreatAsDifferentiable]
    float toFloat();

    /// Computes the arithmetic sum of two values of the conforming type.
    [TreatAsDifferentiable]
    This add(This other);

    /// Computes the arithmetic subtraction from values of the conforming type.
    [TreatAsDifferentiable]
    This sub(This other);

    /// Computes the arithmetic multiplication from values of the conforming type.
    [TreatAsDifferentiable]
    This mul(This other);

    /// Computes the arithmetic division from values of the conforming type.
    [TreatAsDifferentiable]
    This div(This other);

    /// Computes the arithmetic remainder from values of the conforming type.
    [TreatAsDifferentiable]
    This mod(This other);

    /// Computes the arithmetic negation of the conforming type.
    [TreatAsDifferentiable]
    This neg();

    [TreatAsDifferentiable]
    __init(This value);

    /// Multiplies a value of the conforming type by a floating point scale factor..
    [TreatAsDifferentiable]
    This scale<T:__BuiltinFloatingPointType>(T scale);
}

/// A type that can be used as an operand for builtins
[sealed]
[builtin]
interface __BuiltinType 
{
}

/// A type that can be used for arithmetic operations. For defining generic functions that work with builtin scalar arithmetic types only.
/// Builtin types implementing this interface: `int`, `uint`, `int64_t`, `uint64_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `float`, `half`, `double`.
/// To define generic functions that work with both scalar and vector types, use `IInteger` or `IFloat` instead.
[sealed]
[builtin]
interface __BuiltinArithmeticType : __BuiltinType, IArithmetic
{
}

/// A type that can be used for logical/bitwise operations. For defining generic functions that work with builtin scalar logical types only.
/// Builtin types implementing this interface: `bool`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int`, `uint`, `int64_t`, `uint64_t`.
[sealed]
[builtin]
interface __BuiltinLogicalType : __BuiltinType, ILogical
{
}

/// Represents builtin types that logically have a sign (positive/negative/zero).
[sealed]
[builtin]
interface __BuiltinSignedArithmeticType : __BuiltinArithmeticType {}

/// Represents builtin types that can represent integers.
[sealed]
[builtin]
interface __BuiltinIntegerType : __BuiltinArithmeticType, IInteger
{}

/// Represents a `int` or `uint` type.
[sealed]
[builtin]
interface __BuiltinInt32Type : __BuiltinIntegerType
{}

/// Represents a `int64_t` or `uint64_t` type.
[sealed]
[builtin]
interface __BuiltinInt64Type : __BuiltinIntegerType
{}

/// Represents builtin types that can represent a real number.
[sealed]
[builtin]
interface __BuiltinRealType : __BuiltinSignedArithmeticType {}


__attributeTarget(AggTypeDecl)
attribute_syntax [__NonCopyableType] : NonCopyableTypeAttribute;

__attributeTarget(FunctionDeclBase)
attribute_syntax [__NoSideEffect] : NoSideEffectAttribute;

/// Marks a function as being differentiable in forward-mode.
///
/// See the user guide [section on forward-mode differentiation](https://shader-slang.org/slang/user-guide/autodiff.html#invoking-auto-diff-in-slang) for more
///
/// If used on a function that has a definition (i.e. a function body),Slang will use
/// automatic-differentiation to generate a forward-mode derivative of this function,
/// unless an implementation is provided by the user via `[ForwardDerivative(fn)]`
///
/// If used on an interface requirement, the signature of the requirement is modified to
/// include forward-differentiability. Any satisfying method must also be forward-differentiable,
/// or provide an appropriate derivative implementation.
/// See the user guide [section on auto-diff for interfaces](https://shader-slang.org/slang/user-guide/autodiff.html##using-auto-diff-with-interface-requirements-and-interface-types) for more 
/// 
__attributeTarget(FunctionDeclBase)
attribute_syntax [ForwardDifferentiable] : ForwardDifferentiableAttribute;

/// Marks a function as being differentiable for backward-mode auto-diff.
/// Note that in the current implementation, this implies that the method
/// is also forward differentiable.
/// 
/// This attribute is equivalent to using `[Differentiable]`
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [BackwardDifferentiable(order:int = 0)] : BackwardDifferentiableAttribute;

/// Marks a function as being differentiable for both
/// forward and backward mode auto-diff.
///
/// This attribute is equivalent to using `[Differentiable]`
///
/// See the user guide [section on auto-diff invocation](https://shader-slang.org/slang/user-guide/autodiff.html#invoking-auto-diff-in-slang) for more.
///
/// If used on a function that has a definition (i.e. a function body), Slang will use
/// automatic-differentiation to generate the derivative implementations for this function,
/// unless an implementation is provided by the user via `[ForwardDerivative(fn)]` and/or `[BackwardDerivative(fn)]`
///
/// If used on an interface requirement, the signature of the requirement is modified to
/// include differentiability. Any satisfying method must also be differentiable,
/// or provide appropriate derivative implementations.
/// See the user guide [section on auto-diff for interfaces](https://shader-slang.org/slang/user-guide/autodiff.html##using-auto-diff-with-interface-requirements-and-interface-types) for more
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [Differentiable(order:int = 0)] : BackwardDifferentiableAttribute;

__intrinsic_op($(kIROp_RequirePrelude))
void __requirePrelude(constexpr String preludeText);

__intrinsic_op($(kIROp_RequireTargetExtension))
void __requireTargetExtension(constexpr String preludeText);

/// @experimetal
/// Perform a compile-time condition check and emit a compile-time error if the condition is false.
/// @param condition The compile-time condition to check.
/// @param errorMessage The error message to emit if the condition is false.
__intrinsic_op($(kIROp_StaticAssert))
void static_assert(constexpr bool condition, NativeString errorMessage);

/// Represents a 'value' type that is differentiable for the purposes of automatic differentiation.
///
/// See the auto-diff user guide section for an introduction to 
/// differentiable value types (https://shader-slang.org/slang/user-guide/autodiff.html#differentiable-value-types)
///
/// #### Builtin Differentiable Value Types
/// The following built-in types are differentiable: 
/// - Scalars: `float`, `double` and `half`.
/// - Vector/Matrix: `vector` and `matrix` of `float`, `double` and `half` types.
/// - Arrays: `T[n]` is differentiable if `T` is differentiable.
/// - Tuples: `Tuple<each T>` is differentiable if `T` is differentiable. 
/// 
/// The `IDifferentiable` interface requires the following definitions (which can be auto-generated by the compiler for most scenarios)
/// ```csharp
/// interface IDifferentiable
/// {
///     associatedtype Differential : IDifferentiable
///         where Differential.Differential == Differential;
/// 
///     static Differential dzero();
/// 
///     static Differential dadd(Differential, Differential);
/// }
/// ```
/// 
/// As defined by the `IDifferentiable` interface, a differentiable type must have a 
/// `Differential` associated type that stores the derivative of the value. 
/// A further requirement is that the type of the second-order derivative must be the same 
/// `Differential` type. In another words, given a type `T`, `T.Differential` can be different 
/// from `T`, but `T.Differential.Differential` must equal to `T.Differential`.
///
/// In addition, a differentiable type must define the `zero` value of its derivative, 
/// and how to add two derivative values together. These function are used during reverse-mode 
/// auto-diff, to initialize and accumulate derivatives of the given type.
///
/// #### Automatic Fulfillment of `IDifferentiable` Requirements
/// Assume the user has defined the following type:
///
/// ```csharp
/// struct MyRay
/// {
///     float3 origin;
///     float3 dir;
///     int nonDifferentiablePayload;
/// }
/// ```
///
/// The type can be made differentiable by adding `IDifferentiable` conformance:
/// ```csharp
/// struct MyRay : IDifferentiable
/// {
///     float3 origin;
///     float3 dir;
///     int nonDifferentiablePayload;
/// }
/// ```
///
/// Note that this code does not provide any explicit implementation of the `IDifferentiable` requirements. In this case the compiler will automatically synthesize all the requirements. This should provide the desired behavior most of the time. The procedure for synthesizing the interface implementation is as follows:
/// 1. A new type is generated that stores the `Differential` of all differentiable fields. This new type itself will conform to the `IDifferentiable` interface, and it will be used to satisfy the `Differential` associated type requirement.
/// 2. Each differential field will be associated to its corresponding field in the newly synthesized `Differential` type.
/// 3. The `zero` value of the differential type is made from the `zero` value of each field in the differential type.
/// 4. The `dadd` method invokes the `dadd` operations for each field whose type conforms to `IDifferentiable`.
/// 5. If the synthesized `Differential` type contains exactly the same fields as the original type, and the type of each field is the same as the original field type, then the original type itself will be used as the `Differential` type instead of creating a new type to satisfy the `Differential` associated type requirement. This means that all the synthesized `Differential` type use itself to meet its own `IDifferentiable` requirements.
///
/// #### Manual fulfilment of `IDifferentiable` requirements
/// In rare cases where more control is desired, the user can manually provide the implementation. 
/// To do so, we will first define the `Differential` type for `MyRay`, and use it to fulfill 
/// the `Differential` requirement in `MyRay`:
/// ```csharp
/// struct MyRayDifferential
/// {
///     float3 d_origin;
///     float3 d_dir;
/// }
///
/// struct MyRay : IDifferentiable
/// {
///     // Specify that `MyRay.Differential` is `MyRayDifferential`.
///     typealias Differential = MyRayDifferential;
///
///     // Specify that the derivative for `origin` will be stored in `MayRayDifferential.d_origin`.
///     [DerivativeMember(MayRayDifferential.d_origin)]
///     float3 origin;
///
///     // Specify that the derivative for `dir` will be stored in `MayRayDifferential.d_dir`.
///     [DerivativeMember(MayRayDifferential.d_dir)]
///     float3 dir;
///
///     // This is a non-differentiable field so we don't put any attributes on it.
///     int nonDifferentiablePayload;
///
///     // Define zero derivative.
///     static MyRayDifferential dzero()
///     {
///         return {float3(0.0), float3(0.0)};
///     }
///
///     // Define the add operation of two derivatives.
///     static MyRayDifferential dadd(MyRayDifferential v1, MyRayDifferential v2)
///     {
///         MyRayDifferential result;
///         result.d_origin = v1.d_origin + v2.d_origin;
///         result.d_dir = v1.d_dir + v2.d_dir;
///         return result;
///     }
/// }
/// ```
/// 
/// Note that for each struct field that is differentiable, we need to use the `[DerivativeMember]` attribute to associate it with the
/// corresponding field in the `Differential` type, so the compiler knows how to access the derivative for the field.
///
/// However, there is still a missing piece in the above code: we also need to make `MyRayDifferential` conform to `IDifferentiable` because it is required that the `Differential` of a type must itself be `Differential`. Again we can use automatic fulfillment by simply adding `IDifferentiable` conformance to `MyRayDifferential`:
/// ```csharp
/// struct MyRayDifferential : IDifferentiable
/// {
///     float3 d_origin;
///     float3 d_dir;
/// }
/// ```
/// In this case, since all fields in `MyRayDifferential` are differentiable, and the `Differential` of each field is the same as the original type of each field (i.e. `float3.Differential == float3` as defined in the core module), the compiler will automatically use the type itself as its own `Differential`, making `MyRayDifferential` suitable for use as `Differential` of `MyRay`.
///
/// We can also choose to manually implement `IDifferentiable` interface for `MyRayDifferential` as in the following code:
///
/// ```csharp
/// struct MyRayDifferential : IDifferentiable
/// {
///     typealias Differential = MyRayDifferential;
///
///    [DerivativeMember(MyRayDifferential.d_origin)]
///    float3 d_origin;
///
///    [DerivativeMember(MyRayDifferential.d_dir)]
///    float3 d_dir;
///
///    static MyRayDifferential dzero()
///    {
///        return {float3(0.0), float3(0.0)};
///    }
///
///    static MyRayDifferential dadd(MyRayDifferential v1, MyRayDifferential v2)
///    {
///        MyRayDifferential result;
///        result.d_origin = v1.d_origin + v2.d_origin;
///         result.d_dir = v1.d_dir + v2.d_dir;
///         return result;
///     }
/// }
/// ```
/// In this specific case, the automatically generated `IDifferentiable` 
/// implementation will be exactly the same as the manually written code listed above.
///
/// 
__magic_type(DifferentiableType)
[KnownBuiltin("IDifferentiable")]
interface IDifferentiable
{
    // Note: the compiler implementation requires the `Differential` associated type to be defined
    // before anything else.

    __builtin_requirement($( (int)BuiltinRequirementKind::DifferentialType) )
    associatedtype Differential : IDifferentiable;

    /// Returns a zero-initialized value of the differential type.
    __builtin_requirement($( (int)BuiltinRequirementKind::DZeroFunc) )
    static Differential dzero();

    /// Adds two differential values and returns the result.
    __builtin_requirement($( (int)BuiltinRequirementKind::DAddFunc) )
    static Differential dadd(Differential, Differential);

    /// Multiplies a scalar value of a built-in real type with a differential value and returns the result.
    __builtin_requirement($( (int)BuiltinRequirementKind::DMulFunc) )
    __generic<T : __BuiltinRealType>
    static Differential dmul(T, Differential);
};

/// @experimental
///
/// The `IDifferentiablePtrType` interface requires the following definitions.
///
/// ```csharp
/// interface IDifferentiablePtrType
/// {
///     associatedtype Differential : IDifferentiablePtrType
///         where Differential.Differential == Differential;
/// }
/// ```
///
/// Types that conform to this interface can be used with `DifferentialPtrPair<T>`
/// to pass the derivative components to calls to `fwd_diff(fn)` or `bwd_diff(fn)`
///
/// See the auto-diff user guide for more details (https://shader-slang.org/slang/user-guide/autodiff.html#differentiable-ptr-types)
///
/// @remarks Support for this interface is still experimental and subject to change.
///
__magic_type(DifferentiablePtrType)
[KnownBuiltin("IDifferentiablePtr")]
interface IDifferentiablePtrType
{
    __builtin_requirement($( (int)BuiltinRequirementKind::DifferentialPtrType) )
    associatedtype Differential : IDifferentiablePtrType;
};


/// `DifferentialPair<T>` is a built-in type that carries both the original and derivative value of a term. 
/// It is defined as follows:
/// ```csharp
/// struct DifferentialPair<T : IDifferentiable> : IDifferentiable
/// {
///     typealias Differential = DifferentialPair<T.Differential>;
///     property T p {get;}
///     property T.Differential d {get;}
///     static Differential dzero();
///     static Differential dadd(Differential a, Differential b);
/// }
/// ```
/// 
/// Differential pairs can be created via constructor or through the `diffPair()` operation 
/// ```csharp
/// DifferentialPair<float> dpa = DifferentialPair<float>(1.0f, 2.0f);
/// DifferentialPair<float> dpa = diffPair(1.0f, 2.0f);
/// ```
/// Note that derivative pairs are used to pass derivatives into and out of auto-diff functions.
/// See documentation on `fwd_diff` and `bwd_diff` operators for more information.
///
__generic<T : IDifferentiable>
__magic_type(DifferentialPairType)
__intrinsic_type($(kIROp_DifferentialPairUserCodeType))
struct DifferentialPair : IDifferentiable
{
    typedef DifferentialPair<T.Differential> Differential;
    typedef T.Differential DifferentialElementType;

    __intrinsic_op($(kIROp_MakeDifferentialPairUserCode))
    __init(T _primal, T.Differential _differential);

    property p : T
    {
        __intrinsic_op($(kIROp_DifferentialPairGetPrimalUserCode))
        get;
    }

    property v : T
    {
        __intrinsic_op($(kIROp_DifferentialPairGetPrimalUserCode))
        get;
    }

    property d : T.Differential
    {
        __intrinsic_op($(kIROp_DifferentialPairGetDifferentialUserCode))
        get;
    }

    [__unsafeForceInlineEarly]
    T.Differential getDifferential()
    {
        return d;
    }

    [__unsafeForceInlineEarly]
    T getPrimal()
    {
        return p;
    }

    [__unsafeForceInlineEarly]
    static Differential dzero()
    {
        return Differential(T.dzero(), T.Differential.dzero());
    }

    [__unsafeForceInlineEarly]
    static Differential dadd(Differential a, Differential b)
    {
        return Differential(
            T.dadd(
                a.p,
                b.p
            ),
            T.Differential.dadd(a.d, b.d));
    }

    __generic<U : __BuiltinRealType>
    [__unsafeForceInlineEarly]
    static Differential dmul(U a, Differential b)
    {
        return Differential(
            T.dmul<U>(a, b.p),
            T.Differential.dmul<U>(a, b.d));
    }
};

/// @experimental
/// `DifferentialPtrPair<T>` is a built-in type that carries both the original and differential of a 
/// pointer-like object. 
/// `T` must conform to `IDifferentiablePtrType`
/// 
/// It is defined as follows:
/// ```csharp
/// struct DifferentialPtrPair<T : IDifferentiablePtrType> : IDifferentiablePtrType
/// {
///     typealias Differential = DifferentialPtrPair<T.Differential>;
///     property T p {get;}
///     property T.Differential d {get;}
/// }
/// ```
/// @remarks
/// Differential ptr pairs can be created via constructor.
/// ```csharp
/// struct DPtrFloat : IDifferentialPtrType 
/// { 
///     typealias Differential = DPtrFloat;
///     float* ptr;
/// };
/// 
/// DifferentialPtrPair<DPtrFloat> dpa = 
///            DifferentialPtrPair<float>({&outputBuffer[0]}, {&outputBuffer[1]});
/// ```
/// Note that derivative ptr pairs are used to pass derivatives into and out of auto-diff functions.
/// See documentation on `fwd_diff` and `bwd_diff` operators for more information.
///
/// 
__generic<T : IDifferentiablePtrType>
__magic_type(DifferentialPtrPairType)
__intrinsic_type($(kIROp_DifferentialPtrPairType))
struct DifferentialPtrPair : IDifferentiablePtrType
{
    typedef DifferentialPtrPair<T.Differential> Differential;
    typedef T.Differential DifferentialElementType;

    __intrinsic_op($(kIROp_MakeDifferentialPtrPair))
    __init(T _primal, T.Differential _differential);

    property p : T
    {
        __intrinsic_op($(kIROp_DifferentialPtrPairGetPrimal))
        get;
    }

    property v : T
    {
        __intrinsic_op($(kIROp_DifferentialPtrPairGetPrimal))
        get;
    }

    property d : T.Differential
    {
        __intrinsic_op($(kIROp_DifferentialPtrPairGetDifferential))
        get;
    }
};


/// Represents builtin floating point scalar types.
/// To define generic functions that work with both scalar and vector types, use `IFloat` instead.
///
/// Implemented by `float`, `half` and `double` types.
[sealed]
[builtin]
[TreatAsDifferentiable]
interface __BuiltinFloatingPointType : __BuiltinRealType, IFloat
{
        /// Get the value of the mathematical constant pi in this type.
    [Differentiable]
    static This getPi();
}

//@ hidden:

// A type resulting from an `enum` declaration.
[builtin]
__magic_type(EnumTypeType)
interface __EnumType : ILogical
{
    // The type of tags for this `enum`
    //
    // Note: using `__Tag` instead of `Tag` to avoid any
    // conflict if a user had an `enum` case called `Tag`
    associatedtype __Tag : __BuiltinIntegerType;

    __builtin_requirement($( (int)BuiltinRequirementKind::InitLogicalFromInt) )
    __init(__Tag value);
};


//@public:

/// Represents types that provide a subscript operator so that they can be used like an immutable array.
/// @param T The element type returned by the subscript operator.
/// @remarks This interface is implemented by `Array`, `vector`, `matrix`, `StructuredBuffer` and `RWStructuredBuffer` types.
/// @example The follow example shows how to define a generic function that computes the sum of all elements in an array-like type.
/// ```csharp
/// T sum<T:IFloat, U:IArray<T>>(U array)
/// {
///     T result = T(0);
///     for (int i = 0; i < array.getCount(); i++)
///     {
///         result = result + array[i];
///     }
///     return result;
/// }
/// 
/// RWStructuredBuffer<float> outputBuffer;
/// 
/// [numthreads(1, 1, 1)]
/// void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
/// {
///     float arr[3] = { 1.0, 2.0, 3.0 };
///     float4 v = float4(1.0, 2.0, 3.0, 4.0);
///     float2x2 m = float2x2(1.0, 2.0, 3.0, 4.0);
///
///     outputBuffer[0] = sum(arr);  // 6.0
///
///     // treat `v` as array of `float`.
///     outputBuffer[1] = sum(v);    // 10.0
///
///     // treat `m` as array of `float2`.
///     float2 innerSum = sum(m); 
///     outputBuffer[2] = sum(innerSum); // 10.0
/// }
/// ```
interface IArray<T>
{
    /// Returns the number of elements in the conforming type.
    int getCount();

    /// The subscript operator to be provided by a conforming type.
    __subscript(int index) -> T
    {
        get;
    }
}

/// Represents types that provide a subscript operator so that they can be used like a mutable array.
/// @param T The element type returned by the subscript operator.
/// @remarks This interface is implemented by `Array`, `vector`, `matrix`, `StructuredBuffer` and `RWStructuredBuffer` types.
/// @example The follow example shows how to define a generic function uses the `IRWArray` interface to mutate an array-like value.
/// ```csharp
/// void writeToArray<U, T : IRWArray<U>>(inout T array, int index, U value) { array[index] = value; }
/// void writeToBuffer<U, T : IRWArray<U>>(T array, int index, U value) { array[index] = value; }
/// U readFromArray<U, T:IArray<U>>(T array, int index) { return array[index]; }
/// 
/// //TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
/// RWStructuredBuffer<float> outputBuffer;
/// 
/// [numthreads(1, 1, 1)]
/// void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
/// {
///     float arr[3] = { 1.0, 2.0, 3.0 };
///     float4 v = float4(1.0, 2.0, 3.0, 4.0);
///     float2x2 m = float2x2(1.0, 2.0, 3.0, 4.0);
/// 
///     // treat `outputBuffer` as a mutable array of `float`.
///     writeToBuffer(outputBuffer, 0, 1.0f);
/// 
///     // treat `arr` as a mutable array of `float`.
///     writeToArray(arr, 0, 4.0f);
///     outputBuffer[1] = readFromArray(arr, 0); // 4.0
/// 
///     // treat `v` as a mutable array of `float`.
///     writeToArray(v, 3, 3.0f);
///     outputBuffer[2] = readFromArray(v, 3);  // 3.0
/// 
///     // treat `m` as a mutable array of `float2`.
///     writeToArray(m, 1, float2(10.0f, 20.0f));
///     outputBuffer[3] = readFromArray(m, 1).x + readFromArray(m, 1).y; // 30.0
/// 
///     writeToBuffer(outputBuffer, 0, readFromArray(outputBuffer, 0));
/// }
/// ```
interface IRWArray<T> : IArray<T>
{
    /// The subscript operator to be provided by a conforming type. Provides both a `get` and a `set` accessor.
    __subscript(int index)->T
    {
        get;
        set;
    }
}

// The "comma operator" is effectively just a generic function that returns its second
// argument. The left-to-right evaluation order guaranteed by Slang then ensures that
// `left` is evaluated before `right`.
//
//@hidden:
__generic<T,U>
[__unsafeForceInlineEarly]
U operator,(T left, U right)
{
    return right;
}

// The ternary `?:` operator does not short-circuit in HLSL, and Slang no longer
// follow that definition for the scalar condition overload, so this declaration just serves
// for type-checking purpose only.

//@hidden:
__generic<T> __intrinsic_op(select) T operator?: (bool condition, T ifTrue, T ifFalse);

//@hidden:
__generic<T, let N : int> __intrinsic_op(select) vector<T,N> operator?:(vector<bool,N> condition, vector<T,N> ifTrue, vector<T,N> ifFalse);

// Users are advised to use `select` instead if non-short-circuiting behavior is intended.
//@public:
__generic<T> __intrinsic_op(select) T select(bool condition, T ifTrue, T ifFalse);
__generic<T, let N : int> __intrinsic_op(select) vector<T,N> select(vector<bool,N> condition, vector<T,N> ifTrue, vector<T,N> ifFalse);

[ForceInline]
__generic<T> Optional<T> select(bool condition, __none_t ifTrue, T ifFalse)
{
    return select(condition, Optional<T>(none), Optional<T>(ifFalse));
}

[ForceInline]
__generic<T> Optional<T> select(bool condition, T ifTrue, __none_t ifFalse)
{
    return select(condition, Optional<T>(ifTrue), Optional<T>(none));
}

// Allow real-number types to be cast into each other
//@hidden:
__intrinsic_op($(kIROp_FloatCast))
    T __realCast<T : __BuiltinRealType, U : __BuiltinRealType>(U val);
//@hidden:
__intrinsic_op($(kIROp_CastIntToFloat))
T __realCast<T : __BuiltinRealType, U : __BuiltinIntegerType>(U val);
//@hidden:
__intrinsic_op($(kIROp_IntCast))
T __intCast<T : __BuiltinType, U : __BuiltinType>(U val);

//@hidden:
__intrinsic_op($(kIROp_CastIntToFloat))
T __castIntToFloat<T:__BuiltinType, U:__BuiltinType>(U v);

//@hidden:
${{{{
// We are going to use code generation to produce the
// declarations for all of our base types.
static const int kBaseTypeCount = sizeof(kBaseTypes) / sizeof(kBaseTypes[0]);
for (int tt = 0; tt < kBaseTypeCount; ++tt)
{
}}}}

__builtin_type($(int(kBaseTypes[tt].tag)))
struct $(kBaseTypes[tt].name)
    : __BuiltinType

${{{{
    switch (kBaseTypes[tt].tag)
    {
    case BaseType::Half:
    case BaseType::Float:
    case BaseType::Double:
}}}}
    ,  __BuiltinFloatingPointType
    ,  __BuiltinRealType
    ,  __BuiltinSignedArithmeticType
    ,  __BuiltinArithmeticType
${{{{
        break;
    case BaseType::Int8:
    case BaseType::Int16:
    case BaseType::Int:
    case BaseType::Int64:
    case BaseType::IntPtr:
}}}}
    ,  __BuiltinSignedArithmeticType
${{{{
        ; // fall through
    case BaseType::UInt8:
    case BaseType::UInt16:
    case BaseType::UInt:
    case BaseType::UInt64:
    case BaseType::UIntPtr:
}}}}
    ,  __BuiltinArithmeticType
    ,  __BuiltinIntegerType
${{{{
    if (kBaseTypes[tt].tag == BaseType::Int || kBaseTypes[tt].tag == BaseType::UInt)
}}}}
    ,  __BuiltinInt32Type
${{{{
    if (kBaseTypes[tt].tag == BaseType::Int64 || kBaseTypes[tt].tag == BaseType::UInt64)
}}}}
    ,  __BuiltinInt64Type
${{{{
        ; // fall through
    case BaseType::Bool:
}}}}
    ,  __BuiltinLogicalType
${{{{
        break;

    default:
        break;
    }
}}}}
{
${{{{
    // Declare initializers to convert from various other types
    for (int ss = 0; ss < kBaseTypeCount; ++ss)
    {
        // Don't allow conversion to or from `void`
        if (kBaseTypes[tt].tag == BaseType::Void)
            continue;
        if (kBaseTypes[ss].tag == BaseType::Void)
            continue;

        // We need to emit a modifier so that the semantic-checking
        // layer will know it can use these operations for implicit
        // conversion.
        ConversionCost conversionCost = getBaseTypeConversionCost(
            kBaseTypes[tt],
            kBaseTypes[ss]);

        IROp intrinsicOpCode = getBaseTypeConversionOp(
            kBaseTypes[tt],
            kBaseTypes[ss]);

        BuiltinConversionKind builtinConversionKind = kBuiltinConversion_Unknown;
        if (kBaseTypes[tt].tag == BaseType::Double &&
            kBaseTypes[ss].tag == BaseType::Float)
            builtinConversionKind = kBuiltinConversion_FloatToDouble;

        const char* attrib = "";
        if ((kBaseTypes[tt].flags & kBaseTypes[ss].flags & FLOAT_MASK) != 0)
            attrib = "[TreatAsDifferentiable]";
}}}}
    $(attrib)
    __intrinsic_op($(intrinsicOpCode))
    __implicit_conversion($(conversionCost), $(builtinConversionKind))
    __init($(kBaseTypes[ss].name) value);

${{{{
    }

    // Integer type implementations.
    switch (kBaseTypes[tt].tag)
    {
    case BaseType::Bool:
}}}}
        [__unsafeForceInlineEarly] __intrinsic_op($(kIROp_Eql)) bool equals(This other);
        [__unsafeForceInlineEarly] __intrinsic_op($(kIROp_Less)) bool lessThan(This other);
        [__unsafeForceInlineEarly] __intrinsic_op($(kIROp_Leq)) bool lessThanOrEquals(This other);
        [__unsafeForceInlineEarly] This shl(int other) { return __intCast<This>(__shl(__intCast<int>(this), other)); }
        [__unsafeForceInlineEarly] This shr(int other) { return __intCast<This>(__shr(__intCast<int>(this), other)); }
        [__unsafeForceInlineEarly] This bitAnd(This other) { return __intCast<This>(__and(__intCast<int>(this), __intCast<int>(other))); }
        [__unsafeForceInlineEarly] This bitOr(This other) { return __intCast<This>(__or(__intCast<int>(this), __intCast<int>(other))); }
        [__unsafeForceInlineEarly] __intrinsic_op($(kIROp_And)) This and(This other);
        [__unsafeForceInlineEarly] __intrinsic_op($(kIROp_Or)) This or(This other);
        [__unsafeForceInlineEarly] This bitXor(This other)  { return __intCast<This>(__xor(__intCast<int>(this), __intCast<int>(other))); }
        [__unsafeForceInlineEarly] This bitNot()  { return __intCast<This>(__not(__intCast<int>(this))); }
        [__unsafeForceInlineEarly] __intrinsic_op($(kIROp_Not)) This not();
${{{{
        break;
    case BaseType::UInt8:
    case BaseType::UInt16:
    case BaseType::UInt:
    case BaseType::UInt64:
    case BaseType::Int8:
    case BaseType::Int16:
    case BaseType::Int:
    case BaseType::Int64:
    case BaseType::IntPtr:
    case BaseType::UIntPtr:
}}}}
        // If this is a basic integer type, then define explicit
        // initializers that take a value of an `enum` type.
        //
        // TODO: This should actually be restricted, so that this
        // only applies `where T.__Tag == Self`, but we don't have
        // the needed features in our type system to implement
        // that constraint right now.
        //
        __generic<T:__EnumType>
        __intrinsic_op($(kIROp_IntCast))
        __init(T value);

        // Implementation of the `IInteger` interface.
        __intrinsic_op($(kIROp_Less)) bool lessThan(This other);
        __intrinsic_op($(kIROp_Leq))  bool lessThanOrEquals(This other);
        __intrinsic_op($(kIROp_Eql))  bool equals(This other);
        __intrinsic_op($(kIROp_Add))  This add(This other);
        __intrinsic_op($(kIROp_Sub))  This sub(This other);
        __intrinsic_op($(kIROp_Mul))  This mul(This other);
        __intrinsic_op($(kIROp_Div))  This div(This other);
        __intrinsic_op($(kIROp_IRem)) This mod(This other);
        __intrinsic_op($(kIROp_Neg))  This neg();
        __intrinsic_op($(kIROp_Lsh)) This shl(int other);
        __intrinsic_op($(kIROp_Rsh)) This shr(int other);
        __intrinsic_op($(kIROp_BitAnd)) This bitAnd(This other);
        __intrinsic_op($(kIROp_BitOr)) This bitOr(This other);
        [__unsafeForceInlineEarly] This and(This other) {return __intCast<This>(and(__intCast<bool>(this), __intCast<bool>(other))); }
        [__unsafeForceInlineEarly] This or(This other) {return __intCast<This>(__intCast<bool>(this) || __intCast<bool>(other)); }
        __intrinsic_op($(kIROp_BitXor)) This bitXor(This other);
        __intrinsic_op($(kIROp_BitNot)) This bitNot();
        [__unsafeForceInlineEarly] This not() {return __intCast<This>(!__intCast<bool>(this)); }
        __intrinsic_op($(kIROp_IntCast)) int toInt();
        __intrinsic_op($(kIROp_IntCast)) int64_t toInt64();
        __intrinsic_op($(kIROp_IntCast)) uint toUInt();
        __intrinsic_op($(kIROp_IntCast)) uint64_t toUInt64();
${{{{
        break;

    default:
        break;
    }

    // If this is a floating-point type, then we need to
    // implement the `IFloat` interface, which defines the basic `getPi()`
    // function that is used to implement generic versions of `degrees()` and
    // `radians()`.
    //
    switch (kBaseTypes[tt].tag)
    {
    default:
        break;
    case BaseType::Half:
    case BaseType::Float:
    case BaseType::Double:
}}}}
        [Differentiable]
        static $(kBaseTypes[tt].name) getPi() { return $(kBaseTypes[tt].name)(3.14159265358979323846264338328); }

         __intrinsic_op($(kIROp_Less)) bool lessThan(This other);
         __intrinsic_op($(kIROp_Leq))  bool lessThanOrEquals(This other);
         __intrinsic_op($(kIROp_Eql))  bool equals(This other);
         __intrinsic_op($(kIROp_Add))  This add(This other);
         __intrinsic_op($(kIROp_Sub))  This sub(This other);
         __intrinsic_op($(kIROp_Mul))  This mul(This other);
         __intrinsic_op($(kIROp_Div))  This div(This other);
         __intrinsic_op($(kIROp_FRem)) This mod(This other);
         __intrinsic_op($(kIROp_Neg))  This neg();
         __intrinsic_op($(kIROp_FloatCast)) float toFloat();
         __intrinsic_op($(kIROp_Mul)) This scale<T:__BuiltinFloatingPointType>(T s) { return __mul(this, __realCast<This>(s)); }
        typedef $(kBaseTypes[tt].name) Differential;
    
        [__unsafeForceInlineEarly]
        [BackwardDifferentiable]
        static Differential dzero()
        {
            return Differential(0);
        }

        [__unsafeForceInlineEarly]
        [BackwardDifferentiable]
        static Differential dadd(Differential a, Differential b)
        {
            return a + b;
        }
        
        __generic<U : __BuiltinRealType>
        [__unsafeForceInlineEarly]
        [BackwardDifferentiable]
        static Differential dmul(U a, Differential b)
        {
            return __realCast<Differential, U>(a) * b;
        }
${{{{
        break;
    }

    // If this is the `void` type, then we want to allow
    // explicit conversion to it from any other type, using
    // `(void) someExpression`.
    //
    if( kBaseTypes[tt].tag == BaseType::Void )
    {
}}}}
        __generic<T>
        [__readNone]
        __intrinsic_op($(kIROp_CastToVoid))
        __init(T value)
        {}
${{{{
    }

}}}}

}

${{{{
}

// Declare built-in pointer type
// (eventually we can have the traditional syntax sugar for this)
}}}}

//@hidden:
__magic_type(NullPtrType)
struct NullPtr
{
};

//@hidden:
__magic_type(NoneType)
__intrinsic_type($(kIROp_VoidType))
struct __none_t
{
};

//@public:
/// Represents a pointer type.
/// @param T The type of the value pointed to.
/// @remarks `T* val` is equivalent to `Ptr<T> val`.
__generic<T, let addrSpace : uint64_t = $( (uint64_t)AddressSpace::UserPointer)ULL>
__magic_type(PtrType)
__intrinsic_type($(kIROp_PtrType))
struct Ptr
{
    __generic<U>
    __intrinsic_op($(kIROp_BitCast))
    __init(Ptr<U, addrSpace> ptr);

    __intrinsic_op($(kIROp_CastIntToPtr))
    __init(uint64_t val);

    __intrinsic_op($(kIROp_CastIntToPtr))
    __init(int64_t val);

    __generic<TInt : __BuiltinIntegerType>
    __subscript(TInt index) -> T
    {
        // If a 'Ptr[index]' is referred to by a '__ref', call 'kIROp_GetOffsetPtr(index)'
        __intrinsic_op($(kIROp_GetOffsetPtr))
        [nonmutating]
        ref;
    }
};

//@hidden:
__intrinsic_op($(kIROp_AlignedAttr))
void __align_attr(int alignment);

__intrinsic_op($(kIROp_Load))
T __load_aligned<T, U>(T* ptr, U alignmentAttr);

__intrinsic_op($(kIROp_Store))
void __store_aligned<T, U>(T* ptr, T value, U alignmentAttr);

//@public:

/// Load a value from a pointer with a known alignment.
/// Aligned loads are more efficient than unaligned loads on some platforms.
/// @param alignment The alignment of the load operation.
/// @param ptr The pointer to load from.
/// @return The value loaded from the pointer.
/// @remarks When targeting SPIRV, this function maps to an `OpLoad` instruction with the `Aligned` memory operand.
/// The functions maps to normal load operation on other targets.
///
[__NoSideEffect]
[ForceInline]
T loadAligned<int alignment, T>(T* ptr)
{
    return __load_aligned(ptr, __align_attr(alignment));
}

/// Store a value to a pointer with a known alignment.
/// Aligned stores are more efficient than unaligned stores on some platforms.
/// @param alignment The alignment of the store operation.
/// @param ptr The pointer to store value to.
/// @param value The value to store.
/// @remarks When targeting SPIRV, this function maps to an `OpStore` instruction with the `Aligned` memory operand.
/// The functions maps to normal store operation on other targets.
///
[ForceInline]
void storeAligned<int alignment, T>(T* ptr, T value)
{
    __store_aligned(ptr, value, __align_attr(alignment));
}

//@hidden:
__intrinsic_op($(kIROp_Load))
T __load<T, let addrSpace : uint64_t>(Ptr<T, addrSpace> ptr);

__intrinsic_op($(kIROp_Store))
void __store<T, let addrSpace : uint64_t>(Ptr<T, addrSpace> ptr, T val);

__intrinsic_op($(kIROp_GetElementPtr))
Ptr<T, addrSpace> __getElementPtr<T, let addrSpace : uint64_t, TIndex : __BuiltinIntegerType>(Ptr<T, addrSpace> ptr, TIndex index);

__intrinsic_op($(kIROp_GetOffsetPtr))
Ptr<T, addrSpace> __getOffsetPtr<T, let addrSpace : uint64_t, TIndex : __BuiltinIntegerType>(Ptr<T, addrSpace> ptr, TIndex index);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_Less))
bool operator <(Ptr<T, addrSpace> p1, Ptr<T, addrSpace> p2);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_Leq))
bool operator <=(Ptr<T, addrSpace> p1, Ptr<T, addrSpace> p2);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_Greater))
bool operator>(Ptr<T, addrSpace> p1, Ptr<T, addrSpace> p2);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_Geq))
bool operator >=(Ptr<T, addrSpace> p1, Ptr<T, addrSpace> p2);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_Neq))
bool operator !=(Ptr<T, addrSpace> p1, Ptr<T, addrSpace> p2);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_Eql))
bool operator ==(Ptr<T, addrSpace> p1, Ptr<T, addrSpace> p2);

//@public:
extension bool : IRangedValue
{
    __generic<T, let addrSpace : uint64_t>
    __implicit_conversion($(kConversionCost_PtrToBool))
    __intrinsic_op($(kIROp_CastPtrToBool))
    __init(Ptr<T, addrSpace> ptr);

    __generic<T : __EnumType>
    __implicit_conversion($(kConversionCost_IntegerTruncate))
    [__unsafeForceInlineEarly]
    __init(T v)
    {
        return __slang_noop_cast<T.__Tag>(v) != __intCast<T.__Tag>(0);
    }

    static const bool maxValue = true;
    static const bool minValue = false;
}

extension uint64_t : IRangedValue
{
    __generic<T, let addrSpace : uint64_t>
    __intrinsic_op($(kIROp_CastPtrToInt))
    __init(Ptr<T, addrSpace> ptr);

    static const uint64_t maxValue = 0xFFFFFFFFFFFFFFFFULL;
    static const uint64_t minValue = 0;
}

extension int64_t : IRangedValue
{
    __generic<T, let addrSpace : uint64_t>
    __intrinsic_op($(kIROp_CastPtrToInt))
    __init(Ptr<T, addrSpace> ptr);

    static const int64_t maxValue =  0x7FFFFFFFFFFFFFFFLL;
    static const int64_t minValue = -0x8000000000000000LL;
}

extension intptr_t : IRangedValue
{
    __generic<T, let addrSpace : uint64_t>
    __intrinsic_op($(kIROp_CastPtrToInt))
    __init(Ptr<T, addrSpace> ptr);
    static const intptr_t maxValue = $(SLANG_PROCESSOR_X86_64?"0x7FFFFFFFFFFFFFFFz":"0x7FFFFFFFz");
    static const intptr_t minValue = $(SLANG_PROCESSOR_X86_64?"0x8000000000000000z":"0x80000000z");
    static const int size = $(SLANG_PROCESSOR_X86_64?"8":"4");
}

extension uintptr_t : IRangedValue
{
    __generic<T, let addrSpace : uint64_t>
    __intrinsic_op($(kIROp_CastPtrToInt))
    __init(Ptr<T, addrSpace> ptr);
    static const uintptr_t maxValue = $(SLANG_PROCESSOR_X86_64?"0xFFFFFFFFFFFFFFFFz":"0xFFFFFFFFz");
    static const uintptr_t minValue = 0z;
    static const int size = $(SLANG_PROCESSOR_X86_64?"8":"4");
}

//@hidden:
__generic<T>
__magic_type(OutType)
__intrinsic_type($(kIROp_OutType))
struct Out
{};

__generic<T>
__magic_type(InOutType)
__intrinsic_type($(kIROp_InOutType))
struct InOut
{};

__generic<T>
__magic_type(RefType)
__intrinsic_type($(kIROp_RefType))
struct Ref
{};

__generic<T>
__magic_type(ConstRefType)
__intrinsic_type($(kIROp_ConstRefType))
struct ConstRef
{};

typealias __Addr<T> = Ptr<T, $( (uint64_t)AddressSpace::Generic)ULL>;

//@public:

/// `Optional<T>` is a type with one extra value than `T`, this extra value is
/// used to represent a "missing" or "invalid" `T`. This extra value is called
/// `none`. 
///
/// `none` can be compared against with the operators `==` or `!=`.
///
/// An `Optional<T>` value can also be implicitly or explicitly constructed from
/// a value of type `T`
///
/// `Optional<T>` values can be deconstructed with `if(let myT = myOptionalT) ...` 
/// where this branch will only be taken if `myOptionalT` contains a value
/// of type `T` (to be put into scope as `myT`).
__generic<T>
__magic_type(OptionalType)
__intrinsic_type($(kIROp_OptionalType))
struct Optional
{
    /// Return `true` iff this `Optional` contains a value of type `T`
    property bool hasValue
    {
        __intrinsic_op($(kIROp_OptionalHasValue))
        get;
    }

    /// If this `Optional` contains a value of type `T` return that.
    property T value
    {
        __intrinsic_op($(kIROp_GetOptionalValue))
        get;
    }

    __implicit_conversion($(kConversionCost_ValToOptional))
    __intrinsic_op($(kIROp_MakeOptionalValue))
    __init(T val);
};

//@hidden:
__generic<T>
[__unsafeForceInlineEarly]
bool operator==(Optional<T> val, __none_t noneVal)
{
    return !val.hasValue;
}
__generic<T>
[__unsafeForceInlineEarly]
bool operator!=(Optional<T> val, __none_t noneVal)
{
    return val.hasValue;
}
__generic<T>
[__unsafeForceInlineEarly]
bool operator==(__none_t noneVal, Optional<T> val)
{
    return !val.hasValue;
}
__generic<T>
[__unsafeForceInlineEarly]
bool operator!=(__none_t noneVal, Optional<T> val)
{
    return val.hasValue;
}

//@public:

/// A variadic generic storing the product of several types.
///
/// The member names of individual values are `_0`, `_1`, `_2`, ...
///
/// New tuples can also be constructed by swizzling an existing tuple with the
/// concatenation of these names, for example `x._2_1_0` will return the first
/// three members of the tuple `x` in reverse order.
///
/// When all tuple elements conform to `IComparable` tuples can themselves be
/// compared according to a lexicographic ordering.
///
/// The number of elements in a tuple is given by the `countof` function.
__generic<each T>
__magic_type(TupleType)
struct Tuple
{
    __intrinsic_op($(kIROp_MakeTuple))
    __init(expand each T);
}

/// Construct a tuple from several values in order
__intrinsic_op($(kIROp_MakeTuple))
Tuple<T> makeTuple<each T>(T v);

/// Return a tuple containing the values of both input tuples in order
Tuple<T, U> concat<each T, each U>(Tuple<T> t, Tuple<U> u)
{
    return makeTuple(expand each t, expand each u);
}

//@hidden:
[__unsafeForceInlineEarly]
bool __assign(inout bool v, bool newVal)
{
    v = newVal;
    return newVal;
}

[__unsafeForceInlineEarly]
void __tupleLessKernel<T : IComparable>(inout bool result, inout bool exit, T a, T b)
{
    if (!exit)
    {
        if (a.lessThan(b))
        {
            result = true;
            exit = true;
        }
        else if (!a.equals(b))
        {
            exit = true;
        }
    }
}

[__unsafeForceInlineEarly]
void __tupleGreaterKernel<T : IComparable>(inout bool result, inout bool exit, T a, T b)
{
    if (!exit)
    {
        if (!a.lessThanOrEquals(b))
        {
            result = true;
            exit = true;
        }
        else if (!a.equals(b))
        {
            exit = true;
        }
    }
}

//@public:
__generic<each T : IComparable>
extension Tuple<T> : IComparable
{
    bool lessThan(Tuple<T> other)
    {
        bool result = false;
        bool exit = false;
        expand __tupleLessKernel(result, exit, each this, each other);
        return result;
    }
    bool lessThanOrEquals(Tuple<T> other)
    {
        bool result = false;
        bool exit = false;
        expand __tupleGreaterKernel(result, exit, each this, each other);
        return !result;
    }
    bool equals(Tuple<T> other)
    {
        bool result = true;
        expand result && __assign(result, result && (each this).equals(each other));
        return result;
    }
}

/// Represents an interface for a mutating function that can take multiple parameters.
/// The function allows to modify the state of the object it belongs to.
interface IMutatingFunc<TR, each TP>
{
    /// Defines a mutating function that takes multiple parameters and returns a result of type `TR`.
    /// This function can modify the state of the object it is called on.
    [mutating]
    TR operator()(expand each TP p);
}

/// Represents an interface for a function that can take multiple parameters.
/// This interface inherits from `IMutatingFunc` but is used for non-mutating functions.
interface IFunc<TR, each TP> : IMutatingFunc<TR, expand each TP>
{
    /// Defines a non-mutating function that takes multiple parameters and returns a result of type `TR`.
    TR operator()(expand each TP p);
}

/// Represents an interface for a mutating function that can take multiple differentiable parameters.
/// The function allows to modify the state of the object it belongs to and supports differentiation.
interface IDifferentiableMutatingFunc<TR : IDifferentiable, each TP : IDifferentiable> : IMutatingFunc<TR, expand each TP>
{
    /// Defines a mutating function that takes multiple differentiable parameters and returns a differentiable result of type `TR`.
    /// This function can modify the state of the object it is called on and supports automatic differentiation.
    [Differentiable]
    [mutating]
    TR operator()(expand each TP p);
}

/// Represents an interface for a function that can take multiple differentiable parameters and supports differentiation.
/// This interface inherits from both `IFunc` and `IDifferentiableMutatingFunc` but is used for non-mutating differentiable functions.
interface IDifferentiableFunc<TR : IDifferentiable, each TP : IDifferentiable> : IFunc<TR, expand each TP>, IDifferentiableMutatingFunc<TR, expand each TP>
{
    /// Defines a non-mutating function that takes multiple differentiable parameters and returns a differentiable result of type `TR`.
    /// This function supports automatic differentiation.
    [Differentiable]
    TR operator()(expand each TP p);
}

//@hidden:
__generic<T>
__magic_type(NativeRefType)
__intrinsic_type($(kIROp_NativePtrType))
struct NativeRef
{
    __intrinsic_op($(kIROp_GetNativePtr))
    __init(T val);
};

__generic<T>
__intrinsic_op($(kIROp_ManagedPtrAttach))
void __managed_ptr_attach(__ref T val, NativeRef<T> nativeVal);

__generic<T>
[__unsafeForceInlineEarly]
T __attachToNativeRef(NativeRef<T> nativeVal)
{
    T result;
    __managed_ptr_attach(result, nativeVal);
    return result;
}

//@public:

/// Represents a string.
/// This type can only be used on the CPU target.
__magic_type(StringType)
__intrinsic_type($(kIROp_StringType))
struct String
{
    [require(cpp)]
    __intrinsic_op($(kIROp_MakeString))
    __init(int val);

    [require(cpp)]
    __intrinsic_op($(kIROp_MakeString))
    __init(uint val);

    [require(cpp)]
    __intrinsic_op($(kIROp_MakeString))
    __init(int64_t val);

    [require(cpp)]
    __intrinsic_op($(kIROp_MakeString))
    __init(uint64_t val);

    [require(cpp)]
    __intrinsic_op($(kIROp_MakeString))
    __init(float val);

    [require(cpp)]
    __intrinsic_op($(kIROp_MakeString))
    __init(double val);

    [require(cpp)]
    __implicit_conversion($(kConversionCost_None))
    __intrinsic_op($(kIROp_MakeString))
    __init(NativeString value);

    /// Returns the length of the string.
    [require(cpp)]
    int64_t getLength();

    /// Returns the length of the string.
    property int length
    {
        get { return (int)getLength(); }
    }
};

/// @category misc_types
typedef String string;

/// @category misc_types
__magic_type(NativeStringType)
__intrinsic_type($(kIROp_NativeStringType))
struct NativeString
{
    [require(cpp)]
    int getLength()
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "int(strlen($0))";
        }
    }

    [require(cpp)]
    Ptr<void> getBuffer()
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "(void*)((const char*)($0))";
        }
    }

    property int length { [__unsafeForceInlineEarly] get{return getLength();} }

    __implicit_conversion($(kConversionCost_None))
    __intrinsic_op($(kIROp_getNativeStr))
    __init(String value);

    __init() { this = NativeString(""); }
};

extension Ptr<void>
{
    __implicit_conversion($(kConversionCost_PtrToVoidPtr))
    [__unsafeForceInlineEarly]
    __init(NativeString nativeStr) { this = nativeStr.getBuffer(); }

    __generic<T, let addrSpace : uint64_t>
    __intrinsic_op(0)
    __implicit_conversion($(kConversionCost_PtrToVoidPtr))
    __init(Ptr<T, addrSpace> ptr);

    __generic<T>
    __intrinsic_op(0)
    __implicit_conversion($(kConversionCost_PtrToVoidPtr))
    __init(NativeRef<T> ptr);
}

//@hidden:
__magic_type(DynamicType)
__intrinsic_type($(kIROp_DynamicType))
struct __Dynamic
{};

//@public:

extension half : IRangedValue
{
    static const half maxValue = half(65504);
    static const half minValue = half(-65504);
}

extension float : IRangedValue
{
    static const float maxValue = 340282346638528859811704183484516925440.0f;
    static const float minValue = -340282346638528859811704183484516925440.0f;

}

extension double : IRangedValue
{
    static const double maxValue = bit_cast<double>(0x7fefffffffffffffULL);
    static const double minValue = bit_cast<double>(0xffefffffffffffffULL);
}

extension int : IRangedValue
{
    static const int maxValue = 2147483647;
    static const int minValue = -2147483648;
}

extension uint : IRangedValue
{
    static const uint maxValue = 4294967295;
    static const uint minValue = 0;
}

extension int8_t : IRangedValue
{
    static const int8_t maxValue = 127;
    static const int8_t minValue = -128;
}

extension uint8_t : IRangedValue
{
    static const uint8_t maxValue = 255;
    static const uint8_t minValue = 0;
}

extension uint16_t : IRangedValue
{
    static const uint16_t maxValue = 65535;
    static const uint16_t minValue = 0;
}

extension int16_t : IRangedValue
{
    static const int16_t maxValue = 32767;
    static const int16_t minValue = -32768;
}

__generic<T, let N:int>
__magic_type(ArrayExpressionType)
struct Array : IRWArray<T>
{
    __intrinsic_op($(kIROp_GetArrayLength))
    int getCount();
}

/// @category math_types Math types
/// An `N` component vector with elements of type `T`.
__generic<T = float, let N : int = 4>
__magic_type(VectorExpressionType)
struct vector : IRWArray<T>
{
        /// The element type of the vector
    typedef T Element;


        /// Initialize a vector where all elements have the same scalar `value`.
    [TreatAsDifferentiable]
    __implicit_conversion($(kConversionCost_ScalarToVector))
    __intrinsic_op($(kIROp_MakeVectorFromScalar))
    __init(T value);

    /// Initialize a vector from a value of the same type
    // TODO: we should revise semantic checking so this kind of "identity" conversion is not required
    __intrinsic_op(0)
    [TreatAsDifferentiable]
    __init(vector<T,N> value);

    [ForceInline]
    int getCount() { return N; }
}

//@hidden:
static const int kRowMajorMatrixLayout = $(SLANG_MATRIX_LAYOUT_ROW_MAJOR);
static const int kColumnMajorMatrixLayout = $(SLANG_MATRIX_LAYOUT_COLUMN_MAJOR);

//@public:
/// A matrix with `R` rows and `C` columns, with elements of type `T`.
/// @category math_types Math types
__generic<T = float, let R : int = 4, let C : int = 4, let L : int = $(SLANG_MATRIX_LAYOUT_MODE_UNKNOWN)>
__magic_type(MatrixExpressionType)
struct matrix : IRWArray<vector<T,C>>
{
    __intrinsic_op($(kIROp_MakeMatrixFromScalar))
    __implicit_conversion($(kConversionCost_ScalarToMatrix))
    [TreatAsDifferentiable]
    __init(T val);

    /// Initialize a vector from a value of the same type
    // TODO: we should revise semantic checking so this kind of "identity" conversion is not required
    __intrinsic_op(0)
    [TreatAsDifferentiable]
    __init(This value);

    [ForceInline]
    int getCount() { return R; }
}

//@hidden:
__intrinsic_op($(kIROp_Eql))
vector<bool, N> __vectorEql<T, let N : int>(vector<T, N> left, vector<T, N> right);

//@public:
__generic<T:__BuiltinIntegerType, let N : int>
extension vector<T,N> : IInteger
{
    [__unsafeForceInlineEarly] bool lessThan(This other) { return this[0] < other[0]; }
    [__unsafeForceInlineEarly] bool lessThanOrEquals(This other) { return this[0] <= other[0]; }
    [__unsafeForceInlineEarly] bool equals(This other) { return all(__vectorEql(this, other)); }
    __intrinsic_op($(kIROp_Add)) This add(This other);
    __intrinsic_op($(kIROp_Sub)) This sub(This other);
    __intrinsic_op($(kIROp_Mul)) This mul(This other);
    __intrinsic_op($(kIROp_Div)) This div(This other);
    __intrinsic_op($(kIROp_FRem)) This mod(This other);
    __intrinsic_op($(kIROp_Neg)) This neg();

    __intrinsic_op($(kIROp_Lsh))
    This shl(int value);

    __intrinsic_op($(kIROp_Rsh))
    This shr(int value);

    __intrinsic_op($(kIROp_BitAnd))
    This bitAnd(This other);

    __intrinsic_op($(kIROp_BitOr))
    This bitOr(This other);

    __intrinsic_op($(kIROp_BitXor))
    This bitXor(This other);

    __intrinsic_op($(kIROp_BitNot))
    This bitNot();

    __intrinsic_op($(kIROp_And))
    This and(This other);

    __intrinsic_op($(kIROp_Or))
    This or(This other);

    __intrinsic_op($(kIROp_Not))
    This not();

    [__unsafeForceInlineEarly] int toInt() { return __intCast<int>(this[0]); }
    [__unsafeForceInlineEarly] int64_t toInt64() { return __intCast<int64_t>(this[0]); }
    [__unsafeForceInlineEarly] uint toUInt() { return __intCast<uint>(this[0]); }
    [__unsafeForceInlineEarly] uint64_t toUInt64() { return __intCast<uint64_t>(this[0]); }
    [OverloadRank(-1)]
    [__unsafeForceInlineEarly] __init(int v) { this = vector<T,N>(T(v)); }
    [OverloadRank(-1)]
    [__unsafeForceInlineEarly] __init(int64_t v) { this = vector<T,N>(T(v)); }

    [OverloadRank(-1)]
    __implicit_conversion($(kConversionCost_Default))
    __intrinsic_op($(kIROp_CastFloatToInt))
    __generic<U:__BuiltinFloatingPointType>
    __init(vector<U, N> other);

    [OverloadRank(-1)]
    __intrinsic_op($(kIROp_IntCast))
    __generic<U:__BuiltinIntegerType>
    __init(vector<U, N> other);
}

__generic<T:__BuiltinFloatingPointType, let N : int>
extension vector<T,N> : IFloat
{
    [__unsafeForceInlineEarly] bool lessThan(This other) { return this[0] < other[0]; }
    [__unsafeForceInlineEarly] bool lessThanOrEquals(This other) { return this[0] <= other[0]; }
    [__unsafeForceInlineEarly] bool equals(This other) { return all(__vectorEql(this, other)); }
    __intrinsic_op($(kIROp_Add)) This add(This other);
    __intrinsic_op($(kIROp_Sub)) This sub(This other);
    __intrinsic_op($(kIROp_Mul)) This mul(This other);
    __intrinsic_op($(kIROp_Div)) This div(This other);
    __intrinsic_op($(kIROp_FRem)) This mod(This other);
    __intrinsic_op($(kIROp_Neg)) This neg();
    __intrinsic_op($(kIROp_Mul)) This scale<T1:__BuiltinFloatingPointType>(T1 s);
    [__unsafeForceInlineEarly] float toFloat() { return __realCast<float>(this[0]); }

    [OverloadRank(-1)]
    __implicit_conversion($(kConversionCost_Default))
    __intrinsic_op($(kIROp_FloatCast))
    __generic<U:__BuiltinFloatingPointType>
    __init(vector<U, N> other);

    [OverloadRank(-1)]
    __intrinsic_op($(kIROp_CastIntToFloat))
    __generic<U:__BuiltinIntegerType>
    __init(vector<U, N> other);

    [OverloadRank(-1)]
    [__unsafeForceInlineEarly] __init(int v) { this = vector<T,N>(T(v)); }
    [OverloadRank(-1)]
    [__unsafeForceInlineEarly] __init(float v) { this = vector<T,N>(T(v)); }

}

__intrinsic_op($(kIROp_Add))
T __internal_add<T>(T a, T b);

__intrinsic_op($(kIROp_Mul))
T __internal_mul<T, U>(U a, T b);

__generic<T:IDifferentiable, let N : int>
extension vector<T,N> : IDifferentiable
{
    // IDifferentiable
    typedef vector<T, N> Differential;

    [__unsafeForceInlineEarly]
    [BackwardDifferentiable]
    static Differential dzero()
    {
        return Differential(__slang_noop_cast<T>(T.dzero()));
    }

    [__unsafeForceInlineEarly]
    [BackwardDifferentiable]
    static Differential dadd(Differential a, Differential b)
    {
        return __internal_add(a, b);
    }

    __generic<U : __BuiltinRealType>
    [__unsafeForceInlineEarly]
    [BackwardDifferentiable]
    static Differential dmul(U a, Differential b)
    {
        return __internal_mul(__realCast<float>(a), b);
    }
}

__generic<T:__BuiltinFloatingPointType, let N : int, let M : int, let L : int>
extension matrix<T,N,M,L> : IFloat
{
    [TreatAsDifferentiable][__unsafeForceInlineEarly] bool lessThan(This other) { return this < other; }
    [TreatAsDifferentiable][__unsafeForceInlineEarly] bool lessThanOrEquals(This other) { return this <= other; }
    [TreatAsDifferentiable][__unsafeForceInlineEarly] bool equals(This other) { return all(this == other); }
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_Add)) This add(This other);
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_Sub)) This sub(This other);
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_Mul))This mul(This other);
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_Div)) This div(This other);
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_FRem)) This mod(This other);
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_Neg)) This neg();
    [TreatAsDifferentiable] __intrinsic_op($(kIROp_Mul)) This scale<T1:__BuiltinFloatingPointType>(T1 s);
    [TreatAsDifferentiable][__unsafeForceInlineEarly] This scale<T1:__BuiltinFloatingPointType>(T1 s);
    [TreatAsDifferentiable][__unsafeForceInlineEarly] float toFloat() { return __realCast<float>(this[0][0]); }

    [OverloadRank(-1)]
    [TreatAsDifferentiable]
    [__unsafeForceInlineEarly]
    __implicit_conversion($(kConversionCost_ScalarIntegerToFloatMatrix))
    __init(int v) { this = matrix<T,N,M>(T(v)); }

    [OverloadRank(-1)]
    [TreatAsDifferentiable]
    [__unsafeForceInlineEarly]
    __implicit_conversion($(kConversionCost_ScalarToMatrix))
    __init(float v) { this = matrix<T,N,M>(T(v)); }
}

__generic<T:IDifferentiable, let N : int, let M : int, let L : int>
extension matrix<T,N,M,L> : IDifferentiable
{
    // IDifferentiable.
    typedef matrix<T, N,M,L> Differential;

    [__unsafeForceInlineEarly]
    [BackwardDifferentiable]
    static Differential dzero()
    {
        return matrix<T, N,M,L>(__slang_noop_cast<T>(T.dzero()));
    }

    [__unsafeForceInlineEarly]
    [BackwardDifferentiable]
    static Differential dadd(Differential a, Differential b)
    {
        return __internal_add(a, b);
    }

    __generic<U : __BuiltinRealType>
    [__unsafeForceInlineEarly]
    [BackwardDifferentiable]
    static Differential dmul(U a, Differential b)
    {
        return __internal_mul(__realCast<float>(a), b);
    }
}

__generic<let R : int, let C : int, let L : int>
extension matrix<int16_t,R,C,L>
{
    typealias T = int16_t;
    __implicit_conversion($(kConversionCost_IntegerTruncate))
    __init(int value) { this = matrix<T,R,C,L>(T(value)); }
}

//@hidden:
__intrinsic_op(makeVector)
__generic<T, let N:int>
vector<T,N*2> __makeVector(vector<T,N> vec1, vector<T,N> vec2);

//@public:
__generic<T>
extension vector<T, 4>
{
    __generic<let L : int>
    [__unsafeForceInlineEarly]
    __init(matrix<T, 2, 2, L> value)
    {
        this = __makeVector(value[0], value[1]);
    }
}
__generic<T, let L : int>
extension matrix<T, 2, 2, L>
{
    [__unsafeForceInlineEarly]
    __init(vector<T, 4> value)
    {
        this[0] = value.xy;
        this[1] = value.zw;
    }
}

//@hidden:
${{{{
static const struct {
    char const* name;
    char const* glslPrefix;
} kTypes[] =
{
    {"half",        "f16"},
    {"float",       ""},
    {"double",      "d"},

    {"float16_t",   "f16"},
    {"float32_t",   "f32"},
    {"float64_t",   "f64"},

    {"int8_t",      "i8"},
    {"int16_t",     "i16"},
    {"int32_t",     "i32"},
    {"int",         "i"},
    {"int64_t",     "i64"},

    {"uint8_t",     "u8"},
    {"uint16_t",    "u16"},
    {"uint32_t",    "u32"},
    {"uint",        "u"},
    {"uint64_t",    "u64"},

    {"bool",        "b"},
};

static const int kTypeCount = sizeof(kTypes) / sizeof(kTypes[0]);

for (int tt = 0; tt < kTypeCount; ++tt)
{
    // Declare HLSL vector types
    for (int ii = 1; ii <= 4; ++ii)
    {
        sb << "typedef vector<" << kTypes[tt].name << "," << ii << "> " << kTypes[tt].name << ii << ";\n";
    }

    // Declare HLSL matrix types
    for (int rr = 1; rr <= 4; ++rr)
    for (int cc = 1; cc <= 4; ++cc)
    {
        sb << "typedef matrix<" << kTypes[tt].name << "," << rr << "," << cc << "> " << kTypes[tt].name << rr << "x" << cc << ";\n";
    }
}

sb << "\n\n";
sb << "// Cast from vector<T,1> to T, which is a scalar type\n";

for (int tt = 0; tt < kBaseTypeCount; ++tt)
{
    if(kBaseTypes[tt].tag == BaseType::Void) continue;

    const char* tname = kBaseTypes[tt].name;

    sb << "extension " << tname << "\n";
    sb << "{\n";
    sb << "    __implicit_conversion(" << kConversionCost_OneVectorToScalar << ")\n";
    sb << "    __init(vector<" << tname << ",1> v) { this = v[0]; }\n";
    sb << "}\n";
}

// Declare additional built-in generic types
}}}}

//@ public:

__generic<T, L:IBufferDataLayout = DefaultDataLayout>
__intrinsic_type($(kIROp_ConstantBufferType))
__magic_type(ConstantBufferType)
struct ConstantBuffer {}

///@category texture_types
__generic<T>
__intrinsic_type($(kIROp_TextureBufferType))
__magic_type(TextureBufferType)
struct TextureBuffer {}

__generic<T>
__intrinsic_type($(kIROp_ParameterBlockType))
__magic_type(ParameterBlockType)
struct ParameterBlock {}

/// @category stage_io
__generic<T, let MAX_VERTS : uint>
__magic_type(VerticesType)
__intrinsic_type($(kIROp_VerticesType))
[__NonCopyableType]
struct OutputVertices
{
    __intrinsic_op($(kIROp_MetalSetVertex))
    static void _metalSetVertex(uint index, T val);

    __intrinsic_op($(kIROp_MeshOutputSet))
    static void _setVertex(This v, uint index, T val);

    __subscript(uint index) -> T
    {
        // TODO: Make sure this remains write only, we can't do this with just
        // a 'set' operation as it's legal to only write to part of the output
        // buffer, or part of the output buffer at a time.

        [mutating]
        [require(glsl_hlsl_metal_spirv, meshshading)]
        set
        {
            __target_switch
            {
            case metal: _metalSetVertex(index, newValue);
            case glsl: _setVertex(this, index, newValue);
            case hlsl: _setVertex(this, index, newValue);
            case spirv: _setVertex(this, index, newValue);
            }
        }
        
        //
        // If a 'OutputVertices[index]' is referred to by a '__ref', call 'kIROp_MeshOutputRef(index)'
        [require(glsl_hlsl_spirv, meshshading)]
        __intrinsic_op($(kIROp_MeshOutputRef))
        ref;
    }
};

/// @category stage_io
__generic<T, let MAX_PRIMITIVES : uint>
__magic_type(IndicesType)
__intrinsic_type($(kIROp_IndicesType))
[__NonCopyableType]
struct OutputIndices
{
    __intrinsic_op($(kIROp_MetalSetIndices))
    static void __metalSetIndices(uint index, T val);

    // for some reason only here in the indices array it uses the return value as an actual
    // operand, while the others use the value of the instruction (the return value) to access
    // the type of the vertex, as when using the ref there is no third operand
    __intrinsic_op($(kIROp_MeshOutputSet))
    static void __setIndices(This v, uint index, T val);

    __subscript(uint index) -> T
    {
        [mutating]
        [require(glsl_hlsl_metal_spirv, meshshading)]
        set
        {
            __target_switch
            {
            case metal: __metalSetIndices(index, newValue);
            case glsl: __setIndices(this, index, newValue);
            case hlsl: __setIndices(this, index, newValue);
            case spirv: __setIndices(this, index, newValue);
            }
        }
    }
};

/// @category stage_io
__generic<T, let MAX_PRIMITIVES : uint>
__magic_type(PrimitivesType)
__intrinsic_type($(kIROp_PrimitivesType))
[__NonCopyableType]
struct OutputPrimitives
{
    __intrinsic_op($(kIROp_MetalSetPrimitive))
    static void __metalSetPrimitive(uint index, T val);

    __intrinsic_op($(kIROp_MeshOutputSet))
    static void __setPrimitive(This v, uint index, T val);

    __subscript(uint index)->T
    {
        [mutating]
        [require(glsl_hlsl_metal_spirv, meshshading)]
        set
        {
            __target_switch
            {
            case metal: __metalSetPrimitive(index, newValue);
            case glsl: __setPrimitive(this, index, newValue);
            case hlsl: __setPrimitive(this, index, newValue);
            case spirv: __setPrimitive(this, index, newValue);
            }
        }

        // If a 'OutputPrimitives[index]' is referred to by a '__ref', call 'kIROp_MeshOutputRef(index)'
        [require(glsl_hlsl_spirv, meshshading)]
        __intrinsic_op($(kIROp_MeshOutputRef))
        ref;
    }
};

//@ public:

// Need to add constructors to the types above

__generic<T> __extension vector<T, 2>
{
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, T y);
}
__generic<T> __extension vector<T, 3>
{
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, T y, T z);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(vector<T,2> xy, T z);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, vector<T,2> yz);
}
__generic<T> __extension vector<T, 4>
{
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, T y, T z, T w);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(vector<T,2> xy, T z, T w);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, vector<T,2> yz, T w);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, T y, vector<T,2> zw);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(vector<T,2> xy, vector<T,2> zw);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(vector<T,3> xyz, T w);

    [__unsafeForceInlineEarly]
    __intrinsic_op($(kIROp_MakeVector))
    __init(T x, vector<T,3> yzw);
}

${{{{


for( int R = 1; R <= 4; ++R )
for( int C = 1; C <= 4; ++C )
{
    sb << "__generic<T, let L:int> __extension matrix<T, " << R << "," << C << ", L>\n{\n";

    // initialize from R*C scalars
    if (R == 1 || C == 1)
        sb << "[require(hlsl)]\n";
    sb << "__intrinsic_op(" << int(kIROp_MakeMatrix) << ") __init(";
    for( int ii = 0; ii < R; ++ii )
    for( int jj = 0; jj < C; ++jj )
    {
        if ((ii+jj) != 0) sb << ", ";
        sb << "T m" << ii << jj;
    }
    sb << ");\n";

    // Initialize from R C-vectors
    if (R == 1 || C == 1)
        sb << "[require(hlsl)]\n";
    sb << "__intrinsic_op(" << int(kIROp_MakeMatrix) << ") __init(";
    for (int ii = 0; ii < R; ++ii)
    {
        if(ii != 0) sb << ", ";
        sb << "vector<T," << C << "> row" << ii;
    }
    sb << ");\n";

    // initialize from a matrix of larger size
    for(int rr = R; rr <= 4; ++rr)
    for( int cc = C; cc <= 4; ++cc )
    {
        if(rr == R && cc == C) continue;
        if (R == 1 || C == 1)
            sb << "[require(hlsl)]\n";
        sb << "__intrinsic_op(" << int(kIROp_MatrixReshape) << ") __init(matrix<T," << rr << "," << cc << ", L> value);\n";
    }

    // Initialize from matrix and vector
    if (R > 2)
    {
        sb << "__init(matrix<T," << (R - 1) << "," << C << "> m, vector<T," << C << "> row" << (R - 1) << ") ";
        sb << "{ this = This(";
        for (int ii = 0; ii < R - 1; ++ii)
        {
            sb << "m[" << ii << "], ";
        }
        sb << "row" << (R - 1) << "); }\n";
    }

    sb << "}\n";
}

}}}}

//@hidden:
__intrinsic_op($(kIROp_BuiltinCast))
internal T __builtin_cast<T, U>(U u);

// If T is implicitly convertible to U, then vector<T,N> is implicitly convertible to vector<U,N>.
__generic<ToType, let N : int> extension vector<ToType,N>
{
    __implicit_conversion(constraint)
    __intrinsic_op(BuiltinCast)
    __init<FromType>(vector<FromType,N> value) where ToType(FromType) implicit;

    __implicit_conversion(constraint+)
    [__unsafeForceInlineEarly]
    [__readNone]
    [TreatAsDifferentiable]
    __init<FromType>(FromType value) where ToType(FromType) implicit
    {
        this = __builtin_cast<vector<ToType,N>>(vector<FromType,N>(value));
    }
}

// If T is implicitly convertible to U, then matrix<T,R,C,L> is implicitly convertible to matrix<U,R,C,L>.
__generic<ToType, let R : int, let C : int, let L : int> extension matrix<ToType,R,C,L>
{
    __implicit_conversion(constraint)
    __intrinsic_op(BuiltinCast)
    __init<FromType>(matrix<FromType,R,C,L> value) where ToType(FromType) implicit;
}

//@ hidden:
__generic<T, U>
__intrinsic_op(0)
T __slang_noop_cast(U u);


//@ public:

/// Sampling state for filtered texture fetches.
/// @category sampler_types Sampler types
__magic_type(SamplerStateType, $(int(SamplerStateFlavor::SamplerState)))
__intrinsic_type($(kIROp_SamplerStateType))
struct SamplerState
{
}

/// Sampling state for filtered texture fetches that include a comparison operation before filtering.
/// @category sampler_types
__magic_type(SamplerStateType, $(int(SamplerStateFlavor::SamplerComparisonState)))
__intrinsic_type($(kIROp_SamplerComparisonStateType))
struct SamplerComparisonState
{
}

//@ hidden:

${{{{

for (auto op : intrinsicUnaryOps)
{
    for (auto type : kBaseTypes)
    {
        if ((type.flags & op.flags) == 0)
            continue;

        char const* resultType = type.name;
        if (op.flags & BOOL_RESULT) resultType = "bool";

        // scalar version
        sb << "__prefix __intrinsic_op(" << int(op.opCode) << ") " << resultType << " operator" << op.opName << "(" << type.name << " value);\n";
        sb << "__intrinsic_op(" << int(op.opCode) << ") " << resultType << " __" << op.funcName << "(" << type.name << " value);\n";

        // vector version
        sb << "__generic<let N : int> ";
        sb << "__prefix __intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(" << "vector<" << type.name << ",N> value);\n";

        // matrix version
        sb << "__generic<let N : int, let M : int> ";
        sb << "__prefix __intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(" <<  "matrix<" << type.name << ",N,M> value);\n";
    }

    // Synthesize generic versions
    if(op.interface)
    {
        char const* resultType = "T";
        if (op.flags & BOOL_RESULT) resultType = "bool";
        
        // scalar version
        sb << "__generic<T : " << op.interface << ">\n";
        sb << "[OverloadRank(10)]";
        sb << "__prefix __intrinsic_op(" << int(op.opCode) << ") " << resultType << " operator" << op.opName << "(" << "T value);\n";

        // vector version
        sb << "__generic<T : " << op.interface << ", let N : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__prefix __intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(vector<T,N> value);\n";

        // matrix version
        sb << "__generic<T : " << op.interface << ", let N : int, let M : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__prefix __intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(matrix<T,N,M> value);\n";
    }
}

}}}}

__generic<T, let addrSpace : uint64_t>
__intrinsic_op(0)
[require(cpp_cuda_spirv)]
__prefix Ref<T> operator*(Ptr<T, addrSpace> value);

__generic<T>
__intrinsic_op(0)
[require(cpp_cuda_spirv)]
__prefix Ptr<T, $( (uint64_t)AddressSpace::UserPointer)ULL> operator&(__ref T value);

__generic<T>
__intrinsic_op(0)
[require(cpp_cuda_spirv)]
__Addr<T> __get_addr( __ref T value);

__generic<T, let addrSpace : uint64_t>
__intrinsic_op($(kIROp_GetOffsetPtr))
Ptr<T, addrSpace> operator+(Ptr<T, addrSpace> value, int64_t offset);

__generic<T, let addrSpace : uint64_t>
[__unsafeForceInlineEarly]
Ptr<T, addrSpace> operator -(Ptr<T, addrSpace> value, int64_t offset)
{
    return __getOffsetPtr(value, -offset);
}

__generic<T : IArithmetic>
[__unsafeForceInlineEarly]
__prefix T operator+(T value)
{ return value; }

__generic<T : __BuiltinArithmeticType, let N : int>
[__unsafeForceInlineEarly]
__prefix vector<T,N> operator+(vector<T,N> value)
{ return value; }

__generic<T : __BuiltinArithmeticType, let R : int, let C : int>
[__unsafeForceInlineEarly]
__prefix matrix<T,R,C> operator+(matrix<T,R,C> value)
{ return value; }

${{{{

static const struct IncDecOpInfo
{
    char const* name;
    char const* binOp;
} kIncDecOps[] =
{
    { "++", "+" },
    { "--", "-" },
};
static const struct IncDecOpFixity
{
    char const* qual;
    char const* bodyPrefix;
    char const* returnVal;
} kIncDecFixities[] =
{
    { "__prefix", "", "value" },
    { "__postfix", " let result = value;", "result" },
};
for(auto op : kIncDecOps)
for(auto fixity : kIncDecFixities)
{
}}}}

$(fixity.qual)
__generic<T : __BuiltinArithmeticType>
[__unsafeForceInlineEarly]
T operator$(op.name)( in out T value)
{ $(fixity.bodyPrefix) value = value $(op.binOp) __builtin_cast<T>(1); return $(fixity.returnVal); }

$(fixity.qual)
__generic<T : __BuiltinArithmeticType, let N : int>
[__unsafeForceInlineEarly]
vector<T,N> operator$(op.name)(in out vector<T,N> value)
{$(fixity.bodyPrefix) value = value $(op.binOp) __builtin_cast<T>(1); return $(fixity.returnVal); }

$(fixity.qual)
__generic<T : __BuiltinArithmeticType, let R : int, let C : int, let L : int>
[__unsafeForceInlineEarly]
matrix<T,R,C> operator$(op.name)(in out matrix<T,R,C,L> value)
{$(fixity.bodyPrefix) value = value $(op.binOp) __builtin_cast<T>(1); return $(fixity.returnVal); }

$(fixity.qual)
__generic<T, let addrSpace : uint64_t>
[__unsafeForceInlineEarly]
Ptr<T, addrSpace> operator$(op.name)(in out Ptr<T, addrSpace> value)
{$(fixity.bodyPrefix) value = value $(op.binOp) 1; return $(fixity.returnVal); }

${{{{
}

for (auto op : intrinsicBinaryOps)
{
    for (auto type : kBaseTypes)
    {
        if ((type.flags & op.flags) == 0)
            continue;

        char const* leftType = type.name;
        char const* rightType = leftType;
        char const* resultType = leftType;

        if (op.flags & BOOL_RESULT) resultType = "bool";

        // TODO: We should handle a `SHIFT` flag on the op
        // by changing `rightType` to `int` in order to
        // account for the fact that the shift amount should
        // always have a fixed type independent of the LHS.
        //
        // (It is unclear why this change hadn't been made
        // already, so it is possible that such a change
        // breaks overload resolution or other parts of
        // the compiler)

        // scalar version
        sb << "__intrinsic_op(" << int(op.opCode) << ") " << resultType << " operator" << op.opName << "(" << leftType << " left, " << rightType << " right);\n";
        sb << "__intrinsic_op(" << int(op.opCode) << ") " << resultType << " __" << op.funcName << "(" << leftType << " left, " << rightType << " right);\n";

        // vector version
        sb << "__generic<let N : int> ";
        sb << "__intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(vector<" << leftType << ",N> left, vector<" << rightType << ",N> right);\n";

        // matrix version
        sb << "__generic<let N : int, let M : int> ";
        sb << "__intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(matrix<" << leftType << ",N,M> left, matrix<" << rightType << ",N,M> right);\n";

        // We currently synthesize addiitonal overloads
        // for the case where one or the other operand
        // is a scalar. This choice serves a few purposes:
        //
        // 1. It avoids introducing scalar-to-vector or
        // scalar-to-matrix promotions before the operator,
        // which might allow some back ends to produce
        // more optimal code.
        //
        // 2. It avoids concerns about making overload resolution
        // and the inference rules for `N` and `M` able to
        // handle the mixed vector/scalar or matrix/scalar case.
        //
        // 3. Having explicit overloads for the matrix/scalar cases
        // here means that we do *not* need to support a general
        // implicit conversion from scalars to matrices, unless
        // we decide we want to.
        //
        // Note: Case (2) of the motivation shouldn't really apply
        // any more, because we end up having to support similar
        // inteference for built-in binary math functions where
        // vectors and scalars might be combined (and where defining
        // additional overloads to cover all the combinations doesn't
        // seem practical or desirable).
        //
        // TODO: We should consider whether dropping these extra
        // overloads is possible and worth it. The optimization
        // concern (1) could possibly be addressed in specific
        // back-ends. The issue (3) about not wanting to support
        // implicit scalar-to-matrix conversion may be moot if
        // we end up needing to support mixed scalar/matrix input
        // for builtin in non-operator functions anyway.

        // scalar-vector and scalar-matrix
        sb << "__generic<let N : int> ";
        sb << "__intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(" << leftType << " left, vector<" << rightType << ",N> right);\n";

        sb << "__generic<let N : int, let M : int> ";
        sb << "__intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(" << leftType << " left, matrix<" << rightType << ",N,M> right);\n";

        // vector-scalar and matrix-scalar
        sb << "__generic<let N : int> ";
        sb << "__intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(vector<" << leftType << ",N> left, " << rightType << " right);\n";

        sb << "__generic<let N : int, let M : int> ";
        sb << "__intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(matrix<" << leftType << ",N,M> left, " << rightType << " right);\n";
    }

    // Synthesize generic versions
    if(op.interface)
    {
        char const* leftType = "T";
        char const* rightType = leftType;
        char const* resultType = leftType;

        if (op.flags & BOOL_RESULT) resultType = "bool";
        // TODO: handle `SHIFT`

        // scalar version
        sb << "__generic<T : " << op.interface << ">\n";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") " << resultType << " operator" << op.opName << "(" << leftType << " left, " << rightType << " right);\n";

        // vector version
        sb << "__generic<T : " << op.interface << ", let N : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(vector<" << leftType << ",N> left, vector<" << rightType << ",N> right);\n";

        // matrix version
        sb << "__generic<T : " << op.interface << ", let N : int, let M : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(matrix<" << leftType << ",N,M> left, matrix<" << rightType << ",N,M> right);\n";

        // scalar-vector and scalar-matrix
        sb << "__generic<T : " << op.interface << ", let N : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(" << leftType << " left, vector<" << rightType << ",N> right);\n";

        sb << "__generic<T : " << op.interface << ", let N : int, let M : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(" <<  leftType << " left, matrix<" << rightType << ",N,M> right);\n";

        // vector-scalar and matrix-scalar
        sb << "__generic<T : " << op.interface << ", let N : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") vector<" << resultType << ",N> operator" << op.opName << "(vector<" << leftType << ",N> left, " << rightType << " right);\n";

        sb << "__generic<T : " << op.interface << ", let N : int, let M : int> ";
        sb << "[OverloadRank(10)]";
        sb << "__intrinsic_op(" << int(op.opCode) << ") matrix<" << resultType << ",N,M> operator" << op.opName << "(matrix<" << leftType << ",N,M> left, " << rightType << " right);\n";
    }
}

// We will declare the shift operations entirely as generics
// rather than try to handle all the pairings of left-hand
// and right-hand side types.
//
static const struct ShiftOpInfo
{
    char const* name;
    char const* funcName;
    int op;
} kShiftOps[] =
{
    { "<<", "shl", kIROp_Lsh },
    { ">>", "shr", kIROp_Rsh },
};
for(auto info : kShiftOps) {
}}}}

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType>
__intrinsic_op($(info.op))
L operator$(info.name)(L left, R right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType>
__intrinsic_op($(info.op))
L __$(info.funcName)(L left, R right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType>
[__unsafeForceInlineEarly]
L operator$(info.name)=(in out L left, R right)
{
    left = left $(info.name) right;
    return left;
}

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int>
__intrinsic_op($(info.op))
vector<L,N> operator$(info.name)(vector<L,N> left, vector<R,N> right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int>
[__unsafeForceInlineEarly]
vector<L,N> operator$(info.name)=(in out vector<L,N> left, vector<R,N> right)
{
    left = left $(info.name) right;
    return left;
}

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int, let M : int>
__intrinsic_op($(info.op))
matrix<L,N,M> operator$(info.name)(matrix<L,N,M> left, matrix<R,N,M> right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int, let M : int, let Layout : int>
[__unsafeForceInlineEarly]
matrix<L, N, M> operator$(info.name)=(in out matrix<L, N, M, Layout> left, matrix<R, N, M> right)
{
    left = left $(info.name) right;
    return left;
}

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int>
__intrinsic_op($(info.op))
vector<L,N> operator$(info.name)(L left, vector<R,N> right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int, let M : int>
__intrinsic_op($(info.op))
matrix<L,N,M> operator$(info.name)(L left, matrix<R,N,M> right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int>
__intrinsic_op($(info.op))
vector<L,N> operator$(info.name)(vector<L,N> left, R right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int>
[__unsafeForceInlineEarly]
vector<L, N> operator$(info.name)=(in out vector<L, N> left, R right)
{
    left = left $(info.name) right;
    return left;
}

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int, let M : int>
__intrinsic_op($(info.op))
matrix<L,N,M> operator$(info.name)(matrix<L,N,M> left, R right);

__generic<L: __BuiltinIntegerType, R: __BuiltinIntegerType, let N : int, let M : int, let Layout : int>
[__unsafeForceInlineEarly]
matrix<L,N,M> operator$(info.name)=(in out matrix<L,N,M, Layout> left, R right)
{
    left = left $(info.name) right;
    return left;
}

${{{{
}

    static const struct CompoundBinaryOpInfo
    {
        char const* name;
        char const* interface;
    } kCompoundBinaryOps[] =
    {
        { "+",  "__BuiltinArithmeticType" },
        { "-",  "__BuiltinArithmeticType" },
        { "*",  "__BuiltinArithmeticType" },
        { "/",  "__BuiltinArithmeticType" },
        { "%",  "__BuiltinIntegerType" },
        { "%",  "__BuiltinFloatingPointType" },
        { "&",  "__BuiltinLogicalType" },
        { "|",  "__BuiltinLogicalType" },
        { "^",  "__BuiltinLogicalType" },
    };
    for( auto op : kCompoundBinaryOps )
    {
    }}}}

    __generic<T : $(op.interface)>
    [__unsafeForceInlineEarly]
    T operator$(op.name)=(in out T left, T right)
    {
        left = left $(op.name) right;
        return left;
    }

    __generic<T : $(op.interface), let N : int>
    [__unsafeForceInlineEarly]
    vector<T,N> operator$(op.name)=(in out vector<T,N> left, vector<T,N> right)
    {
        left = left $(op.name) right;
        return left;
    }

    __generic<T : $(op.interface), let N : int>
    [__unsafeForceInlineEarly]
    vector<T,N> operator$(op.name)=(in out vector<T,N> left, T right)
    {
        left = left $(op.name) right;
        return left;
    }

    __generic<T : $(op.interface), let R : int, let C : int, let Layout : int>
    [__unsafeForceInlineEarly]
    matrix<T,R,C> operator$(op.name)=(in out matrix<T,R,C,Layout> left, matrix<T,R,C> right)
    {
        left = left $(op.name) right;
        return left;
    }

    __generic<T : $(op.interface), let R : int, let C : int, let Layout : int>
    [__unsafeForceInlineEarly]
    matrix<T,R,C> operator$(op.name)=(in out matrix<T,R,C, Layout> left, T right)
    {
        left = left $(op.name) right;
        return left;
    }

    ${{{{
    }

}}}}

//@ public:


/// Bit cast between types. `T` and `U` must have the same size.
/// They can be any scalar, vector, matrix, struct or array types.
/// @category conversion
__generic<T, U>
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_BitCast))
T bit_cast(U value);

// Create Existential object
__generic<T, U>
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_CreateExistentialObject))
T createDynamicObject(uint typeId, U value);

/// Reinterpret type `U` as type `T`. `T` and `U` 
/// can be any scalar, vector, matrix, struct or array types.
/// @category conversion
__generic<T, U>
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_Reinterpret))
T reinterpret(U value);

/// `bitfieldInsert` inserts the bits least significant bits of `insert` into base at `offset` offset.
/// The returned value will have bits [offset, offset + bits + 1] taken from [0, bits - 1] of `insert`
/// and all other bits taken directly from the corresponding bits of `base`.
__generic<T>
[__readNone]
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_BitfieldInsert))
T bitfieldInsert(T base, T insert, uint offset, uint bits);

/// `bitfieldExtract` extracts a subset of the bits of `value` and
/// returns it in the least significant bits of the result. The range of bits extracted is [offset, offset + bits - 1].
__generic<T>
[__readNone]
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_BitfieldExtract))
T bitfieldExtract(T value, uint offset, uint bits);

/// Use an otherwise unused value
/// This can be used to silence the warning about returning before initializing an out paramter.
__generic<T>
[__readNone]
[ForceInline]
__intrinsic_op($(kIROp_Unmodified))
void unused(inout T){}

// This can be used to silence the warning about not writing to an inout parameter.
__generic<T>
[__readNone]
[ForceInline]
__intrinsic_op($(kIROp_Unmodified))
void unmodified(out T){}

// Specialized function

/// Given a string returns an integer hash of that string.
__intrinsic_op($(kIROp_GetStringHash))
int getStringHash(String string);

/// Use will produce a syntax error in downstream compiler
/// Useful for testing diagnostics around compilation errors of downstream compiler
/// It 'returns' an int so can be used in expressions without the front end complaining.
[require(cpp_cuda_glsl_hlsl)]
int __SyntaxError()
{
    __target_switch
    {
    case cpp: __intrinsic_asm " @ ";
    case cuda: __intrinsic_asm " @ ";
    case glsl: __intrinsic_asm " @ ";
    case hlsl: __intrinsic_asm " @ ";
    }
}

//@ hidden:

/// For downstream compilers that allow sizeof/alignof/offsetof
/// Can't be called in the C/C++ style. Need to use __size_of<some_type>() as opposed to sizeof(some_type).
__generic<T>
[__readNone]
[require(cpp_cuda)]
int __sizeOf()
{
    __intrinsic_asm "sizeof($[0])", T;
}

__generic<T>
[__readNone]
[require(cpp_cuda)]
int __sizeOf(T v)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "sizeof($T0)";
    case cuda: __intrinsic_asm "sizeof($T0)";
    }
}

__generic<T>
[__readNone]
[require(cpp_cuda)]
int __alignOf()
{
    __target_switch
    {
    case cuda :
    case cpp :
        __intrinsic_asm "SLANG_ALIGN_OF($[0])", T;
    }
}

__generic<T>
[__readNone]
[require(cpp_cuda)]
int __alignOf(T v)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "SLANG_ALIGN_OF($T0)";
    case cuda: __intrinsic_asm "SLANG_ALIGN_OF($T0)";
    }
}

// It would be nice to have offsetof equivalent, but it's not clear how that would work in terms of the Slang language.
// Here we allow calculating the offset of a field in bytes from an *instance* of the type.
__generic<T,F>
[__readNone]
[require(cpp_cuda)]
int __offsetOf(in T t, in F field)
{
    __target_switch
    {
    case cpp: __intrinsic_asm "int(((char*)&($1)) - ((char*)&($0))";
    case cuda: __intrinsic_asm "int(((char*)&($1)) - ((char*)&($0)))";
    }
}

/// Mark beginning of "interlocked" operations in a fragment shader.
[require(glsl_spirv, GL_ARB_fragment_shader_interlock, fragment)]
__intrinsic_op($(kIROp_BeginFragmentShaderInterlock))
void beginInvocationInterlock();

/// Mark end of "interlocked" operations in a fragment shader.
[require(glsl_spirv, GL_ARB_fragment_shader_interlock, fragment)]
__intrinsic_op($(kIROp_EndFragmentShaderInterlock))
void endInvocationInterlock();

// Operators to apply to `enum` types

//@ hidden:

__generic<E : __EnumType>
__intrinsic_op($(kIROp_Eql))
bool operator==(E left, E right);

__generic<E : __EnumType>
__intrinsic_op($(kIROp_Neq))
bool operator!=(E left, E right);

//@ hidden:

// public interfaces for generic arithmetic types.

__generic<T : IComparable>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool operator<(T v0, T v1)
{
    return v0.lessThan(v1);
}
__generic<T : IComparable>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool operator>(T v0, T v1)
{
    return v1.lessThan(v0);
}
__generic<T : IComparable>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool operator ==(T v0, T v1)
{
    return v0.equals(v1);
}
__generic<T : IComparable>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool operator >=(T v0, T v1)
{
    return v1.lessThanOrEquals(v0);
}
__generic<T : IComparable>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool operator <=(T v0, T v1)
{
    return v0.lessThanOrEquals(v1);
}
__generic<T : IComparable>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool operator !=(T v0, T v1)
{
    return !v0.equals(v1);
}

${{{{
const char* arithmeticInterfaces[] = {"IArithmetic", "IFloat"};
const char* attribs[] = {"", "[TreatAsDifferentiable]"};
for (Index i = 0; i < 2; i++) {
    const auto interfaceName = arithmeticInterfaces[i];
    const auto attrib = attribs[i];
    Index overloadRank = i - 3;
}}}}
$(attrib)
__generic<T : $(interfaceName)>
[__unsafeForceInlineEarly]
[OverloadRank($(overloadRank))]
T operator +(T v0, T v1)
{
    return v0.add(v1);
}

$(attrib)
__generic<T : $(interfaceName)>
[__unsafeForceInlineEarly]
[OverloadRank($(overloadRank))]
T operator -(T v0, T v1)
{
    return v0.sub(v1);
}

$(attrib)
__generic<T : $(interfaceName)>
[__unsafeForceInlineEarly]
[OverloadRank($(overloadRank))]
T operator *(T v0, T v1)
{
    return v0.mul(v1);
}

$(attrib)
__generic<T : $(interfaceName)>
[__unsafeForceInlineEarly]
[OverloadRank($(overloadRank))]
T operator /(T v0, T v1)
{
    return v0.div(v1);
}

$(attrib)
__generic<T : $(interfaceName)>
[__unsafeForceInlineEarly]
[OverloadRank($(overloadRank))]
T operator %(T v0, T v1)
{
    return v0.mod(v1);
}

$(attrib)
__generic<T : $(interfaceName)>
[__unsafeForceInlineEarly]
[OverloadRank($(overloadRank))]
__prefix T operator -(T v0)
{
    return v0.neg();
}

${{{{
    } // foreach ["IArithmetic", "IFloat"]
}}}}

__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
T operator &(T v0, T v1)
{
    return v0.bitAnd(v1);
}
__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
T operator &&(T v0, T v1)
{
    return v0.and(v1);
}

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool and(bool v0, bool v1)
{
    return __and(v0, v1);
}

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
__intrinsic_op($(kIROp_And))
vector<bool, N> and<let N : int>(vector<bool, N> v0, vector<bool, N> v1);

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
vector<bool, N> and<let N : int>(bool b, vector<bool, N> v)
{
    return and(vector<bool, N>(b), v);
}

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
vector<bool, N> and<let N : int>(vector<bool, N> v, bool b)
{
    return and(v, vector<bool, N>(b));
}

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
bool or(bool v0, bool v1)
{
    return __or(v0, v1);
}

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
__intrinsic_op($(kIROp_Or))
vector<bool, N> or<let N : int>(vector<bool, N> v0, vector<bool, N> v1);

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
vector<bool, N> or<let N : int>(bool b, vector<bool, N> v)
{
    return or(vector<bool, N>(b), v);
}

[__unsafeForceInlineEarly]
[OverloadRank(-10)]
vector<bool, N> or<let N : int>(vector<bool, N> v, bool b)
{
    return or(v, vector<bool, N>(b));
}

__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
T operator |(T v0, T v1)
{
    return v0.bitOr(v1);
}
__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
T operator ||(T v0, T v1)
{
    return v0.or(v1);
}
__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
T operator ^(T v0, T v1)
{
    return v0.bitXor(v1);
}
__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
__prefix T operator ~(T v0)
{
    return v0.bitNot();
}
__generic<T : ILogical>
[__unsafeForceInlineEarly]
[OverloadRank(-10)]
__prefix T operator !(T v0)
{
    return v0.not();
}

// The operator overloads defined above already allows Enum types to be used
// in logical operators, but we still provide overloads for __EnumTypes and map
// them directly to intrinsic op to allow constant propagation at AST level to
// work on enum types.

__generic<T : __EnumType>
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_BitAnd))
T operator &(T v0, T v1);

__generic<T : __EnumType>
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_BitOr))
T operator |(T v0, T v1);

__generic<T : __EnumType>
[__unsafeForceInlineEarly]
__intrinsic_op($(kIROp_BitNot))
__prefix T operator ~(T v0);


// IR level type traits.

__generic<T>
__intrinsic_op($(kIROp_undefined))
T __declVal();

__generic<T>
__intrinsic_op($(kIROp_DefaultConstruct))
T __default();

__generic<T, U>
__intrinsic_op($(kIROp_TypeEquals))
bool __type_equals_impl(T t, U u);

__generic<T, U>
[__unsafeForceInlineEarly]
bool __type_equals(T t, U u)
{
    return __type_equals_impl(__declVal<T>(), __declVal<U>());
}

__generic<T, U>
[__unsafeForceInlineEarly]
bool __type_equals()
{
    return __type_equals_impl(__declVal<T>(), __declVal<U>());
}

__generic<T>
__intrinsic_op($(kIROp_IsBool))
bool __isBool_impl(T t);

__generic<T>
[__unsafeForceInlineEarly]
bool __isBool()
{
    return __isBool_impl(__declVal<T>());
}

__generic<T>
__intrinsic_op($(kIROp_IsInt))
bool __isInt_impl(T t);

__generic<T>
[__unsafeForceInlineEarly]
bool __isInt()
{
    return __isInt_impl(__declVal<T>());
}

__generic<T>
__intrinsic_op($(kIROp_IsFloat))
bool __isFloat_impl(T t);

__generic<T>
__intrinsic_op($(kIROp_IsHalf))
bool __isHalf_impl(T t);

__generic<T>
[__unsafeForceInlineEarly]
bool __isFloat()
{
    return __isFloat_impl(__declVal<T>());
}

__generic<T>
[__unsafeForceInlineEarly]
bool __isHalf()
{
    return __isHalf_impl(__declVal<T>());
}

__generic<T>
__intrinsic_op($(kIROp_IsUnsignedInt))
bool __isUnsignedInt_impl(T t);

__generic<T>
[__unsafeForceInlineEarly]
bool __isUnsignedInt()
{
    return __isUnsignedInt_impl(__declVal<T>());
}

__generic<T>
__intrinsic_op($(kIROp_IsSignedInt))
bool __isSignedInt_impl(T t);

__generic<T>
[__unsafeForceInlineEarly]
bool __isSignedInt()
{
    return __isSignedInt_impl(__declVal<T>());
}

__generic<T>
__intrinsic_op($(kIROp_IsVector))
bool __isVector_impl(T t);

__generic<T>
[__unsafeForceInlineEarly]
bool __isVector()
{
    return __isVector_impl(__declVal<T>());
}

__generic<T>
__intrinsic_op($(kIROp_GetNaturalStride))
int __naturalStrideOf_impl(T v);

__generic<T>
[__unsafeForceInlineEarly]
int __naturalStrideOf()
{
    return __naturalStrideOf_impl(__declVal<T>());
}

__intrinsic_op($(kIROp_AlignOf))
int __alignOf_intrinsic_impl<T>(T t);

[ForceInline]
int __alignOf_intrinsic<T>()
{
    return __alignOf_intrinsic_impl<T>(__default<T>());
}

[__unsafeForceInlineEarly]
int32_t __elemToByteOffset<T>(int32_t elemOffset)
{
    return elemOffset * __naturalStrideOf<T>();
}

[__unsafeForceInlineEarly]
int32_t __byteToElemOffset<T>(int32_t byteOffset)
{
    return byteOffset / __naturalStrideOf<T>();
}

__intrinsic_op($(kIROp_TreatAsDynamicUniform))
T asDynamicUniform<T>(T v);

__generic<T>
__intrinsic_op( $(kIROp_GetLegalizedSPIRVGlobalParamAddr))
__Addr<T> __getLegalizedSPIRVGlobalParamAddr(T val);

__intrinsic_op($(kIROp_RequireComputeDerivative))
void __requireComputeDerivative();

__intrinsic_op($(kIROp_RequireMaximallyReconverges))
void __requireMaximallyReconverges();

__intrinsic_op($(kIROp_RequireQuadDerivatives))
void __requireQuadDerivatives();

//@ public:
/// @category misc_types 
enum MemoryOrder
{
    /// No memory operation ordering constraints
    Relaxed = $(kIRMemoryOrder_Relaxed),
    /// Ensures that all subsequent memory operations in the same thread are not reordered before it
    Acquire = $(kIRMemoryOrder_Acquire),
    /// Ensures that all prior memory operations in the same thread are not reordered after it
    Release = $(kIRMemoryOrder_Release),
    /// Combines both acquire and release semantics
    AcquireRelease = $(kIRMemoryOrder_AcquireRelease),
    /// Provides the strongest ordering: total order exists between all SeqCst operations
    SeqCst = $(kIRMemoryOrder_SeqCst),
}

/// Represents types that can be used in any atomic operations.
/// Implemented by builtin scalar types: `int`, `uint`, `int64_t`, `uint64_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `float`, `double` and `half`.
[sealed] interface IAtomicable {}

/// Represents types that can be used in atomic arithmetic operations.
/// Implemented by builtin scalar types: `int`, `uint`, `int64_t`, `uint64_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `float`, `double` and `half`.
[sealed] interface IArithmeticAtomicable : IAtomicable, IArithmetic {}

/// Represents types that can be used in atomic bit operations.
/// Implemented by builtin scalar types: `int`, `uint`, `int64_t`, `uint64_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`.
[sealed] interface IBitAtomicable : IArithmeticAtomicable, IInteger {}

extension int : IBitAtomicable {}
extension uint : IBitAtomicable {}
extension int64_t : IBitAtomicable {}
extension uint64_t : IBitAtomicable {}
extension double : IArithmeticAtomicable {}
extension float : IArithmeticAtomicable {}
extension half : IArithmeticAtomicable {}

/// A wrapper for `IAtomicable` types to introduce atomic load and store
/// operations. Values of this type are to be stored in buffers or groupshared
/// memory.
///
/// All operations take a `MemoryOrder` parameter which influenced the
/// semantics of the performed operation
///
/// All operations take place at the device scope.
///
/// Operators `+=`, `-=`, `&=`, `|=`, `^=`, `++`, `--` are overloaded to
/// operate on `Atomic<T>`.
__magic_type(AtomicType)
__intrinsic_type($(kIROp_AtomicType))
[require(cuda_glsl_hlsl_metal_spirv_wgsl)]
struct Atomic<T : IAtomicable>
{
    /// Atomically load the stored `T` value
    __intrinsic_op($(kIROp_AtomicLoad))
    [__ref] T load(MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically store a new `T` value
    __intrinsic_op($(kIROp_AtomicStore))
    [__ref] void store(T newValue, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically replace the stored `T` value with a new `T` value and return
    /// replaced value
    __intrinsic_op($(kIROp_AtomicExchange))
    [__ref] T exchange(T newValue, MemoryOrder order = MemoryOrder.Relaxed); // returns old value

    /// Atomically replace and return the stored `T` value with a new `T` value
    /// only if the stored value is equal to the specified comparison value.
    ///
    /// If the comparison value is equal to the stored value, then the
    /// `successOrder` `MemoryOrder` is used, otherwise the `failOrder`
    /// `MemoryOrder` is used. 
    /// 
    /// `successOrder` must be at least as strong as `failOrder`
    ///
    /// `failOrder` must not be `MemoryOrder.Release` or `MemoryOrder.AcquireRelease`
    __intrinsic_op($(kIROp_AtomicCompareExchange))
    [__ref] T compareExchange(
        T compareValue,
        T newValue,
        MemoryOrder successOrder = MemoryOrder.Relaxed,
        MemoryOrder failOrder = MemoryOrder.Relaxed);
}

/// These additional members are only available when `T` conforms to `IArithmeticAtomicable`.
extension<T : IArithmeticAtomicable> Atomic<T>
{
    /// Atomically adds the given value to the stored value and returns the
    /// original stored value.
    __intrinsic_op($(kIROp_AtomicAdd))
    [__ref] T add(T value, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically subtracts the given value from the stored value and returns
    /// the original stored value.
    __intrinsic_op($(kIROp_AtomicSub))
    [__ref] T sub(T value, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically computes the maximum of the stored value and the given
    /// value, storing the result and returning the original stored value.
    __intrinsic_op($(kIROp_AtomicMax))
    [__ref] T max(T value, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically computes the minimum of the stored value and the given
    /// value, storing the result and returning the original stored value.
    __intrinsic_op($(kIROp_AtomicMin))
    [__ref] T min(T value, MemoryOrder order = MemoryOrder.Relaxed);
}

/// These additional members are only available when `T` conforms to `IBitAtomicable`.
extension<T : IBitAtomicable> Atomic<T>
{
    /// Atomically performs a bitwise AND operation between the stored value
    /// and the given value, storing the result and returning the original
    /// stored value.
    __intrinsic_op($(kIROp_AtomicAnd))
    [__ref] T and(T value, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically performs a bitwise OR operation between the stored value and
    /// the given value, storing the result and returning the original stored
    /// value.
    __intrinsic_op($(kIROp_AtomicOr))
    [__ref] T or(T value, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically performs a bitwise XOR operation between the stored value
    /// and the given value, storing the result and returning the original
    /// stored value.
    __intrinsic_op($(kIROp_AtomicXor))
    [__ref] T xor(T value, MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically increments the stored value and returns the original stored
    /// value.
    __intrinsic_op($(kIROp_AtomicInc))
    [__ref] T increment(MemoryOrder order = MemoryOrder.Relaxed);

    /// Atomically decrements the stored value and returns the original stored
    /// value.
    __intrinsic_op($(kIROp_AtomicDec))
    [__ref] T decrement(MemoryOrder order = MemoryOrder.Relaxed);
}

//@ hidden:

__generic<T : IArithmeticAtomicable>
[ForceInline]
T operator +=(__ref Atomic<T> v, T value)
{
    return v.add(value) + value;
}
__generic<T : IArithmeticAtomicable>
[ForceInline]
T operator -=(__ref Atomic<T> v, T value)
{
    return v.sub(value) - value;
}
__generic<T : IBitAtomicable>
[ForceInline]
T operator &=(__ref Atomic<T> v, T value)
{
    return v.and(value) & value;
}
__generic<T : IBitAtomicable>
[ForceInline]
T operator |=(__ref Atomic<T> v, T value)
{
    return v.or(value) | value;
}
__generic<T : IBitAtomicable>
[ForceInline]
T operator ^=(__ref Atomic<T> v, T value)
{
    return v.xor(value) ^ value;
}

__generic<T : IBitAtomicable>
[ForceInline]
__prefix T operator ++(__ref Atomic<T> v)
{
    return v.increment() + T(1);
}
__generic<T : IBitAtomicable>
[ForceInline]
__postfix T operator ++(__ref Atomic<T> v)
{
    return v.increment();
}
__generic<T : IBitAtomicable>
[ForceInline]
__prefix T operator --(__ref Atomic<T> v)
{
    return v.decrement() - T(1);
}
__generic<T : IBitAtomicable>
[ForceInline]
__postfix T operator --(__ref Atomic<T> v)
{
    return v.decrement();
}
// Binding Attributes

//@public:

/// Declare the Vulkan binding location of a global shader variable.
/// @param binding The binding location.
/// @param set The descriptor set index of the binding.
__attributeTarget(DeclBase)
attribute_syntax [vk_binding(binding: int, set: int = 0)]			: GLSLBindingAttribute;

/// Declare the Vulkan binding location of a global shader variable.
/// @param binding The binding location.
/// @param set The descriptor set index of the binding.
__attributeTarget(DeclBase)
attribute_syntax [gl_binding(binding: int, set: int = 0)]			: GLSLBindingAttribute;


/// Mark a global variable as a Vulkan shader record.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_shader_record]			                        : ShaderRecordAttribute;

/// Mark a global variable as a Vulkan shader record.
__attributeTarget(VarDeclBase)
attribute_syntax [shader_record]			                        : ShaderRecordAttribute;

/// Mark a global variable as a Vulkan push constant.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_push_constant]									: PushConstantAttribute;

/// Mark a global variable as a Vulkan push constant.
__attributeTarget(VarDeclBase)
attribute_syntax [push_constant]									: PushConstantAttribute;

/// Mark a global variable as a Vulkan specialization constant.
__attributeTarget(VarDeclBase)
attribute_syntax[vk_specialization_constant] : SpecializationConstantAttribute;
/// Mark a global variable as a Vulkan specialization constant.
__attributeTarget(VarDeclBase)
attribute_syntax[SpecializationConstant] : SpecializationConstantAttribute;
/// Mark a global variable as a Vulkan specialization constant.
/// @param location The index of the specialization constant.
__attributeTarget(VarDeclBase)
attribute_syntax[vk_constant_id(location: int)] : VkConstantIdAttribute;

/// Declare the Vulkan location of a global variable.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_location(location : int)] : GLSLLocationAttribute;

/// Declare the Vulkan binding index of a global variable.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_index(index : int)] : GLSLIndexAttribute;

/// Declare offset for a struct field. Applies to all kinds of structs when targeting Vulkan.
/// Applies only to structs that are directly used as interface blocks (such as push constants and uniforms)
/// when targeting GLSL, as GLSL does not support the `offset` qualifier on regular structs.
/// This attribute has no effect on other targets.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_offset(index: int)] : VkStructOffsetAttribute;

/// @deprecated
/// Use `spirv_asm` instead for inline SPIR-V assembly.
__attributeTarget(FuncDecl)
attribute_syntax [vk_spirv_instruction(op : int, set : String = "")]     : SPIRVInstructionOpAttribute;

/// Declare the Vulkan input attachment index of a global variable.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_input_attachment_index(location : int)] : GLSLInputAttachmentIndexLayoutAttribute;

/// @internal
__attributeTarget(FuncDecl)
attribute_syntax [spv_target_env_1_3] : SPIRVTargetEnv13Attribute;

/// @internal
__attributeTarget(VarDeclBase)
attribute_syntax [disable_array_flattening] : DisableArrayFlatteningAttribute;

/// Marks a enum type as an unscoped enum. The enum cases of an unscoped enum are not scoped within the enum type, and can be
/// referenced directly without the enum type name.
__attributeTarget(EnumDecl)
attribute_syntax [UnscopedEnum] : UnscopedEnumAttribute;

/// Marks a enum type as a bit flag enum. Bit flag enums will have their enum cases assigned to powers of 2 instead of consecutive integers.
///
__attributeTarget(EnumDecl)
attribute_syntax[Flags] : FlagsAttribute;

// Statement Attributes

/// A hint to the downstream compiler to unroll the loop until the specified number of iterations reached.
/// This attribute does not affect Slang compiler's behavior.
/// To unroll a loop in the Slang compiler before emitting target code, use the `[ForceUnroll]` attribute.
__attributeTarget(LoopStmt)
attribute_syntax [unroll(count: int = 0)]   : UnrollAttribute;

/// Instructs the Slang compiler to unroll the loop until the specified number of iterations before
/// emiting targett code.
/// @param count The maximun number of iterations to unroll the loop.
__attributeTarget(LoopStmt)
attribute_syntax [ForceUnroll(count: int = 0)]   : ForceUnrollAttribute;

/// A hint to the downstream compiler to preserve a loop.
__attributeTarget(LoopStmt)
attribute_syntax [loop]                 : LoopAttribute;

/// Used on loop statements as a hint to the downstream compiler to perform less aggressive optimization on the loop
/// in favor of faster compilation time.
/// This attribute has no effect on targets other than HLSL.
__attributeTarget(LoopStmt)
attribute_syntax [fastopt]              : FastOptAttribute;

/// A hint to the downstream compiler that UAV conditions are allowed in the loop.
/// This attribute has no effect on targets other than HLSL.
/// @deprecated
__attributeTarget(LoopStmt)
attribute_syntax [allow_uav_condition]  : AllowUAVConditionAttribute;

__attributeTarget(LoopStmt)
attribute_syntax [MaxIters(count)]      : MaxItersAttribute;

/// A hint to the downstream compiler to flatten an if statement.
__attributeTarget(IfStmt)
attribute_syntax [flatten]              : FlattenAttribute;

/// A hint to the downstream compiler to preserve the branching behavior of an if statement.
__attributeTarget(IfStmt)
__attributeTarget(SwitchStmt)
attribute_syntax [branch]               : BranchAttribute;

/// A hint to the downstream compiler to preserve the `switch` statement as is.
/// This attribute has no effect on targets other than HLSL.
__attributeTarget(SwitchStmt)
attribute_syntax [forcecase]            : ForceCaseAttribute;

/// A hint to the downstream compiler to translate the `switch` statement into subroutine calls.
/// This attribute has no effect on targets other than HLSL.
__attributeTarget(SwitchStmt)
attribute_syntax [call]                 : CallAttribute;

// Entry-point Attributes

// All Stages

/// Marks a function as a shader entry point.
/// @param stage The stage of the shader. Must be one of "vertex", "fragment", "compute", "geometry", "hull", "domain", "raygeneration",
// "intersection", "anyhit", "closesthit", "miss", "callable", "task", "mesh".
__attributeTarget(FuncDecl)
attribute_syntax [shader(stage)]    : EntryPointAttribute;

/// Marks a function as a shader entry point.
/// @param stage The stage of the shader. Must be one of "vertex", "fragment", "compute", "geometry", "hull", "domain", "raygeneration",
// "intersection", "anyhit", "closesthit", "miss", "callable", "task", "mesh".
__attributeTarget(FuncDecl)
attribute_syntax [Shader(stage)]    : EntryPointAttribute;

// Hull Shader
/// Used on an hull shader entrypoint to declare the upperbound of the tessellation factor that the hull shader can return.
/// @param factor The maximum tessellation factor the hull shader can return.
__attributeTarget(FuncDecl)
attribute_syntax [maxtessfactor(factor: float)]     : MaxTessFactorAttribute;

/// Used on an hull shader entrypoint to declare the number of control points the hull shader will produce per thread.
/// @param count The number of control points the hull shader will produce per thread.
/// @remarks The attribute indicates be the number of times the hull shader function will be executed.
__attributeTarget(FuncDecl)
attribute_syntax [outputcontrolpoints(count: int)]  : OutputControlPointsAttribute;

/// Used on an hull shader entrypoint to declare the output primitive type of the tessellator.
/// @param topology The output primitive type, must be one of "point", "line", "triangle_cw", and "triangle_ccw".
__attributeTarget(FuncDecl)
attribute_syntax [outputtopology(topology)]         : OutputTopologyAttribute;

/// Used on an hull shader entrypoint to specify the patch type used by the hull shader.
/// @param mode The patch type used by the hull shader. Valid values are "tri", "quad" and "isoline".
__attributeTarget(FuncDecl)
attribute_syntax [partitioning(mode)]               : PartitioningAttribute;

/// Used on a hull shader entrypoint to specify the associated function that computes the patch constant data.
/// @param name The name of the function (in string literal) that computes the patch constant data.
__attributeTarget(FuncDecl)
attribute_syntax [patchconstantfunc(name)]          : PatchConstantFuncAttribute;

// Hull/Domain Shader

/// Used on an hull shader entrypoint to specify the patch type used by the hull shader.
/// @param patchType The patch type used by the hull shader. Valid values are "tri", "quad" and "isoline".
__attributeTarget(FuncDecl)
attribute_syntax [domain(patchType)]   : DomainAttribute;

// Geometry Shader

/// Used on a geometry shader entry point to specify the maximum number of vertices that the geometry shader can output.
/// @param count The maximum number of vertices that the geometry shader can output.
__attributeTarget(FuncDecl)
attribute_syntax [maxvertexcount(count: int)]   : MaxVertexCountAttribute;

/// Used on a geometry shader entry point to specify the number of instances to execute for each input primitive.
/// @param count The number of instances to execute for each input primitive. 
/// @remarks When using this attribute, a geometry shader can declare a parameter with `SV_GSInstanceID` semantic to get the instance index.
__attributeTarget(FuncDecl)
attribute_syntax [instance(count: int)]         : InstanceAttribute;

// Fragment ("Pixel") Shader
/// Used on a fragment shader entry point to specify that early depth stencil testing can be enabled.
__attributeTarget(FuncDecl)
attribute_syntax [earlydepthstencil]    : EarlyDepthStencilAttribute;

// Compute Shader

/// Specifies the size of the thread group a compute shader.
/// @param x The number of threads in the x dimension of a thread group.
/// @param y The number of threads in the y dimension of a thread group.
/// @param z The number of threads in the z dimension of a thread group.
__attributeTarget(FuncDecl)
attribute_syntax [numthreads(x: int, y: int = 1, z: int = 1)]   : NumThreadsAttribute;

/// Specifies the size of the thread group a compute shader.
/// @param x The number of threads in the x dimension of a thread group.
/// @param y The number of threads in the y dimension of a thread group.
/// @param z The number of threads in the z dimension of a thread group.
__attributeTarget(FuncDecl)
attribute_syntax [NumThreads(x: int, y: int = 1, z: int = 1)]   : NumThreadsAttribute;

/// Indicate a compute shader entry point is only compatible with the specified wave size.
/// @param numLanes The wave size this shader entrypoint is compatible with. Must be one of 4, 8, 16, 32, 64, 128.
__attributeTarget(FuncDecl)
attribute_syntax [WaveSize(numLanes: int)]   : WaveSizeAttribute;

//@hidden:
__attributeTarget(VarDeclBase)
attribute_syntax [__vulkanRayPayload(location : int = -1)] : VulkanRayPayloadAttribute;

__attributeTarget(VarDeclBase)
attribute_syntax [__vulkanCallablePayload(location : int = -1)] : VulkanCallablePayloadAttribute;

__attributeTarget(VarDeclBase)
attribute_syntax [__vulkanHitObjectAttributes(location : int = -1)] : VulkanHitObjectAttributesAttribute;

__attributeTarget(VarDeclBase)
attribute_syntax [__vulkanHitAttributes] : VulkanHitAttributesAttribute;

//@public:
/// Mark a function or a property or subscript accessor as mutating. A mutating function receives the implicit `this` parameter
/// as an `inout` parameter, so that mutations to members access from `this` argument will be visible to the caller.
///
/// @remarks
/// By default, Slang treats all member functions as non-mutating. For example, consider the following function:
/// ```csharp
/// struct S
/// {
///    int x;
///    void foo()
///    {
///       x = 1; // error: `x` is not an l-value.
///    }
/// }
///
/// ```
/// The line `x = 1` will lead to a compile time error because by-default, all member methods in Slang are non-mutating. To
/// allow `foo` to modify `x`, you can use `[mutating]` to mark the function as such:
/// ```csharp
/// struct S
/// {
///    int x;
///    [mutating]
///    void foo()
///    {
///       x = 1; // ok
///    }
/// }
/// ```
/// @see `[nonmutating]`.
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [mutating] : MutatingAttribute;

/// Marks a function or a property and subscript accessor as non-mutating. A non-mutating function receives the implicit `this` parameter
/// as an `in` parameter, so mutations to members accessed from `this` argument will be prohibited by the compiler.
/// @remarks
/// Member functions of a type are non-mutating by default, so this attribute is not necessary in most cases.
/// However, the `set` accessor of a property or subscript is mutating by default, and you can use `[nonmutating]` to mark it as non-mutating.
/// For example:
/// ```csharp
/// struct S
/// {
///    int* ptr_x;
///    property x : int
///    {
///      get { return *ptr_x; }
///
///      [nonmutating]
///      set { *ptr_x = value; }
///    }
/// }
/// uniform S s; // `s` is not mutable.
/// void test() { s.x = 1; } // OK, because the `set` accessor is non-mutating.
/// ```
/// In the above example, the property `x` reads and writes to a memory location pointed to by `ptr_x`. Therefore, the `set` accessor is not actually
/// modifying any field of `S`, and does not need to take `this` as an `inout` parameter. Using `[nonmutating]` here on the set accessor will allow
/// it to be called with a non-mutating value of `S`.
/// @see `[mutating]`.
///
__attributeTarget(AccessorDecl)
attribute_syntax [nonmutating] : NonmutatingAttribute;

/// @internal
/// Marks a member function to make `this` argument passed by const reference.
__attributeTarget(FunctionDeclBase)
attribute_syntax [constref] : ConstRefAttribute;

/// @internal
/// Marks a member function to make `this` argument passed by reference.
__attributeTarget(FunctionDeclBase)
attribute_syntax [__ref] : RefAttribute;

/// @internal
/// Indicates that a function computes its result as a function of its arguments without loading/storing any memory or other state.
///
/// This is equivalent to the LLVM `readnone` function attribute.
__attributeTarget(FunctionDeclBase)
attribute_syntax [__readNone] : ReadNoneAttribute;

/// Represents the applicable target for an attribute.
enum _AttributeTargets
{
    Struct = $( (int) UserDefinedAttributeTargets::Struct), /// Struct types.
    Var = $( (int) UserDefinedAttributeTargets::Var), /// Global and local variables and constants.
    Function = $( (int) UserDefinedAttributeTargets::Function), /// Functions or member functions.
    Param = $( (int) UserDefinedAttributeTargets::Param), /// Function parameters.
};

/// Mark a struct type as a user defined attribute type.
/// @param target The type of declarations to which the attribute can be applied.
__attributeTarget(StructDecl)
attribute_syntax [__AttributeUsage(target : _AttributeTargets)] : AttributeUsageAttribute;

/// Specify the storage format of a read-write texture. Can only be used on a texture typed struct field or global parameter.
/// @param format The storage format of the texture.
/// @see Please refer to `_Texture` for a complete list of allowed format strings.
__attributeTarget(VarDeclBase)
attribute_syntax [format(format : String)] : FormatAttribute;

/// Specify the storage format of a read-write texture. Can only be used on a texture typed struct field or global parameter.
/// This is an alias of the `[format]` attribute.
/// @param format The storage format of the texture.
/// @see Please refer to `_Texture` for a complete list of allowed format strings.
__attributeTarget(VarDeclBase)
attribute_syntax [vk_image_format(format : String)] : FormatAttribute;

__attributeTarget(Decl)
attribute_syntax [allow(diagnostic: String)] : AllowAttribute;

/// Mark declaration to require a specific target capability.
/// @param capability The required capability.
__attributeTarget(Decl)
attribute_syntax [require(capability)] : RequireCapabilityAttribute;

// Linking
__attributeTarget(Decl)
attribute_syntax [__extern] : ExternAttribute;

__attributeTarget(FunctionDeclBase)
attribute_syntax [__unsafeForceInlineEarly] : UnsafeForceInlineEarlyAttribute;

/// Perform inlining of the function at the call site during Slang compilation.
/// @remarks By default Slang does not inline user defined functions, and will preserve the function call hierarchy in the generated code.
/// Use this attribute on a function to force the Slang compiler to inline the function before emitting target code.
__attributeTarget(FunctionDeclBase)
attribute_syntax [ForceInline] : ForceInlineAttribute;

/// @internal
/// Specify the overload rank of a function for overload resolution.
__attributeTarget(FunctionDeclBase)
attribute_syntax [OverloadRank] : OverloadRankAttribute;

/// @experimental
/// Mark a function as imported from a DLL. Valid only on CPU host targets.
__attributeTarget(FuncDecl)
attribute_syntax [DllImport(modulePath: String)] : DllImportAttribute;

/// @experimental
/// Mark a function to be exported as a DLL symbol. Valid only on CPU host targets.
__attributeTarget(FuncDecl)
attribute_syntax [DllExport] : DllExportAttribute;

/// @experimental
/// Mark a function as a pytorch kernel entrypoint.
__attributeTarget(FuncDecl)
attribute_syntax [TorchEntryPoint] : TorchEntryPointAttribute;

/// @experimental
/// Mark a function for export as a CUDA device function. Valid only on CUDA target.
__attributeTarget(FuncDecl)
attribute_syntax [CudaDeviceExport] : CudaDeviceExportAttribute;

/// @experimental
/// Mark a function for export as a CUDA host function. Valid only on CUDA target.
__attributeTarget(FuncDecl)
attribute_syntax [CudaHost] : CudaHostAttribute;

/// @experimental
/// Mark a function for export as a CUDA kernel function. Valid only on CUDA target.
__attributeTarget(FuncDecl)
attribute_syntax [CudaKernel] : CudaKernelAttribute;

/// @experimental
/// Mark a function for export as a CUDA device function. Valid only on CUDA target.
__attributeTarget(FuncDecl)
attribute_syntax [CUDADeviceExport] : CudaDeviceExportAttribute;

/// @experimental
/// Mark a function for export as a CUDA host function. Valid only on CUDA target.
__attributeTarget(FuncDecl)
attribute_syntax [CUDAHost] : CudaHostAttribute;

/// @experimental
/// Mark a function for export as a CUDA kernel function. Valid only on CUDA target.
__attributeTarget(FuncDecl)
attribute_syntax [CUDAKernel] : CudaKernelAttribute;

/// @experimental
/// Mark a function for automatic pytorch binding.
__attributeTarget(FuncDecl)
attribute_syntax [AutoPyBindCUDA] : AutoPyBindCudaAttribute;

/// @experimental
/// Mark a type for export to slang-torch.
__attributeTarget(AggTypeDecl)
attribute_syntax [PyExport(name: String)] : PyExportAttribute;

/// @experimental
/// Mark an interface as a COM interface. Valid only on CPU target.
/// @param guid The GUID of the COM interface.
__attributeTarget(InterfaceDecl)
attribute_syntax [COM(guid: String)] : ComInterfaceAttribute;

// Inheritance Control

/// @experimental
/// Mark a type as sealed, preventing it from being inherited from or implemented by types defined in other modules.
/// @see `[open]`.
__attributeTarget(AggTypeDecl)
attribute_syntax [sealed] : SealedAttribute;

/// @experimental
/// Mark a type as open, allowing it from being inherited from or implemented by types defined in other modules.
/// This is the default behavior.
/// @see `[sealed]`.
__attributeTarget(AggTypeDecl)
attribute_syntax [open] : OpenAttribute;

/// @experimental
/// Mark an interface type to allow dynmaic dispatch, and declare the maximum size in bytes that an implementation type
/// of the interface can have.
__attributeTarget(InterfaceDecl)
attribute_syntax [anyValueSize(size:int)] : AnyValueSizeAttribute;

/// @experimental
/// Mark an interface type for specialization only. Such interface types cannot be used for dynamic dispatch.
__attributeTarget(InterfaceDecl)
attribute_syntax [Specialize] : SpecializeAttribute;

/// @internal
/// Marks a declaration as a builtin declaration.
__attributeTarget(DeclBase)
attribute_syntax [builtin] : BuiltinAttribute;

// @hidden:
__attributeTarget(DeclBase)
attribute_syntax[__AutoDiffBuiltin] : AutoDiffBuiltinAttribute;

__attributeTarget(DeclBase)
attribute_syntax [__requiresNVAPI] : RequiresNVAPIAttribute;

// @public:

/// Mark a type to require a target specific prelude.
/// The prelude will be included in the generated code for the specified target if the resulting code uses
/// the marked type.
__attributeTarget(AggTypeDecl)
attribute_syntax[RequirePrelude(target, prelude:String)] : RequirePreludeAttribute;

__attributeTarget(DeclBase)
attribute_syntax [__AlwaysFoldIntoUseSiteAttribute] : AlwaysFoldIntoUseSiteAttribute;

/// Inform the downstream compiler to not inline the function.
__attributeTarget(FunctionDeclBase)
attribute_syntax [noinline] : NoInlineAttribute;

__attributeTarget(StructDecl)
attribute_syntax [payload] : PayloadAttribute;

/// Mark a declaration as deprecated.
/// @param message The diagnostic message to show when the declaration is used.
__attributeTarget(DeclBase)
attribute_syntax [deprecated(message: String)] : DeprecatedAttribute;

/// Controls the behavior of the compiler when a differentiable function is detected to have side-effects.
/// @category misc_types
enum SideEffectBehavior
{
    /// Causes a warning if the method is detected to have side-effects
    Warn = 0,

    /// Suppresses the warning
    Allow = 1
};

/// Mark a differentiable function to prefer recomputation over checkpointing when a value computed in the primal pass is needed
/// during backward derivative propagation.
__attributeTarget(FunctionDeclBase)
attribute_syntax[PreferRecompute(behavior: SideEffectBehavior = SideEffectBehavior.Warn)] : PreferRecomputeAttribute;

/// Mark a differentiable function to prefer checkpointing over recomputation when a value computed in the primal pass is needed
/// during backward derivative propagation.
__attributeTarget(FunctionDeclBase)
attribute_syntax [PreferCheckpoint] : PreferCheckpointAttribute;

// @hidden:
__attributeTarget(DeclBase)
attribute_syntax [KnownBuiltin(name : String)] : KnownBuiltinAttribute;

__attributeTarget(FunctionDeclBase)
attribute_syntax [__GLSLRequireShaderInputParameter(parameterNumber:int)] : GLSLRequireShaderInputParameterAttribute;

__attributeTarget(FuncDecl)
attribute_syntax [noRefInline] : NoRefInlineAttribute;

// @public:

/// Mark a function's return value as non-uniform.
__attributeTarget(FunctionDeclBase)
attribute_syntax [NonUniformReturn] : NonDynamicUniformAttribute;

/// Mark a compute shader entry point to allow it to use implicit derivatives.
/// @remarks This attributes causes Slang to emit `DerivativeGroupQuadsNV` execution mode when producing SPIR-V. The attribute has no
/// effect on other targets.
__attributeTarget(FuncDecl)
attribute_syntax [DerivativeGroupQuad] : DerivativeGroupQuadAttribute;

/// Mark a compute shader entry point to allow it to use implicit derivatives.
/// @remarks This attributes causes Slang to emit `DerivativeGroupLinearNV` execution mode when producing SPIR-V. The attribute has no
/// effect on other targets.
__attributeTarget(FuncDecl)
attribute_syntax [DerivativeGroupLinear] : DerivativeGroupLinearAttribute;

/// Emits `MaximallyReconvergesKHR` execution mode when producing SPIR-V.
/// This attribute has no effect on other targets.
__attributeTarget(FuncDecl)
attribute_syntax [MaximallyReconverges] : MaximallyReconvergesAttribute;

/// Emits `QuadDerivativesKHR` execution mode when producing SPIR-V.
/// This attribute has no effect on other targets.
__attributeTarget(FuncDecl)
attribute_syntax [QuadDerivatives] : QuadDerivativesAttribute;

/// Emits `RequireFullQuadsKHR` execution mode when producing SPIR-V.
/// This attribute has no effect on other targets.
__attributeTarget(FuncDecl)
attribute_syntax [RequireFullQuads] : RequireFullQuadsAttribute;

__generic<T>
typealias NodePayloadPtr = Ptr<T, $( (uint64_t)AddressSpace::NodePayloadAMDX)>;

__attributeTarget(StructDecl)
attribute_syntax [raypayload] : RayPayloadAttribute;

