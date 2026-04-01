/// `[ForwardDerivative(fwdFn)]` attribute can be used to provide a forward-mode
/// derivative implementation.
/// Invoking `fwd_diff(decoratedFn)` will place a call to `fwdFn` instead of synthesizing
/// a derivative implementation.
/// The same behavior holds if `decoratedFn` is used in a differentiable context.
///
///
/// @remarks
/// - The signature of `fwdFn` must match the expected signature of `fwd_diff(decoratedFn)`.
/// - See the [reference](https://shader-slang.org/slang/user-guide/autodiff.html#fwd_difff--slang_function---slang_function) for `fwd_diff` for a full list of signature rules.
/// - See the [user guide's section](https://shader-slang.org/slang/user-guide/autodiff.html#user-defined-derivative-functions) on custom derivatives for an introduction to this approach. 
///
/// - This attribute can be used on generic functions, member functions and accessors.
/// - For generic functions, the generic signatures (parameters + constraints) of both functions must match exactly.
///
/// - The decorated function will be considered forward-differentiable. There is no need for a `[Differentiable]` tag. 
/// If the `[Differentiable]` tag is present, 
/// and no custom backward derivative is specified with `[BackwardDerivative]`, then the
/// Slang will use auto-diff to generate the backward=mode derivative, but will use the provided
/// derivative for forward-mode.
/// 
/// Example:
/// ```csharp
/// [ForwardDerivative(foo_fwd)]
/// T foo<T : IFloat, P : IArray<T>>(T x, P xarr) { /* ... */ }
///
/// DifferentialPair<T> foo_fwd<T : IFloat, P : IArray<T>>( // Use the same generic signature for a match.
///     DifferentialPair<T> x, P dp_xarr) { /* ... */ }
/// ```
/// 
/// For member functions, or functions nested inside namespaces, `fwdFn` may need a fully qualified name.
///
/// Example:
/// ```csharp
/// namespace A
/// {
///     DifferentialPair<float> foo(DifferentialPair<float> x) { /* ... */ }
/// }
/// [ForwardDerivative(A.foo)] // Use namespace and/or parent struct names
/// float bar(float x) { /* ... */ }
/// ```
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [ForwardDerivative(function)] : ForwardDerivativeAttribute;

/// `[BackwardDerivative(bwdFn)]` attribute can be used to provide a forward-mode
/// derivative implementation.
/// Invoking `bwd_diff(decoratedFn)` will place a call to `bwdFn` instead of synthesizing
/// a derivative implementation.
/// The same behavior holds if `decoratedFn` is used in a differentiable context.
///
/// @remarks
/// The signature of `bwdFn` must match the expected signature of `bwd_diff(decoratedFn)`.
/// See the [reference](https://shader-slang.org/slang/user-guide/autodiff.html#bwd_difff--slang_function---slang_function)
/// for `bwd_diff` for a full list of signature rules.
///
/// See the [user guide's section](https://shader-slang.org/slang/user-guide/autodiff.html#user-defined-derivative-functions) on custom derivatives for an introduction to custom
/// derivatives.
///
/// This attribute can be used on generic functions, member functions and accessors.
/// For generic functions, the generic signatures (parameters + constraints) of both functions
/// must match exactly. 
/// Overloaded functions are also supported. The compiler will attempt to resolve the overload
/// from the expected derivative signature. If it is unable to do so, it will issue a 
/// diagnostic error.
///
/// The decorated function will be considered differentiable.
/// There is no need for a `[Differentiable]` tag.
///
/// Example:
/// ```csharp
/// [BackwardDerivative(foo_bwd)]
/// T foo<T : IFloat, P : IArray<T>>(T x, P xarr) { /* ... */ }
///
/// void foo_bwd<T : IFloat, P : IArray<T>>( // Use the same generic signature for a match.
///     inout DifferentialPair<T> x, P dp_xarr, T.Differential dresult) { /* ... */ }
/// ```
///
/// For member functions, or functions nested inside namespaces, `bwdFn` may need to be a fully qualified 
/// name.
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [BackwardDerivative(function)] : BackwardDerivativeAttribute;

/// `[PrimalSubstitute(substFn)]` attribute denotes a substitute `substFn` that should be used for
/// differentiation instead of the original function. This serves as a sort of 'reference' implementation
/// where the original function cannot be differentiated (for whatever reason).
///
/// See the auto-diff user guide for more: https://shader-slang.org/slang/user-guide/autodiff.html#primal-substitute-functions
///
/// @example The following example shows in more detail on how primal substitute affects derivative computation.
/// ```csharp
/// [PrimalSubstitute(myFuncSubst)]
/// float myFunc(float x) { return x*x; }
///
/// [Differentiable]
/// float myFuncSubst(float x) { return x*x*x; }
///
/// [Differentiable]
/// float caller(float x) { return myFunc(x); }
///
/// let a = caller(4.0); /// a == 16.0 (calling myFunc)
/// let b = fwd_diff(caller)(diffPair(4.0, 1.0)).p; /// b == 64.0 (calling myFuncSubst)
/// let c = fwd_diff(caller)(diffPair(4.0, 1.0)).d; /// c == 48.0 (calling derivative of myFuncSubst)
/// ```
/// @remarks
/// `substFn` must have a function definition (i.e. a body).
/// 
/// In case that a function has both custom defined derivatives and a differentiable 
/// primal substitute, the primal substitute overrides the custom defined derivative 
/// on the original function. All calls to the original function will be translated 
/// into calls to the primal substitute first, and differentiation step follows after. 
/// This means that the derivatives of the primal substitute function will be used instead 
/// of the derivatives defined on the original function.
///
/// This attribute can be used on generic functions and member functions. 
/// For generic functions, the generic signatures (parameters + constraints) of both functions
/// must match exactly.
/// For member functions, or functions nested inside namespaces, `substFn` should be a fully qualified name.
/// `substFn`
/// Example:
/// ```csharp
/// namespace A
/// {
///     float foo(float x) { /* ... */ }
/// }
/// [PrimalSubstitute(A.foo)] // Use namespace and/or parent struct names
/// float bar(float x) { /* ... */ }
/// ```
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [PrimalSubstitute(function)] : PrimalSubstituteAttribute;

/// `[ForwardDerivativeOf(fn)]` is the back-reference version of `[ForwardDerivative(derivFn)]`
///
/// When used to decorate a function, the decorated function is considered the forward-derivative
/// implementation of the referenced function `fn`.
///
/// Apart from this, the semantics of the custom derivative are the same as for 
/// `[ForwardDerivative(derivFn)]`
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [ForwardDerivativeOf(function)] : ForwardDerivativeOfAttribute;

/// `[BackwardDerivativeOf(fn)]` is the back-reference version of `[BackwardDerivative(derivFn)]`
///
/// When used to decorate a function, the decorated function is considered the backward-derivative
/// implementation of the referenced function `fn`.
///
/// Apart from this, the semantics of the custom derivative are the same as for
/// `[BackwardDerivative(derivFn)]`
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [BackwardDerivativeOf(function)] : BackwardDerivativeOfAttribute;

/// `[PrimalSubstituteOf(fn)]` is the back-reference version of `[PrimalSubstitute(substFn)]`
///
/// When used to decorate a function, that function is considered the substitute for the 
/// referenced `fn`. 
/// Apart from this difference, the semantics of the substitution are the same as for 
/// `[PrimalSubstitute(substFn)]`
///
__attributeTarget(FunctionDeclBase)
attribute_syntax [PrimalSubstituteOf(function)] : PrimalSubstituteOfAttribute;

__attributeTarget(DeclBase)
attribute_syntax [DerivativeMember(memberName)] : DerivativeMemberAttribute;

// Exclude "this" parameter from differentiation. Effectively like putting a 'no_diff' on the 
// "this" parameter.
//
__attributeTarget(FunctionDeclBase)
attribute_syntax [NoDiffThis] : NoDiffThisAttribute;

// A 'none-type' that acts as a run-time sentinel for zero differentials. This is primarily
// for internal use.
//
[__AutoDiffBuiltin]
struct NullDifferential : IDifferentiable
{ 
    // for now, we'll use at least one field to make sure the type is non-empty
    uint dummy;
    typedef NullDifferential Differential;

    [Differentiable]
    [ForceInline]
    static Differential dzero() { return { 0 }; }

    [Differentiable]
    [ForceInline]
    static Differential dadd(Differential, Differential) { return { 0 }; }

    [Differentiable]
    [ForceInline]
    static Differential dmul<T: __BuiltinRealType>(T, Differential) { return { 0 }; }
};

// Existential check for null differential type
__intrinsic_op($(kIROp_IsDifferentialNull))
bool isDifferentialNull(IDifferentiable obj);

/// Represents a GPU view of a tensor.
__generic<T>
__magic_type(TensorViewType)
__intrinsic_type($(kIROp_TensorViewType))
struct TensorView
{
    [__NoSideEffect]
    [require(cuda)]
    Ptr<T> data_ptr()
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.data_ptr<$[0]>()", T;
        }
    }

    [__NoSideEffect]
    [require(cuda)]
    Ptr<T> data_ptr_at(uint index)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.data_ptr_at<$[0]>($1)", T;
        }
    }

    __generic<let N : int>
    [__NoSideEffect]
    [require(cuda)]
    Ptr<T> data_ptr_at(vector<uint, N> index)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.data_ptr_at<$[0]>($1)", T;
        }
    }

    __implicit_conversion($(kConversionCost_ImplicitDereference))
    __intrinsic_op($(kIROp_TorchTensorGetView))
    __init(TorchTensor<T> t);

    [__NoSideEffect]
    [require(cuda)]
    T load(uint x)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.load<$[0]>($1)", T;
        }
    }

    [__NoSideEffect]
    [require(cuda)]
    T load(uint x, uint y)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.load<$[0]>($1, $2)", T;
        }
    }

    [__NoSideEffect]
    [require(cuda)]
    T load(uint x, uint y, uint z)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.load<$[0]>($1, $2, $3)", T;
        }
    }

    [__NoSideEffect]
    [require(cuda)]
    T load(uint x, uint y, uint z, uint w)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.load<$[0]>($1, $2, $3, $4)", T;
        }
    }

    [__NoSideEffect]
    [require(cuda)]
    T load(uint i0, uint i1, uint i2, uint i3, uint i4)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.load<$[0]>($1, $2, $3, $4, $5)", T;
        }
    }

    __generic<let N : int>
    [__NoSideEffect]
    [require(cuda)]
    T load(vector<uint, N> index)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.load<$TR>($1)";
        }
    }

    [require(cuda)]
    void store(uint x, T val)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.store<$T2>($1, $2)";
        }
    }

    [require(cuda)]
    void store(uint x, uint y, T val)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.store<$T3>($1, $2, $3)";
        }
    }

    [require(cuda)]
    void store(uint x, uint y, uint z, T val)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.store<$T4>($1, $2, $3, $4)";
        }
    }

    [require(cuda)]
    void store(uint x, uint y, uint z, uint w, T val)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.store<$T5>($1, $2, $3, $4, $5)";
        }
    }

    [require(cuda)]
    void store(uint i0, uint i1, uint i2, uint i3, uint i4, T val)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.store<$T6>($1, $2, $3, $4, $5, $6)";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void store(vector<uint, N> index, T val)
    {
        __target_switch
        {
        case cuda:
            __intrinsic_asm "$0.store<$T2>($1, $2)";
        }
    }

    [require(cuda)]
    void InterlockedAdd(uint index, T val, out T oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicAdd($0.data_ptr_at<$T2>($1), $2)";
        }
    }
    
    __generic<let N:int>
    [require(cuda)]
    void InterlockedAdd(vector<uint, N> index, T val, out T oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicAdd($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [__readNone]
    [require(cuda)]
    uint dims()
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "$0.dimensionCount";
        }
    }

    [__readNone]
    [require(cuda)]
    uint size(uint i)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "$0.sizes[$1]";
        }
    }

    [__readNone]
    [require(cuda)]
    uint stride(uint i)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "$0.strides[$1]";
        }
    }

    __subscript(uint index) -> T
    {
        [ForceInline] [__NoSideEffect] get { return load(index); }
        
        [ForceInline]
        [nonmutating]
        set
        {
            store(index, newValue);
        }

        [__NoSideEffect]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1)", T;
            }
        }
    }
    __subscript(uint i1, uint i2) -> T
    {
        [ForceInline] [__NoSideEffect] get { return load(i1, i2); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i1, i2, newValue);
        }
        
        [__NoSideEffect]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1, $2)", T;
            }
        }
    }
    __subscript(uint2 i) -> T
    {
        [ForceInline] [__NoSideEffect] get { return load(i.x, i.y); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i.x, i.y, newValue);
        }

        [__NoSideEffect]
        [nonmutating]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1.x, $1.y)", T;
            }
        }
    }
    __subscript(uint i1, uint i2, uint i3) -> T
    {
        [ForceInline] [__NoSideEffect] get { return load(i1, i2, i3); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i1, i2, i3, newValue);
        }

        [__NoSideEffect]
        [nonmutating]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1, $2, $3)", T;
            }
        }
    }
    __subscript(uint3 i) -> T
    {
        [ForceInline] [__NoSideEffect] get { return load(i.x, i.y, i.z); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i.x, i.y, i.z, newValue);
        }

        [__NoSideEffect]
        [nonmutating]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1.x, $1.y, $1.z)", T;
            }
        }
    }
    __subscript(uint i1, uint i2, uint i3, uint i4) -> T
    {
        [ForceInline]
        [__NoSideEffect]
        get { return load(i1, i2, i3, i4); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i1, i2, i3, i4, newValue);
        }

        [__NoSideEffect]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1, $2, $3, $4)", T;
            }
        }
    }
    __subscript(uint4 i) -> T
    {
        [__NoSideEffect][ForceInline] get { return load(i.x, i.y, i.z, i.w); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i.x, i.y, i.z, i.w, newValue);
        }

        [__NoSideEffect]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1.x, $1.y, $1.z, $1.w)", T;
            }
        }
    }
    __subscript(uint i1, uint i2, uint i3, uint i4, uint i5) -> T
    {
        [ForceInline] [__NoSideEffect] get { return load(i1, i2, i3, i4, i5); }

        [ForceInline]
        [nonmutating]
        set
        {
            store(i1, i2, i3, i4, i5, newValue);
        }

        [__NoSideEffect]
        [nonmutating]
        ref
        {
            __target_switch
            {
            case cuda:
                __intrinsic_asm "$0.load<$[0]>($1, $2, $3, $4, $5)", T;
            }
        }
    }
}

${{{{
for (auto atomicIntegerTypeName : kCudaAtomicIntegerTypes)
{
}}}}
extension TensorView<$(atomicIntegerTypeName)>
{
    typealias __Element = $(atomicIntegerTypeName);

    [require(cuda)]
    void InterlockedMin(uint index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicMin($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void InterlockedMin(vector<uint, N> index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicMin($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedMax<T>(uint index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicMax($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void InterlockedMax(vector<uint, N> index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicMax($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedAnd<T>(uint index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicAnd($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void InterlockedAnd(vector<uint, N> index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicAnd($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedOr<T>(uint index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicOr($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void InterlockedOr(vector<uint, N> index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicOr($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedXor<T>(uint index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicXor($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void InterlockedXor(vector<uint, N> index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicXor($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedExchange(uint index, __Element va, out __Element oldVall)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicExch($0.data_ptr_at<$T2>($1), $2)";
        }
    }
    
    __generic<let N:int>
    [require(cuda)]
    void InterlockedExchange(vector<uint, N> index, __Element val, out __Element oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicExch($0.data_ptr_at<$T2>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedCompareExchange(uint index, __Element compare, __Element val)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "atomicCAS($0.data_ptr_at<$T2>($1), $2, $3)";
        }
    }
    
    __generic<let N:int>
    [require(cuda)]
    void InterlockedCompareExchange(vector<uint, N> index, __Element compare, __Element val)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "atomicCAS($0.data_ptr_at<$T2>($1), $2, $3)";
        }
    }
}

${{{{
} // end for atomicIntegerTypeName
}}}}

extension TensorView<float>
{
    [require(cuda)]
    float InterlockedExchange(uint index, float val, out float oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicExch($0.data_ptr_at<float>($1), $2)";
        }
    }
    
    __generic<let N:int>
    [require(cuda)]
    float InterlockedExchange(vector<uint, N> index, float val, out float oldVal)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "*($3) = atomicExch($0.data_ptr_at<float>($1), $2)";
        }
    }

    [require(cuda)]
    void InterlockedCompareExchange(uint index, float compare, float val)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "atomicCAS($0.data_ptr_at<uint32_t>($1), slang_bit_cast<uint32_t>($2), slang_bit_cast<uint32_t>($3))";
        }
    }

    __generic<let N : int>
    [require(cuda)]
    void InterlockedCompareExchange(vector<uint, N> index, float compare, float val)
    {
        __target_switch
        {
        case cuda: __intrinsic_asm "atomicCAS($0.data_ptr_at<uint32_t>($1), slang_bit_cast<uint32_t>($2), slang_bit_cast<uint32_t>($3))";
        }
    }
}

interface IDiffTensorWrapper
{
    // Derivatives for universal load/store operations.

    __generic<T : __BuiltinFloatingPointType>
    T load_forward(uint i);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T load_forward(vector<uint, N> i);

    __generic<T : __BuiltinFloatingPointType>
    void load_backward(uint i, T dOut);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void load_backward(vector<uint, N> i, T dOut);

    __generic<T : __BuiltinFloatingPointType>
    void store_forward(uint i, T dx);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void store_forward(vector<uint, N> i, T dx);

    __generic<T : __BuiltinFloatingPointType>
    T store_backward(uint i);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T store_backward(vector<uint, N> i);

    // Derivatives for loadOnce/storeOnce operations. These operations
    // are designed to only run once per-address and don't need atomic
    // gradient handling.
    //

    __generic<T : __BuiltinFloatingPointType>
    T loadOnce_forward(uint i);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T loadOnce_forward(vector<uint, N> i);

    __generic<T : __BuiltinFloatingPointType>
    void loadOnce_backward(uint i, T dOut);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void loadOnce_backward(vector<uint, N> i, T dOut);

    __generic<T : __BuiltinFloatingPointType>
    void storeOnce_forward(uint i, T dx);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void storeOnce_forward(vector<uint, N> i, T dx);

    __generic<T : __BuiltinFloatingPointType>
    T storeOnce_backward(uint i);

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T storeOnce_backward(vector<uint, N> i);
};

struct AtomicAdd : IDiffTensorWrapper
{
    TensorView<float> diff;

    // Derivatives for universal load/store operations.

    __generic<T : __BuiltinFloatingPointType>
    T load_forward(uint i)
    {
        return __realCast<T, float>(diff.load(i));
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T load_forward(vector<uint, N> i)
    {
        return __realCast<T, float>(diff.load(i));
    }

    __generic<T : __BuiltinFloatingPointType>
    void load_backward(uint i, T dOut)
    {
        float oldVal;
        diff.InterlockedAdd(i, __realCast<float, T>(dOut), oldVal);
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void load_backward(vector<uint, N> i, T dOut)
    {
        float oldVal;
        diff.InterlockedAdd(i, __realCast<float, T>(dOut), oldVal);
    }

    __generic<T : __BuiltinFloatingPointType>
    void store_forward(uint i, T dx)
    {
        diff.store(i, __realCast<float, T>(dx));
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void store_forward(vector<uint, N> i, T dx)
    {
        diff.store(i, __realCast<float, T>(dx));
    }

    __generic<T : __BuiltinFloatingPointType>
    T store_backward(uint i)
    {
        float oldVal;
        diff.InterlockedExchange(i, (float)0, oldVal);
        return __realCast<T, float>(oldVal);
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T store_backward(vector<uint, N> i)
    {
        float oldVal;
        diff.InterlockedExchange(i, (float)0, oldVal);
        return __realCast<T, float>(oldVal);
    }

    // Derivatives for loadOnce/storeOnce operations. These operations
    // are designed to only run once per-address and don't need atomic
    // gradient handling.
    //

    __generic<T : __BuiltinFloatingPointType>
    T loadOnce_forward(uint i)
    {
        return __realCast<T, float>(diff.load(i));
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T loadOnce_forward(vector<uint, N> i)
    {
        return __realCast<T, float>(diff.load(i));
    }

    __generic<T : __BuiltinFloatingPointType>
    void loadOnce_backward(uint i, T dOut)
    {
        diff.store(i, __realCast<float, T>(dOut));
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void loadOnce_backward(vector<uint, N> i, T dOut)
    {
        diff.store(i, __realCast<float, T>(dOut));
    }

    __generic<T : __BuiltinFloatingPointType>
    void storeOnce_forward(uint i, T dx)
    {
        diff.store(i, __realCast<float, T>(dx));
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    void storeOnce_forward(vector<uint, N> i, T dx)
    {
        diff.store(i, __realCast<float, T>(dx));
    }

    __generic<T : __BuiltinFloatingPointType>
    T storeOnce_backward(uint i)
    {
        return __realCast<T, float>(diff.load(i));
    }

    __generic<T : __BuiltinFloatingPointType, let N : int>
    T storeOnce_backward(vector<uint, N> i)
    {
        return __realCast<T, float>(diff.load(i));
    }
};

__generic<T: __BuiltinFloatingPointType = float, A : IDiffTensorWrapper = AtomicAdd>
struct DiffTensorView
{
    TensorView<T> primal;
    A diff;

    uint size(uint i)
    {
        return primal.size(i);
    }

    uint dims()
    {
        return primal.dims();
    }

    uint stride(uint i)
    {
        return primal.stride(i);
    }

    // Constructors
    __init(TensorView<T> primal, A diff)
    {
        this.primal = primal;
        this.diff = diff;
    }

    __init(TensorView<T> primal)
    {
        this.primal = primal;
    }

    // Universal load/store operations.

    [BackwardDerivative(__load_backward)]
    [ForwardDerivative(__load_forward)]
    T load(uint i) { return primal.load(i); }

    [BackwardDerivative(__load_backward)]
    [ForwardDerivative(__load_forward)]
    __generic<let N : int>
    T load(vector<uint, N> i) { return primal.load(i); }

    DifferentialPair<T> __load_forward(uint x)
    {
        return diffPair(primal.load(x), reinterpret<T.Differential, T>(diff.load_forward<T>(x)));
    }

    [ForceInline]
    __generic<let N : int>
    DifferentialPair<T> __load_forward(vector<uint, N> x)
    {
        return diffPair(primal.load(x), reinterpret<T.Differential, T>(diff.load_forward<T, N>(x)));
    }

    [ForceInline]
    void __load_backward(uint x, T.Differential dOut)
    {
        diff.load_backward<T>(x, reinterpret<T, T.Differential>(dOut));
    }

    [ForceInline]
    __generic<let N : int>
    void __load_backward(vector<uint, N> x, T.Differential dOut)
    {
        diff.load_backward<T, N>(x, reinterpret<T, T.Differential>(dOut));
    }

    [BackwardDerivative(__store_backward)]
    [ForwardDerivative(__store_forward)]
    void store(uint x, T val) { primal.store(x, val); }

    [BackwardDerivative(__store_backward)]
    [ForwardDerivative(__store_forward)]
    __generic<let N : int>
    void store(vector<uint, N> x, T val) { primal.store(x, val); }

    void __store_forward(uint x, DifferentialPair<T> dpval)
    {
        primal.store(x, dpval.p);
        diff.store_forward<T>(x, reinterpret<T, T.Differential>(dpval.d));
    }

    __generic<let N : int>
    void __store_forward(vector<uint, N> x, DifferentialPair<T> dpval)
    {
        primal.store(x, dpval.p);
        diff.store_forward<T, N>(x, reinterpret<T, T.Differential>(dpval.d));
    }

    [ForceInline]
    void __store_backward(uint x, inout DifferentialPair<T> dpval)
    {
        dpval = diffPair(dpval.p, reinterpret<T.Differential, T>(diff.store_backward<T>(x)));
    }

    [ForceInline]
    __generic<let N : int>
    void __store_backward(vector<uint, N> x, inout DifferentialPair<T> dpval)
    {
        dpval = diffPair(dpval.p, reinterpret<T.Differential, T>(diff.store_backward<T, N>(x)));
    }

    __subscript(uint index)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(index); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(index, newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    __subscript(uint2 index)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(index); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(index, newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    __subscript(uint x, uint y)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(uint2(x, y)); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(uint2(x, y), newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    __subscript(uint3 index)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(index); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(index, newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    __subscript(uint x, uint y, uint z)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(uint3(x, y, z)); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(uint3(x, y, z), newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    __subscript(uint4 index)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(index); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(index, newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    __subscript(uint x, uint y, uint z, uint w)->T
    {
        [__unsafeForceInlineEarly] [Differentiable] [__NoSideEffect] get { return load(uint4(x, y, z, w)); }
        [__unsafeForceInlineEarly] [Differentiable] [nonmutating] set { store(uint4(x, y, z, w), newValue); }

        [__NoSideEffect]
        [nonmutating]
        ref;
    }

    // loadOnce/storeOnce operations. These operations are designed to only run once per-address and 
    // don't need atomic gradient handling.
    //

    [BackwardDerivative(__loadOnce_backward)]
    [ForwardDerivative(__loadOnce_forward)]
    T loadOnce(uint i) { return primal.load(i); }

    [BackwardDerivative(__loadOnce_backward)]
    [ForwardDerivative(__loadOnce_forward)]
    __generic<let N : int>
    T loadOnce(vector<uint, N> i) { return primal.load(i); }

    DifferentialPair<T> __loadOnce_forward(uint x)
    {
        return diffPair(primal.load(x), reinterpret<T.Differential, T>(diff.loadOnce_forward<T>(x)));
    }

    __generic<let N : int>
    DifferentialPair<T> __loadOnce_forward(vector<uint, N> x)
    {
        return diffPair(primal.load(x), reinterpret<T.Differential, T>(diff.loadOnce_forward<T, N>(x)));
    }

    [ForceInline]
    void __loadOnce_backward(uint x, T.Differential dOut)
    {
        diff.loadOnce_backward<T>(x, reinterpret<T, T.Differential>(dOut));
    }

    [ForceInline]
    __generic<let N : int>
    void __loadOnce_backward(vector<uint, N> x, T.Differential dOut)
    {
        diff.loadOnce_backward<T, N>(x, reinterpret<T, T.Differential>(dOut));
    }

    [BackwardDerivative(__storeOnce_backward)]
    [ForwardDerivative(__storeOnce_forward)]
    void storeOnce(uint x, T val) { primal.store(x, val); }

    [BackwardDerivative(__storeOnce_backward)]
    [ForwardDerivative(__storeOnce_forward)]
    __generic<let N : int>
    void storeOnce(vector<uint, N> x, T val) { primal.store(x, val); }

    void __storeOnce_forward(uint x, DifferentialPair<T> dpval)
    {
        primal.store(x, dpval.p);
        diff.storeOnce_forward<T>(x, reinterpret<T, T.Differential>(dpval.d));
    }

    __generic<let N : int>
    void __storeOnce_forward(vector<uint, N> x, DifferentialPair<T> dpval)
    {
        primal.store(x, dpval.p);
        diff.storeOnce_forward<T, N>(x, reinterpret<T, T.Differential>(dpval.d));
    }

    [ForceInline]
    void __storeOnce_backward(uint x, inout DifferentialPair<T> dpval)
    {
        dpval = diffPair(dpval.p, reinterpret<T.Differential, T>(diff.storeOnce_backward<T>(x)));
    }

    [ForceInline]
    __generic<let N : int>
    void __storeOnce_backward(vector<uint, N> x, inout DifferentialPair<T> dpval)
    {
        dpval = diffPair(dpval.p, reinterpret<T.Differential, T>(diff.storeOnce_backward<T, N>(x)));
    }
};

/// Represents the handle of a Torch tensor object.
__generic<T>
__intrinsic_type($(kIROp_TorchTensorType))
struct TorchTensor
{
    __intrinsic_op($(kIROp_TorchTensorGetView))
    [CudaHost]
    TensorView<T> getView();

    [__readNone]
    [CudaHost]
    [require(cpp_cuda)]
    uint dims()
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "$0.dims()";
        case cuda: __intrinsic_asm "$0.dims()";
        }
    }

    [__readNone]
    [CudaHost]
    [require(cpp_cuda)]
    uint size(uint i)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "$0.size($1)";
        case cuda: __intrinsic_asm "$0.size($1)";
        }
    }

    [__readNone]
    [CudaHost]
    [require(cpp_cuda)]
    uint stride(uint i)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "$0.stride($1)";
        case cuda: __intrinsic_asm "$0.stride($1)";
        }
    }

    [__readNone]
    [CudaHost]
    [require(cpp_cuda)]
    Ptr<T> data_ptr()
    {
        __target_switch
        {
        case cpp:
        case cuda:
            __intrinsic_asm "$0.data_ptr<$[0]>()", T;
        }
    }

    __intrinsic_op($(kIROp_AllocateTorchTensor))
    [CudaHost]
    static TorchTensor<T> alloc(uint x);

    __intrinsic_op($(kIROp_AllocateTorchTensor))
    [CudaHost]
    static TorchTensor<T> alloc(uint x, uint y);

    __intrinsic_op($(kIROp_AllocateTorchTensor))
    [CudaHost]
    static TorchTensor<T> alloc(uint x, uint y, uint z);

    __intrinsic_op($(kIROp_AllocateTorchTensor))
    [CudaHost]
    static TorchTensor<T> alloc(uint x, uint y, uint z, uint w);

    __intrinsic_op($(kIROp_AllocateTorchTensor))
    [CudaHost]
    static TorchTensor<T> alloc(uint i0, uint i1, uint i2, uint i3, uint i4);

    __intrinsic_op($(kIROp_AllocateTorchTensor))
    [CudaHost]
    static TorchTensor<T> emptyLike(TorchTensor<T> other);

    [CudaHost]
    [require(cpp)]
    void fillZero()
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "$0.zero_()";
        }
    }

    [CudaHost]
    [require(cpp)]
    void fillValue(T val)
    {
        __target_switch
        {
        case cpp: __intrinsic_asm "$0.fill_($1)";
        }
    }

    [CudaHost]
    static TorchTensor<T> zerosLike(TorchTensor<T> other)
    {
        var result = emptyLike(other);
        result.fillZero();
        return result;
    }

}

[require(cpp)]
void syncTorchCudaStream()
{
    __target_switch
    {
    case cpp: __intrinsic_asm "AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()))";
    }
}

/// Constructs a `DifferentialPair` value from a primal value and a differential value.
__generic<T: IDifferentiable>
__intrinsic_op($(kIROp_MakeDifferentialPairUserCode))
DifferentialPair<T> diffPair(T primal, T.Differential diff);

/// Constructs a `DifferentialPair` value from a primal value and a zero differential value.
__generic<T: IDifferentiable>
[__unsafeForceInlineEarly]
DifferentialPair<T> diffPair(T primal)
{
    return diffPair(primal, T.dzero());
}

[__unsafeForceInlineEarly]
void updatePrimal<T : IDifferentiable>(inout DifferentialPair<T> p, T newPrimal)
{
    p = DifferentialPair<T>(newPrimal, p.d);
}

[__unsafeForceInlineEarly]
void updateDiff<T : IDifferentiable>(inout DifferentialPair<T> p, T.Differential newDiff)
{
    p = DifferentialPair<T>(p.p, newDiff);
}

[__unsafeForceInlineEarly]
void updatePair<T : IDifferentiable>(inout DifferentialPair<T> p, T newPrimal, T.Differential newDiff)
{
    p = DifferentialPair<T>(newPrimal, newDiff);
}

__generic<T, let N:int>
__intrinsic_op($(kIROp_MakeArrayFromElement))
Array<T,N> makeArrayFromElement(T element);


__generic<T:IDifferentiable, let N:int>
extension Array<T, N> : IDifferentiable
{
    typedef Array<T.Differential, N> Differential;

    [__unsafeForceInlineEarly]
    static Differential dzero()
    {
        return makeArrayFromElement<T.Differential, N>(T.dzero());
    }

    [__unsafeForceInlineEarly]
    static Differential dadd(Differential a, Differential b)
    {
        Array<T.Differential, N> result;
        for (int i = 0; i < N; i++)
            result[i] = T.dadd(a[i], b[i]);
        return result;
    }

    __generic<U : __BuiltinRealType>
    [__unsafeForceInlineEarly]
    static Differential dmul(U a, Differential b)
    {
        Array<T.Differential, N> result;
        for (int i = 0; i < N; i++)
            result[i] = T.dmul<U>(a, b[i]);
        return result;
    }
}

__generic<T : IDifferentiablePtrType, let N : int>
extension Array<T, N> : IDifferentiablePtrType
{
    typedef Array<T.Differential, N> Differential;
}

__generic<each T : IDifferentiable>
extension Tuple<T> : IDifferentiable
{
    typealias Differential = Tuple<expand(each T).Differential>;

    [__unsafeForceInlineEarly]
    static Differential dzero()
    {
        return makeTuple(expand (each T).dzero());
    }

    [__unsafeForceInlineEarly]
    static Differential dadd(Differential a, Differential b)
    {
        return makeTuple(expand(each T).dadd(each a, each b));
    }

    __generic<U : __BuiltinRealType>
    [__unsafeForceInlineEarly]
    static Differential dmul(U a, Differential b)
    {
        return makeTuple(expand(each T).dmul(a, each b));
    }
}

// Matrix transpose
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[ForceInline]
[ForwardDerivativeOf(transpose)]
[PreferRecompute]
[BackwardDifferentiable]
DifferentialPair<matrix<T, M, N>> __d_transpose(DifferentialPair<matrix<T, N, M>> m)
{
    return DifferentialPair<matrix<T, M, N>>(transpose(m.p), transpose(m.d));
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[ForceInline]
[BackwardDerivativeOf(transpose)]
[PreferRecompute]
[BackwardDifferentiable]
void __d_transpose(inout DifferentialPair<matrix<T, N, M>> m, matrix<T, M, N>.Differential dOut)
{
    m = diffPair(m.p, transpose(dOut));
}

// vector-matrix
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[ForceInline]
[ForwardDerivativeOf(mul)]
[PreferRecompute]
[BackwardDifferentiable]
DifferentialPair<vector<T, M>> mul(DifferentialPair<vector<T, N>> left, DifferentialPair<matrix<T, N, M>> right)
{
    let primal = mul(left.p, right.p);
    let diff = mul(left.d, right.p) + mul(left.p, right.d);
    return DifferentialPair<vector<T,M>>(primal, diff);
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[BackwardDerivativeOf(mul)]
[PreferRecompute]
[BackwardDifferentiable]
void __d_mul(inout DifferentialPair<vector<T, N>> left, inout DifferentialPair<matrix<T, N, M>> right, vector<T, M>.Differential dOut)
{
    vector<T, N>.Differential left_d_result;
    matrix<T, N, M>.Differential right_d_result;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        T sum = T(0);
        [ForceUnroll]
        for (int j = 0; j < M; ++j)
        {
            sum += right.p[i][j] * dOut[j];
            right_d_result[i][j] = left.p[i] * dOut[j];
        }
        left_d_result[i] = sum;
    }
    left = diffPair(left.p, left_d_result);
    right = diffPair(right.p, right_d_result);
}

// matrix-vector
__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[ForceInline]
[ForwardDerivativeOf(mul)]
[PreferRecompute]
[BackwardDifferentiable]
DifferentialPair<vector<T,N>> mul(DifferentialPair<matrix<T,N,M>> left, DifferentialPair<vector<T,M>> right)
{
    let primal = mul(left.p, right.p);
    let diff = mul(left.d, right.p) + mul(left.p, right.d);
    return DifferentialPair<vector<T,N>>(primal, diff);
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int>
[BackwardDerivativeOf(mul)]
[PreferRecompute]
[BackwardDifferentiable]
void __d_mul(inout DifferentialPair<matrix<T, N, M>> left, inout DifferentialPair<vector<T, M>> right, vector<T, N>.Differential dOut)
{
    matrix<T, N, M>.Differential left_d_result;
    vector<T, M>.Differential right_d_result;
    [ForceUnroll]
    for (int j = 0; j < M; ++j)
    {
        T sum = T(0);
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
        {
            sum += left.p[i][j] * dOut[i];
            left_d_result[i][j] = right.p[j] * dOut[i];
        }
        right_d_result[j] = sum;
    }
    left = diffPair(left.p, left_d_result);
    right = diffPair(right.p, right_d_result);
}

// matrix-matrix
__generic<T : __BuiltinFloatingPointType, let R : int, let N : int, let C : int>
[ForceInline]
[ForwardDerivativeOf(mul)]
[PreferRecompute]
[BackwardDifferentiable]
DifferentialPair<matrix<T,R,C>> mul(DifferentialPair<matrix<T,R,N>> left, DifferentialPair<matrix<T,N,C>> right)
{
    let primal = mul(left.p, right.p);
    let diff = mul(left.d, right.p) + mul(left.p, right.d);
    return DifferentialPair<matrix<T,R,C>>(primal, diff);
}

__generic<T : __BuiltinFloatingPointType, let R : int, let N : int, let C : int>
[BackwardDerivativeOf(mul)]
[PreferRecompute]
[BackwardDifferentiable]
void mul(inout DifferentialPair<matrix<T, R, N>> left, inout DifferentialPair<matrix<T, N, C>> right, matrix<T, R, C>.Differential dOut)
{
    matrix<T, R, N>.Differential left_d_result;
    [ForceUnroll]
    for (int r = 0; r < R; ++r)
        [ForceUnroll]
        for (int n = 0; n < N; ++n)
            left_d_result[r][n] = T(0.0);

    matrix<T, N, C>.Differential right_d_result;
    [ForceUnroll]
    for (int n = 0; n < N; ++n)
        [ForceUnroll]
        for (int c = 0; c < C; ++c)
            right_d_result[n][c] = T(0.0);

    [ForceUnroll]
    for (int r = 0; r < R; ++r)
    {
        [ForceUnroll]
        for (int c = 0; c < C; ++c)
        {
            [ForceUnroll]
            for (int n = 0; n < N; ++n)
            {
                left_d_result[r][n] += right.p[n][c] * dOut[r][c];
                right_d_result[n][c] += left.p[r][n] * dOut[r][c];
            }
        }
    }
    left = diffPair(left.p, left_d_result);
    right = diffPair(right.p, right_d_result);
}

// Vector dot product
__generic<T : __BuiltinFloatingPointType, let N : int>
[ForwardDerivativeOf(dot)]
[PreferRecompute]
[BackwardDifferentiable]
DifferentialPair<T> __d_dot(DifferentialPair<vector<T, N>> dpx, DifferentialPair<vector<T, N>> dpy)
{
    T result = T(0);
    T.Differential d_result = T.dzero();
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        result = result + dpx.p[i] * dpy.p[i];
        d_result = T.dadd(d_result, __slang_noop_cast<T.Differential>(dpx.p[i] * dpy.d[i]));
        d_result = T.dadd(d_result, __slang_noop_cast<T.Differential>(dpy.p[i] * dpx.d[i]));
    }
    return DifferentialPair<T>(result, d_result);
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDerivativeOf(dot)]
[PreferRecompute]
[BackwardDifferentiable]
void __d_dot(inout DifferentialPair<vector<T, N>> dpx, inout DifferentialPair<vector<T, N>> dpy, T.Differential dOut)
{
    vector<T, N>.Differential x_d_result, y_d_result;
    [ForceUnroll]
    for (int i = 0; i < N; ++i)
    {
        x_d_result[i] = dpy.p[i] * __slang_noop_cast<T>(dOut);
        y_d_result[i] = dpx.p[i] * __slang_noop_cast<T>(dOut);
    }
    dpx = diffPair(dpx.p, x_d_result);
    dpy = diffPair(dpy.p, y_d_result);
}

// Cross product
__generic<T : __BuiltinFloatingPointType>
[ForwardDerivativeOf(cross)]
[PreferRecompute]
[BackwardDifferentiable]
DifferentialPair<vector<T, 3>> __d_cross(DifferentialPair<vector<T, 3>> a, DifferentialPair<vector<T, 3>> b)
{
    /*
    cx = ay * bz − az * by
    cy = az * bx − ax * bz
    cz = ax * by − ay * bx
    */
    T aybz = a.p.y * b.p.z;
    T azby = a.p.z * b.p.y;
    T px = aybz - azby;
    T dx = a.d.y * b.p.z + a.p.y * b.d.z - a.d.z * b.p.y - a.p.z * b.d.y;

    T azbx = a.p.z * b.p.x;
    T axbz = a.p.x * b.p.z;
    T py = azbx - axbz;
    T dy = a.d.z * b.p.x + a.p.z * b.d.x - a.d.x * b.p.z - a.p.x * b.d.z;

    T axby = a.p.x * b.p.y;
    T aybx = a.p.y * b.p.x;
    T pz = axby - aybx;
    T dz = a.d.x * b.p.y + a.p.x * b.d.y - a.d.y * b.p.x - a.p.y * b.d.x;
    
    return DifferentialPair<vector<T, 3>>(vector<T, 3>(px, py, pz), vector<T, 3>.Differential(dx, dy, dz));
}

__generic<T : __BuiltinFloatingPointType>
[BackwardDerivativeOf(cross)]
[PreferRecompute]
[BackwardDifferentiable]
void __d_cross(inout DifferentialPair<vector<T, 3>> a, inout DifferentialPair<vector<T, 3>> b, vector<T, 3>.Differential dOut)
{
    /*
    cx = ay * bz − az * by
    cy = az * bx − ax * bz
    cz = ax * by − ay * bx
    */
    T dax = (-b.p.z * dOut.y) + (b.p.y * dOut.z);
    T day = (b.p.z * dOut.x) + (-b.p.x * dOut.z);
    T daz = (-b.p.y * dOut.x) + (b.p.x * dOut.y);

    T dbx = (a.p.z * dOut.y) + (-a.p.y * dOut.z);
    T dby = (-a.p.z * dOut.x) + (a.p.x * dOut.z);
    T dbz = (a.p.y * dOut.x) + (-a.p.x * dOut.y);

    a = diffPair(a.p, vector<T, 3>.Differential(dax, day, daz));
    b = diffPair(b.p, vector<T, 3>.Differential(dbx, dby, dbz));
}

#define VECTOR_MATRIX_BINARY_DIFF_IMPL(NAME)                                                 \
    __generic<T : __BuiltinFloatingPointType, let N : int>                                   \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<vector<T, N>> __d_##NAME##_vector(                                      \
        DifferentialPair<vector<T, N>> dpx, DifferentialPair<vector<T, N>> dpy)              \
    {                                                                                        \
        vector<T, N> result;                                                                 \
        vector<T, N>.Differential d_result;                                                  \
        [ForceUnroll] for (int i = 0; i < N; ++i)                                            \
        {                                                                                    \
            DifferentialPair<T> dp_elem = __d_##NAME(                                        \
                DifferentialPair<T>(dpx.p[i], __slang_noop_cast<T.Differential>(dpx.d[i])),  \
                DifferentialPair<T>(dpy.p[i], __slang_noop_cast<T.Differential>(dpy.d[i]))); \
            result[i] = dp_elem.p;                                                           \
            d_result[i] = __slang_noop_cast<T>(dp_elem.d);                                   \
        }                                                                                    \
        return DifferentialPair<vector<T, N>>(result, d_result);                             \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let M : int, let N : int>                      \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<matrix<T, M, N>> __d_##NAME##_matrix(                                   \
        DifferentialPair<matrix<T, M, N>> dpx, DifferentialPair<matrix<T, M, N>> dpy)        \
    {                                                                                        \
        matrix<T, M, N> result;                                                              \
        matrix<T, M, N>.Differential d_result;                                               \
        [ForceUnroll] for (int i = 0; i < M; ++i)                                            \
        [ForceUnroll] for (int j = 0; j < N; ++j)                                            \
        {                                                                                    \
            DifferentialPair<T> dp_elem = __d_##NAME(                                        \
                DifferentialPair<T>(dpx.p[i][j], __slang_noop_cast<T.Differential>(dpx.d[i][j])),  \
                DifferentialPair<T>(dpy.p[i][j], __slang_noop_cast<T.Differential>(dpy.d[i][j]))); \
            result[i][j] = dp_elem.p;                                                        \
            d_result[i][j] = __slang_noop_cast<T>(dp_elem.d);                                \
        }                                                                                    \
        return DifferentialPair<matrix<T, M, N>>(result, d_result);                          \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let N : int>                                   \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME##_vector(                                                                \
            inout DifferentialPair<vector<T, N>> dpx,                                        \
            inout DifferentialPair<vector<T, N>> dpy,                                        \
            vector<T, N>.Differential dOut)                                                  \
    {                                                                                        \
        vector<T, N>.Differential left_d_result, right_d_result;                             \
        [ForceUnroll] for (int i = 0; i < N; ++i)                                            \
        {                                                                                    \
            DifferentialPair<T> left_dp = diffPair(dpx.p[i], T.dzero());                     \
            DifferentialPair<T> right_dp = diffPair(dpy.p[i], T.dzero());                    \
            __d_##NAME(left_dp, right_dp, __slang_noop_cast<T.Differential>(dOut[i]));       \
            left_d_result[i] = __slang_noop_cast<T>(left_dp.d);                              \
            right_d_result[i] = __slang_noop_cast<T>(right_dp.d);                            \
        }                                                                                    \
        dpx = diffPair(dpx.p, left_d_result);                                                \
        dpy = diffPair(dpy.p, right_d_result);                                               \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let M : int, let N : int>                      \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME##_matrix(                                                                \
            inout DifferentialPair<matrix<T, M, N>> dpx,                                     \
            inout DifferentialPair<matrix<T, M, N>> dpy,                                     \
            matrix<T, M, N>.Differential dOut)                                               \
    {                                                                                        \
        matrix<T, M, N>.Differential left_d_result, right_d_result;                          \
        [ForceUnroll] for (int i = 0; i < M; ++i)                                            \
        [ForceUnroll] for (int j = 0; j < N; ++j)                                            \
        {                                                                                    \
            DifferentialPair<T> left_dp = diffPair(dpx.p[i][j], T.dzero());                  \
            DifferentialPair<T> right_dp = diffPair(dpy.p[i][j], T.dzero());                 \
            __d_##NAME(left_dp, right_dp, __slang_noop_cast<T.Differential>(dOut[i][j]));    \
            left_d_result[i][j] = __slang_noop_cast<T>(left_dp.d);                           \
            right_d_result[i][j] = __slang_noop_cast<T>(right_dp.d);                         \
        }                                                                                    \
        dpx = diffPair(dpx.p, left_d_result);                                                \
        dpy = diffPair(dpy.p, right_d_result);                                               \
    }

#define VECTOR_MATRIX_TERNARY_DIFF_IMPL(NAME)                                                \
    __generic<T : __BuiltinFloatingPointType, let N : int>                                   \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<vector<T, N>> __d_##NAME##_vector(                                      \
        DifferentialPair<vector<T, N>> dpx,                                                  \
        DifferentialPair<vector<T, N>> dpy,                                                  \
        DifferentialPair<vector<T, N>> dpz)                                                  \
    {                                                                                        \
        vector<T, N> result;                                                                 \
        vector<T, N>.Differential d_result;                                                  \
        [ForceUnroll] for (int i = 0; i < N; ++i)                                            \
        {                                                                                    \
            DifferentialPair<T> dp_elem = __d_##NAME(                                        \
                DifferentialPair<T>(dpx.p[i], __slang_noop_cast<T.Differential>(dpx.d[i])),  \
                DifferentialPair<T>(dpy.p[i], __slang_noop_cast<T.Differential>(dpy.d[i])),  \
                DifferentialPair<T>(dpz.p[i], __slang_noop_cast<T.Differential>(dpz.d[i]))); \
            result[i] = dp_elem.p;                                                           \
            d_result[i] = __slang_noop_cast<T>(dp_elem.d);                                   \
        }                                                                                    \
        return DifferentialPair<vector<T, N>>(result, d_result);                             \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let M : int, let N : int>                      \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<matrix<T, M, N>> __d_##NAME##_matrix(                                   \
        DifferentialPair<matrix<T, M, N>> dpx,                                               \
        DifferentialPair<matrix<T, M, N>> dpy,                                               \
        DifferentialPair<matrix<T, M, N>> dpz)                                               \
    {                                                                                        \
        matrix<T, M, N> result;                                                              \
        matrix<T, M, N>.Differential d_result;                                               \
        [ForceUnroll] for (int i = 0; i < M; ++i)                                            \
        [ForceUnroll] for (int j = 0; j < N; ++j)                                            \
        {                                                                                    \
            DifferentialPair<T> dp_elem = __d_##NAME(                                        \
                DifferentialPair<T>(dpx.p[i][j], __slang_noop_cast<T.Differential>(dpx.d[i][j])),  \
                DifferentialPair<T>(dpy.p[i][j], __slang_noop_cast<T.Differential>(dpy.d[i][j])),  \
                DifferentialPair<T>(dpz.p[i][j], __slang_noop_cast<T.Differential>(dpz.d[i][j]))); \
            result[i][j] = dp_elem.p;                                                        \
            d_result[i][j] = __slang_noop_cast<T>(dp_elem.d);                                \
        }                                                                                    \
        return DifferentialPair<matrix<T, M, N>>(result, d_result);                          \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let N : int>                                   \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME##_vector(                                                                \
            inout DifferentialPair<vector<T, N>> dpx,                                        \
            inout DifferentialPair<vector<T, N>> dpy,                                        \
            inout DifferentialPair<vector<T, N>> dpz,                                        \
            vector<T, N>.Differential dOut)                                                  \
    {                                                                                        \
        vector<T, N>.Differential left_d_result, middle_d_result, right_d_result;            \
        [ForceUnroll] for (int i = 0; i < N; ++i)                                            \
        {                                                                                    \
            DifferentialPair<T> left_dp = diffPair(dpx.p[i], T.dzero());                     \
            DifferentialPair<T> middle_dp = diffPair(dpy.p[i], T.dzero());                   \
            DifferentialPair<T> right_dp = diffPair(dpz.p[i], T.dzero());                    \
            __d_##NAME(left_dp, middle_dp, right_dp,                                         \
                __slang_noop_cast<T.Differential>(dOut[i]));                                 \
            left_d_result[i] = __slang_noop_cast<T>(left_dp.d);                              \
            middle_d_result[i] = __slang_noop_cast<T>(middle_dp.d);                          \
            right_d_result[i] = __slang_noop_cast<T>(right_dp.d);                            \
        }                                                                                    \
        dpx = diffPair(dpx.p, left_d_result);                                                \
        dpy = diffPair(dpy.p, middle_d_result);                                              \
        dpz = diffPair(dpz.p, right_d_result);                                               \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let M : int, let N : int>                      \
    [BackwardDifferentiable][PreferRecompute]                                                \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME##_matrix(                                                                \
            inout DifferentialPair<matrix<T, M, N>> dpx,                                     \
            inout DifferentialPair<matrix<T, M, N>> dpy,                                     \
            inout DifferentialPair<matrix<T, M, N>> dpz,                                     \
            matrix<T, M, N>.Differential dOut)                                               \
    {                                                                                        \
        matrix<T, M, N>.Differential left_d_result, middle_d_result, right_d_result;         \
        [ForceUnroll] for (int i = 0; i < M; ++i)                                            \
        [ForceUnroll] for (int j = 0; j < N; ++j)                                            \
        {                                                                                    \
            DifferentialPair<T> left_dp = diffPair(dpx.p[i][j], T.dzero());                  \
            DifferentialPair<T> middle_dp = diffPair(dpy.p[i][j], T.dzero());                \
            DifferentialPair<T> right_dp = diffPair(dpz.p[i][j], T.dzero());                 \
            __d_##NAME(left_dp, middle_dp, right_dp,                                         \
                __slang_noop_cast<T.Differential>(dOut[i][j]));                              \
            left_d_result[i][j] = __slang_noop_cast<T>(left_dp.d);                           \
            middle_d_result[i][j] = __slang_noop_cast<T>(middle_dp.d);                       \
            right_d_result[i][j] = __slang_noop_cast<T>(right_dp.d);                         \
        }                                                                                    \
        dpx = diffPair(dpx.p, left_d_result);                                                \
        dpy = diffPair(dpy.p, middle_d_result);                                              \
        dpz = diffPair(dpz.p, right_d_result);                                               \
    }

#define UNARY_DERIVATIVE_IMPL(NAME, FWD_DIFF_FUNC, BWD_DIFF_FUNC)                            \
    __generic<T : __BuiltinFloatingPointType>                                                \
    [BackwardDifferentiable] [PreferRecompute]                                               \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<T> __d_##NAME(DifferentialPair<T> dpx)                                  \
    {                                                                                        \
        typealias ReturnType = T;                                                            \
        return DifferentialPair<T>(NAME(dpx.p), FWD_DIFF_FUNC);                              \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let N : int>                                   \
    [BackwardDifferentiable] [PreferRecompute]                                               \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<vector<T, N>> __d_##NAME##_vector(DifferentialPair<vector<T, N>> dpx)   \
    {                                                                                        \
        typealias ReturnType = vector<T, N>;                                                 \
        return DifferentialPair<ReturnType>(NAME(dpx.p), FWD_DIFF_FUNC);                     \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let M : int, let N : int>                      \
    [BackwardDifferentiable] [PreferRecompute]                                               \
    [ForwardDerivativeOf(NAME)]                                                              \
    DifferentialPair<matrix<T, M, N>> __d_##NAME##_m(DifferentialPair<matrix<T, M, N>> dpm)  \
    {                                                                                        \
        typealias ReturnType = vector<T, N>;                                                 \
        matrix<T, M, N>.Differential diff;                                                   \
        [ForceUnroll] for (int i = 0; i < M; i++)                                            \
        {                                                                                    \
            var dpx = diffPair(dpm.p[i], dpm.d[i]);                                          \
            diff[i] = __slang_noop_cast<vector<T, N>>(FWD_DIFF_FUNC);                        \
        }                                                                                    \
        return diffPair(NAME(dpm.p), diff);                                                  \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType>                                                \
    [BackwardDifferentiable] [PreferRecompute]                                               \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME(inout DifferentialPair<T> dpx, T.Differential dOut)                      \
    {                                                                                        \
        typealias ReturnType = T;                                                            \
        dpx = diffPair(dpx.p, BWD_DIFF_FUNC);                                                \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let N : int>                                   \
    [BackwardDifferentiable] [PreferRecompute]                                               \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME##_vector(                                                                \
        inout DifferentialPair<vector<T, N>> dpx, vector<T, N>.Differential dOut)            \
    {                                                                                        \
        typealias ReturnType = vector<T, N>;                                                 \
        dpx = diffPair(dpx.p, BWD_DIFF_FUNC);                                                \
    }                                                                                        \
    __generic<T : __BuiltinFloatingPointType, let M : int, let N : int>                      \
    [BackwardDifferentiable] [PreferRecompute]                                               \
    [BackwardDerivativeOf(NAME)]                                                             \
    void __d_##NAME##_matrix(                                                                \
        inout DifferentialPair<matrix<T, M, N>> m, matrix<T, M, N>.Differential mdOut)       \
    {                                                                                        \
        typealias ReturnType = vector<T, N>;                                                 \
        matrix<T, M, N>.Differential diff;                                                   \
        [ForceUnroll] for (int i = 0; i < M; i++)                                            \
        {                                                                                    \
            var dpx = diffPair(m.p[i], m.d[i]);                                              \
            var dOut = __slang_noop_cast<vector<T, N>>(mdOut[i]);                            \
            diff[i] = BWD_DIFF_FUNC;                                                         \
        }                                                                                    \
        m = diffPair(m.p, diff);                                                             \
    }
#define SIMPLE_UNARY_DERIVATIVE_IMPL(NAME, DIFF_FUNC) UNARY_DERIVATIVE_IMPL(NAME, __mul_p_d(DIFF_FUNC, dpx.d), __mul_p_d(DIFF_FUNC, dOut))

/// Element-wise multiply for scalars and vectors for (T, T.Differential)
__generic<T : __BuiltinFloatingPointType>
[__unsafeForceInlineEarly]
[Differentiable]
T.Differential __mul_p_d(T a, T.Differential b)
{
    return __slang_noop_cast<T.Differential>(a * __slang_noop_cast<T>(b));
}

__generic<T : __BuiltinFloatingPointType>
[__unsafeForceInlineEarly]
[Differentiable]
T __mul_p_d(T a, T b)
{
    return (a * b);
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__unsafeForceInlineEarly]
[Differentiable]
vector<T, N> __mul_p_d(vector<T, N> a, vector<T, N> b)
{
    return a * b;
}


/// Detach and set derivatives to zero.
__generic<T : IDifferentiable>
__intrinsic_op($(kIROp_DetachDerivative))
T detach(T x);

#define SLANG_SQR(x) ((x)*(x))

#define SLANG_SIGN(x) select(((x)>T(0.0)), ReturnType(T(1.0)), select(((x)==T(0.0)), ReturnType(T(0.0)), ReturnType(T(-1.0))))

// Absolute value
UNARY_DERIVATIVE_IMPL(abs, (__mul_p_d(SLANG_SIGN(dpx.p), (dpx.d))), (__mul_p_d(SLANG_SIGN(dpx.p), (dOut))))
// Saturate
UNARY_DERIVATIVE_IMPL(saturate, select(dpx.p < T(0.0) || dpx.p > T(1.0), ReturnType.dzero(), dpx.d), select(dpx.p < T(0.0) || dpx.p > T(1.0), ReturnType.dzero(), dOut))
// frac
UNARY_DERIVATIVE_IMPL(frac, dpx.d, dOut)
// raidans, degrees
SIMPLE_UNARY_DERIVATIVE_IMPL(radians, ReturnType(T(0.01745329251994329576923690768489)))
SIMPLE_UNARY_DERIVATIVE_IMPL(degrees, ReturnType(T(57.295779513082320876798154814105)))
// Exponent
SIMPLE_UNARY_DERIVATIVE_IMPL(exp, exp(dpx.p))
SIMPLE_UNARY_DERIVATIVE_IMPL(exp2, exp2(dpx.p)* T(50.69314718055994530941723212145818))
// sin, sinh
SIMPLE_UNARY_DERIVATIVE_IMPL(sin, cos(dpx.p))
SIMPLE_UNARY_DERIVATIVE_IMPL(sinh, cosh(dpx.p))
// cos, cosh
SIMPLE_UNARY_DERIVATIVE_IMPL(cos, -sin(dpx.p))
SIMPLE_UNARY_DERIVATIVE_IMPL(cosh, sinh(dpx.p))
// tan, tanh
SIMPLE_UNARY_DERIVATIVE_IMPL(tan, T(1.0) / (cos(dpx.p) * cos(dpx.p)))
SIMPLE_UNARY_DERIVATIVE_IMPL(tanh, T(1.0) / (cosh(dpx.p) * cosh(dpx.p)))
// Logarithm
SIMPLE_UNARY_DERIVATIVE_IMPL(log, T(1.0) / dpx.p)
SIMPLE_UNARY_DERIVATIVE_IMPL(log10, T(1.0) / (dpx.p * T(52.3025850929940456840179914546844)))
SIMPLE_UNARY_DERIVATIVE_IMPL(log2, T(1.0) / (dpx.p * T(50.69314718055994530941723212145818)))
// Square root
SIMPLE_UNARY_DERIVATIVE_IMPL(sqrt, T(0.5) / sqrt(max(ReturnType(T(1e-7)), dpx.p)))
// Reciprocal
SIMPLE_UNARY_DERIVATIVE_IMPL(rcp, T(-1.0) / max(ReturnType(T(1e-7)), dpx.p * dpx.p))
// rsqrt
SIMPLE_UNARY_DERIVATIVE_IMPL(rsqrt, T(-0.5) / (dpx.p * sqrt(dpx.p)))
// Arc-sin
SIMPLE_UNARY_DERIVATIVE_IMPL(asin, T(1.0) / sqrt(T(1.0) - dpx.p * dpx.p))
// Arc-cos
SIMPLE_UNARY_DERIVATIVE_IMPL(acos, T(-1.0) / sqrt(T(1.0) - dpx.p * dpx.p))
// Arc-tan
SIMPLE_UNARY_DERIVATIVE_IMPL(atan, T(1.0) / (T(1.0) + dpx.p * dpx.p))

// Atan2
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(atan2)]
DifferentialPair<T> __d_atan2(DifferentialPair<T> dpy, DifferentialPair<T> dpx)
{
    T.Differential dx = __mul_p_d(-dpy.p / (dpx.p * dpx.p + dpy.p * dpy.p), dpx.d);
    T.Differential dy = __mul_p_d(dpx.p / (dpx.p * dpx.p + dpy.p * dpy.p), dpy.d);
    return DifferentialPair<T>(
        atan2(dpy.p, dpx.p),
        T.dadd(dx, dy));
}

__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(atan2)]
void __d_atan2(inout DifferentialPair<T> dpy, inout DifferentialPair<T> dpx, T.Differential dOut)
{
    dpx = diffPair(dpx.p, __mul_p_d(-dpy.p / (dpx.p * dpx.p + dpy.p * dpy.p), dOut));
    dpy = diffPair(dpy.p, __mul_p_d(dpx.p / (dpx.p * dpx.p + dpy.p * dpy.p), dOut));
}

VECTOR_MATRIX_BINARY_DIFF_IMPL(atan2)

// fmod
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(fmod)]
DifferentialPair<T> __d_fmod(DifferentialPair<T> x, DifferentialPair<T> y)
{
    return DifferentialPair<T>(fmod(x.p, y.p), x.d);
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(fmod)]
void __d_fmod(inout DifferentialPair<T> x, inout DifferentialPair<T> y, T.Differential dOut)
{
    x = diffPair(x.p, dOut);
    y = diffPair(y.p);
}
VECTOR_MATRIX_BINARY_DIFF_IMPL(fmod)

// Raise to a power
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(pow)]
DifferentialPair<T> __d_pow(DifferentialPair<T> dpx, DifferentialPair<T> dpy)
{
    // Special case
    if (dpx.p < T(1e-6))
    {
        return DifferentialPair<T>(T(0.0), T.dzero());
    }

    T val = pow(dpx.p, dpy.p);
    T.Differential d1 = __mul_p_d((val * log(dpx.p)), dpy.d);
    T.Differential d2 = __mul_p_d((val * dpy.p / dpx.p), dpx.d);
    return DifferentialPair<T>(
        val,
        T.dadd(d1, d2)
    );
}

__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(pow)]
void __d_pow(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpy, T.Differential dOut)
{
    // Special case
    if (dpx.p < T(1e-6))
    {
        dpx = diffPair(dpx.p, T.dzero());
        dpy = diffPair(dpy.p, T.dzero());
    }
    else
    {
        T val = pow(dpx.p, dpy.p);
        dpx = diffPair(
            dpx.p,
            (__mul_p_d((val * dpy.p / dpx.p), dOut)));
        dpy = diffPair(
            dpy.p,
            (__mul_p_d((val * log(dpx.p)), dOut)));
    }
}

VECTOR_MATRIX_BINARY_DIFF_IMPL(pow)

// Maximum
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(max)]
DifferentialPair<T> __d_max(DifferentialPair<T> dpx, DifferentialPair<T> dpy)
{
    return DifferentialPair<T>(
        max(dpx.p, dpy.p),
        dpx.p > dpy.p ? dpx.d : (dpx.p < dpy.p ? dpy.d : __mul_p_d(T(0.5), T.dadd(dpx.d, dpy.d)))
    );
}

__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(max)]
void __d_max(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpy, T.Differential dOut)
{
    dpx = diffPair(dpx.p, dpx.p > dpy.p ? dOut : (dpx.p < dpy.p ? T.dzero() : __mul_p_d(T(0.5), dOut)));
    dpy = diffPair(dpy.p, dpy.p > dpx.p ? dOut : (dpy.p < dpx.p ? T.dzero() : __mul_p_d(T(0.5), dOut)));
}

VECTOR_MATRIX_BINARY_DIFF_IMPL(max)

// Minimum
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(min)]
DifferentialPair<T> __d_min(DifferentialPair<T> dpx, DifferentialPair<T> dpy)
{
    return DifferentialPair<T>(
        min(dpx.p, dpy.p),
        dpx.p < dpy.p ? dpx.d : (dpx.p > dpy.p ? dpy.d : __mul_p_d(T(0.5), T.dadd(dpx.d, dpy.d)))
    );
}

__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(min)]
void __d_min(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpy, T.Differential dOut)
{
    dpx = diffPair(dpx.p, dpx.p < dpy.p ? dOut : (dpx.p > dpy.p ? T.dzero() : __mul_p_d(T(0.5), dOut)));
    dpy = diffPair(dpy.p, dpy.p < dpx.p ? dOut : (dpy.p > dpx.p ? T.dzero() : __mul_p_d(T(0.5), dOut)));
}

VECTOR_MATRIX_BINARY_DIFF_IMPL(min)

// Lerp
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(lerp)]
DifferentialPair<T> __d_lerp(DifferentialPair<T> dpx, DifferentialPair<T> dpy, DifferentialPair<T> dps)
{
    return DifferentialPair<T>(
        lerp(dpx.p, dpy.p, dps.p),
        T.dadd(T.dadd(__mul_p_d((T(1.0) - dps.p), dpx.d), __mul_p_d(dps.p, dpy.d)), __mul_p_d((dpy.p - dpx.p), dps.d))
    );
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(lerp)]
void __d_lerp(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpy, inout DifferentialPair<T> dps, T.Differential dOut)
{
    dpx = diffPair(dpx.p, __mul_p_d((T(1.0) - dps.p), dOut));
    dpy = diffPair(dpy.p, __mul_p_d(dps.p, dOut));
    dps = diffPair(dpy.p, __mul_p_d((dpy.p - dpx.p), dOut));
}
VECTOR_MATRIX_TERNARY_DIFF_IMPL(lerp)

//  Clamp
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[ForwardDerivativeOf(clamp)]
DifferentialPair<T> __d_clamp(DifferentialPair<T> dpx, DifferentialPair<T> dpMin, DifferentialPair<T> dpMax)
{
    return DifferentialPair<T>(
        clamp(dpx.p, dpMin.p, dpMax.p),
        (dpx.p >= dpMin.p && dpx.p <= dpMax.p) ? dpx.d : (dpx.p < dpMin.p ? dpMin.d : dpMax.d));
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
[BackwardDerivativeOf(clamp)]
void __d_clamp(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpMin, inout DifferentialPair<T> dpMax, T.Differential dOut)
{
    // Propagate the derivative to x if x is within [min, max] (including the boundaries).
    dpx = diffPair(dpx.p, (dpx.p >= dpMin.p && dpx.p <= dpMax.p) ? dOut : T.dzero());
    // If x is strictly below min or above max, the gradient is instead applied to the clamp bounds
    dpMin = diffPair(dpMin.p, dpx.p < dpMin.p ? dOut : T.dzero());
    dpMax = diffPair(dpMax.p, dpx.p > dpMax.p ? dOut : T.dzero());
}
VECTOR_MATRIX_TERNARY_DIFF_IMPL(clamp)

// fma
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[ForwardDerivativeOf(fma)]
[PreferRecompute]
DifferentialPair<T> __d_fma(DifferentialPair<T> dpx, DifferentialPair<T> dpy, DifferentialPair<T> dpz)
{
    return DifferentialPair<T>(
        fma(dpx.p, dpy.p, dpz.p),
        T.dadd(T.dadd(__mul_p_d(dpy.p, dpx.d), __mul_p_d(dpx.p, dpy.d)), dpz.d));
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[BackwardDerivativeOf(fma)]
[PreferRecompute]
void __d_fma(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpy, inout DifferentialPair<T> dpz, T.Differential dOut)
{
    dpx = diffPair(dpx.p, __mul_p_d(dpy.p, dOut));
    dpy = diffPair(dpy.p, __mul_p_d(dpx.p, dOut));
    dpz = diffPair(dpz.p, dOut);
}
VECTOR_MATRIX_TERNARY_DIFF_IMPL(fma)

// mad
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[ForwardDerivativeOf(mad)]
[PreferRecompute]
DifferentialPair<T> __d_mad(DifferentialPair<T> dpx, DifferentialPair<T> dpy, DifferentialPair<T> dpz)
{
    return DifferentialPair<T>(
        mad(dpx.p, dpy.p, dpz.p),
        T.dadd(T.dadd(__mul_p_d(dpy.p, dpx.d), __mul_p_d(dpx.p, dpy.d)), dpz.d));
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[BackwardDerivativeOf(mad)]
[PreferRecompute]
void __d_mad(inout DifferentialPair<T> dpx, inout DifferentialPair<T> dpy, inout DifferentialPair<T> dpz, T.Differential dOut)
{
    dpx = diffPair(dpx.p, __mul_p_d(dpy.p, dOut));
    dpy = diffPair(dpy.p, __mul_p_d(dpx.p, dOut));
    dpz = diffPair(dpz.p, dOut);
}
VECTOR_MATRIX_TERNARY_DIFF_IMPL(mad)

// Smoothstep
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PreferRecompute]
T __smoothstep_impl(T minVal, T maxVal, T x)
{
    let t = saturate((x - minVal) / (maxVal - minVal));
    return t * t * (T(3.0) - T(2.0) * t);
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[ForwardDerivativeOf(smoothstep)]
[PreferRecompute]
DifferentialPair<T> __d_smoothstep(DifferentialPair<T> minVal, DifferentialPair<T> maxVal, DifferentialPair<T> x)
{
    return __fwd_diff(__smoothstep_impl)(minVal, maxVal, x);
}
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[BackwardDerivativeOf(smoothstep)]
[PreferRecompute]
void __d_smoothstep(inout DifferentialPair<T> minVal, inout DifferentialPair<T> maxVal, inout DifferentialPair<T> x, T.Differential dOut)
{
    __bwd_diff(__smoothstep_impl)(minVal, maxVal, x, dOut);
}
VECTOR_MATRIX_TERNARY_DIFF_IMPL(smoothstep)

// Vector length
__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[PreferRecompute]
T __length_impl(vector<T, N> x)
{
    T len = T(0.0);
    [ForceUnroll] for (int i = 0; i < N; i++)
    {
        len += x[i] * x[i];
    }
    return sqrt(len);
}

__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[ForwardDerivativeOf(length)]
[ForceInline]
[PreferRecompute]
DifferentialPair<T> __d_length(DifferentialPair<vector<T, N>> x)
{
    return __fwd_diff(__length_impl)(x);
}

__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[BackwardDerivativeOf(length)]
[ForceInline]
[PreferRecompute]
void __d_length(inout DifferentialPair<vector<T, N>> x, T.Differential dOut)
{
    return __bwd_diff(__length_impl)(x, dOut);
}

// Vector distance
__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[PreferRecompute]
T __distance_impl(vector<T, N> x, vector<T, N> y)
{
    return length(y - x);
}
__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[ForwardDerivativeOf(distance)]
[ForceInline]
[PreferRecompute]
DifferentialPair<T> __d_distance(DifferentialPair<vector<T, N>> x, DifferentialPair<vector<T, N>> y)
{
    return __fwd_diff(__distance_impl)(x, y);
}

__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[BackwardDerivativeOf(distance)]
[ForceInline]
[PreferRecompute]
void __d_distance(inout DifferentialPair<vector<T, N>> x, inout DifferentialPair<vector<T, N>> y, T.Differential dOut)
{
    return __bwd_diff(__distance_impl)(x, y, dOut);
}

// Vector normalize
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[PreferRecompute]
vector<T, N> __normalize_impl(vector<T, N> x)
{
    let r = T(1.0) / length(x);
    return x * r;
}
__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[ForwardDerivativeOf(normalize)]
[ForceInline]
[PreferRecompute]
DifferentialPair<vector<T, N>> __d_normalize(DifferentialPair<vector<T, N>> x)
{
    return __fwd_diff(__normalize_impl)(x);
}
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[BackwardDerivativeOf(normalize)]
[ForceInline]
[PreferRecompute]
void __d_distance(inout DifferentialPair<vector<T, N>> x, vector<T, N>.Differential dOut)
{
    return __bwd_diff(__normalize_impl)(x, dOut);
}

// Vector reflect
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
vector<T, N> __reflect_impl(vector<T, N> i, vector<T, N> n)
{
    return  i - n * (T(2.0) * dot(i, n));
}
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[ForwardDerivativeOf(reflect)]
[ForceInline]
DifferentialPair<vector<T, N>> __d_reflect(DifferentialPair<vector<T, N>> i, DifferentialPair<vector<T, N>> n)
{
    return __fwd_diff(__reflect_impl)(i, n);
}
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[BackwardDerivativeOf(reflect)]
[ForceInline]
void __d_reflect(inout DifferentialPair<vector<T, N>> i, inout DifferentialPair<vector<T, N>> n, vector<T, N>.Differential dOut)
{
    return __bwd_diff(__reflect_impl)(i, n, dOut);
}

// Vector refract
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
vector<T, N> __refract_impl(vector<T, N> i, vector<T, N> n, T eta)
{
    let k = T(1.0) - eta * eta * (T(1.0) - dot(n, i) * dot(n, i));
    return (k < T(0.0)) ? vector<T, N>(T(0.0)) : eta * i - (eta * dot(n, i) + sqrt(max(T(0.0),k))) * n;
}
__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[ForwardDerivativeOf(refract)]
[ForceInline]
DifferentialPair<vector<T, N>> __d_refract(DifferentialPair<vector<T, N>> i, DifferentialPair<vector<T, N>> n, DifferentialPair<T> eta)
{
    return __fwd_diff(__refract_impl)(i, n, eta);
}
__generic<T: __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[BackwardDerivativeOf(refract)]
[ForceInline]
void __d_refract(inout DifferentialPair<vector<T, N>> i, inout DifferentialPair<vector<T, N>> n, inout DifferentialPair<T> eta, vector<T, N>.Differential dOut)
{
    return __bwd_diff(__refract_impl)(i, n, eta, dOut);
}

// Sine and cosine
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PrimalSubstituteOf(sincos)]
[PreferRecompute]
void __sincos_impl(T x, out T s, out T c)
{
    s = sin(x);
    c = cos(x);
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[PreferRecompute]
[PrimalSubstituteOf(sincos)]
void __sincos_impl(vector<T, N> x, out vector<T, N> s, out vector<T, N> c)
{
    s = sin(x);
    c = cos(x);
}

__generic<T : __BuiltinFloatingPointType, let N : int, let M : int, let L1 : int, let L2 : int>
[BackwardDifferentiable]
[PrimalSubstituteOf(sincos)]
[PreferRecompute]
void __sincos_impl(matrix<T, N, M> x, out matrix<T, N, M, L1> s, out matrix<T, N, M, L2> c)
{
    s = sin(x);
    c = cos(x);
}


// dst (obsolete)
__generic<T : __BuiltinFloatingPointType>
[BackwardDifferentiable]
[PrimalSubstituteOf(dst)]
vector<T, 4> __dst_impl(vector<T, 4> src0, vector<T, 4> src1)
{
    vector<T, 4> dest;
    dest.x = T(1.0);
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w; ;
    return dest;
}

// Legacy lighting function (obsolete)
[__readNone]
[BackwardDifferentiable]
[PrimalSubstituteOf(lit)]
float4 __lit_impl(float n_dot_l, float n_dot_h, float m)
{
    let ambient = 1.0f;
    let diffuse = max(n_dot_l, 0.0f);
    let specular = ((n_dot_l < 0.0f || n_dot_h < 0.0) ? 0.0 : pow(n_dot_h, m));
    return float4(ambient, diffuse, specular, 1.0f);
}
// Matrix determinant
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[__readNone]
T __determinant_impl(matrix<T,N,N> m)
{
    T result = T(0);
    switch (N)
    {
    case 1:
        result = m[0][0];
        break;
    case 2:
        result = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        break;
    case 3:
        result =  m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
		      - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
			  + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
        break;
    case 4:
        T s00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
		T s01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
		T s02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
		T s03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
		T s04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
		T s05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

		result = m[0][0] * (m[1][1] * s00 - m[1][2] * s01 + m[1][3] * s02)
			 - m[0][1] * (m[1][0] * s00 - m[1][2] * s03 + m[1][3] * s04)
			 + m[0][2] * (m[1][0] * s01 - m[1][1] * s03 + m[1][3] * s05)
			 - m[0][3] * (m[1][0] * s02 - m[1][1] * s04 + m[1][2] * s05);
        break;
    }
    return result;
}
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[ForwardDerivativeOf(determinant)]
[ForceInline]
DifferentialPair<T> __determinant_impl(DifferentialPair<matrix<T,N,N>> m)
{
    return __fwd_diff(__determinant_impl)(m);
}
__generic<T : __BuiltinFloatingPointType, let N : int>
[BackwardDifferentiable]
[BackwardDerivativeOf(determinant)]
[ForceInline]
void __d_determinant(inout DifferentialPair<matrix<T,N,N>> m, T.Differential dOut)
{
    __bwd_diff(__determinant_impl)(m, dOut);
}
