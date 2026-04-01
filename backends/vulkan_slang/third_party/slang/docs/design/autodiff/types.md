
This documentation is intended for Slang contributors and is written from a compiler engineering point of view. For Slang users, see the user-guide at this link: [https://shader-slang.com/slang/user-guide/autodiff.html](https://shader-slang.com/slang/user-guide/autodiff.html)

Before diving into this document, please review the document on [Basics](./basics.md) for the fundamentals of automatic differentiation. 

# Components of the Type System
Here we detail the main components of the type system: the `IDifferentiable` interface to define differentiable types, the `DifferentialPair<T>` type to carry a primal and corresponding differential in a single type. 
We also detail how auto-diff operators are type-checked (the higher-order function checking system), how the `no_diff` decoration can be used to avoid differentiation through attributed types, and the derivative data flow analysis that warns the the user of unintentionally stopping derivatives.

## `interface IDifferentiable`
Defined in core.meta.slang, `IDifferentiable` forms the basis for denoting differentiable types, both within the core module, and otherwise. 
The definition of `IDifferentiable` is designed to encapsulate the following 4 items:
1. `Differential`: The type of the differential value of the conforming type. This allows custom data-structures to be defined to carry the differential values, which may be optimized for space instead of relying solely on compiler synthesis/

Since the computation of derivatives is inherently linear, we only need access to a few operations. These are:

2. `dadd(Differential, Differential) -> Differential`: Addition of two values of the differential type. It's implementation must be associative and commutative, or the resulting derivative code may be incorrect.
3. `dzero() -> Differential`: Additive identity (i.e. the zero or empty value) that can be used to initialize variables during gradient aggregation
4. `dmul<S:__BuiltinRealType>(S, Differential)`: Scalar multiplication of a real number with the differential type. It's implementation must be distributive over differential addition (`dadd`).

Points 2, 3 & 4 are derived from the concept of vector spaces. The derivative values of any Slang function always form a vector space (https://en.wikipedia.org/wiki/Vector_space).

### Derivative member associations
In certain scenarios, the compiler needs information on how the fields in the original type map to the differential type. Particularly, this is a problem when differentiate the implicit construction of a struct through braces (i.e. `{}`), represented by `kIROp_MakeStruct`. We provide the decorator `[DerivativeMember(DifferentialTypeName.fieldName)]` (ASTNode: DerivativeMemberAttribute, IR: kIROp_DerivativeMemberDecoration) to explicitly mark these associations.
Example
```C
struct MyType : IDifferentiable
{
    typealias Differential = MyDiffType;
    float a;

    [DerivativeMember(MyDiffType.db)]
    float b;

    /* ... */
};

struct MyDiffType
{
    float db;
};
```

### Automatic Synthesis of `IDifferentible` Conformances for Aggregate Types
It can be tedious to expect users to hand-write the associated `Differential` type, the corresponding mappings and interface methods for every user-defined `struct` type. For aggregate types, these are trivial to construct by analysing which of their components conform to `IDifferentiable`. 
The synthesis proceeds in roughly the following fashion:
1. `IDifferentiable`'s components are tagged with special decorator `__builtin_requirement(unique_integer_id)` which carries an enum value from `BuiltinRequirementKind`.
2. When checking that types conform to their interfaces, if a user-provided definition does not satisfy a requirement with a built-in tag, we perform synthesis by dispatching to `trySynthesizeRequirementWitness`. 
3. For _user-defined types_, Differential **types** are synthesized during conformance-checking through `trySynthesizeDifferentialAssociatedTypeRequirementWitness` by checking if each constituent type conforms to `IDifferentiable`, looking up the corresponding `Differential` type, and constructing a new aggregate type from these differential types. Note that since it is possible that a `Differential` type of a constituent member has not yet been synthesized, we have additional logic in the lookup system (`trySynthesizeRequirementWitness`) that synthesizes a temporary empty type with a `ToBeSynthesizedModifier`, so that the fields can be filled in later, when the member type undergoes conformance checking.
4. For _user-defined types_, Differential methods (`dadd`, `dzero` and `dmul`) are synthesized in `trySynthesizeDifferentialMethodRequirementWitness` by utilizing the `Differential` member and its `[DifferentialMember]` decorations to determine which fields need to be considered and the base type to use for each field. There are two synthesis patterns. The fully-inductive pattern is used for `dadd` and `dzero` which works by calling `dadd` and `dzero` respectively on the individual fields of the `Differential` type under consideration. 
Example:
```C
// Synthesized from "struct T {FT1 field1; FT2 field2;}"
T.Differential dadd(T.Differential a, T.Differential b)
{
    return Differential(
        FT1.dadd(a.field1, b.field1),
        FT2.dadd(a.field2, b.field2),
    )
}
```
On the other hand, `dmul` uses the fixed-first arg pattern since the first argument is a common scalar, and proceeds inductively on all the other args.
Example:
```C
// Synthesized from "struct T {FT1 field1; FT2 field2;}"
T.Differential dmul<S:__BuiltinRealType>(S s, T.Differential a)
{
    return Differential(
        FT1<S>.dmul(s, a.field1),
        FT2<S>.dmul(s, a.field2),
    )
}
```
5. During auto-diff, the compiler can sometimes synthesize new aggregate types. The most common case is the intermediate context type (`kIROp_BackwardDerivativeIntermediateContextType`), which is lowered into a standard struct once the auto-diff pass is complete. It is important to synthesize the `IDifferentiable` conformance for such types since they may be further differentiated (through higher-order differentiation). This implementation is contained in `fillDifferentialTypeImplementationForStruct(...)` and is roughly analogous to the AST-side synthesis.

### Differentiable Type Dictionaries
During auto-diff, the IR passes frequently need to perform lookups to check if an `IRType` is differentiable, and retrieve references to the corresponding `IDifferentiable` methods. These lookups also need to work on generic parameters (that are defined inside generic containers), and existential types that are interface-typed parameters.

To accommodate this range of different type systems, Slang uses a type dictionary system that associates a dictionary of relevant types with each function. This works in the following way:
1. When `CheckTerm()` is called on an expression within a function that is marked differentiable (`[Differentiable]`), we check if the resolved type conforms to `IDifferentiable`. If so, we add this type to the dictionary along with the witness to its differentiability. The dictionary is currently located on `DifferentiableAttribute` that corresponds to the `[Differentiable]` modifier.

2. When lowering to IR, we create a `DifferentiableTypeDictionaryDecoration` which holds the IR versions of all the types in the dictionary as well as a reference to their `IDifferentiable` witness tables.

3. When synthesizing the derivative code, all the transcriber passes use `DifferentiableTypeConformanceContext::setFunc()` to load the type dictionary. `DifferentiableTypeConformanceContext` then provides convenience functions to lookup differentiable types, appropriate `IDifferentiable` methods, and construct appropriate `DifferentialPair<T>`s.

### Looking up Differential Info on _Generic_ types
Generically defined types are also lowered into the differentiable type dictionary, but rather than having a concrete witness table, the witness table is itself a parameter. When auto-diff passes need to find the differential type or place a call to the IDifferentiable methods, this is turned into a lookup on the witness table parameter (i.e. `Lookup(<InterfaceRequirementKey>, <WitnessTableParameter>)`). Note that these lookups instructions are inserted into the generic parent container rather than the inner most function. 
Example:
```C
T myFunc<T:IDifferentiable>(T a)
{
    return a * a;
}

// Reverse-mode differentiated version
void bwd_myFunc<T:IDifferentiable>(
    inout DifferentialPair<T> dpa,
    T.Differential dOut) // T.Differential is Lookup('Differential', T_Witness_Table)
{
    T.Differential da = T.dzero(); // T.dzero is Lookup('dzero', T_Witness_Table)

    da = T.dadd(dpa.p * dOut, da); // T.dadd is Lookup('dadd', T_Witness_Table)
    da = T.dadd(dpa.p * dOut, da);

    dpa = diffPair(dpa.p, da);
}
```

### Looking up Differential Info on _Existential_ types
Existential types are interface-typed values, where there are multiple possible implementations at run-time. The existential type carries information about the concrete type at run-time and is effectively a 'tagged union' of all possible types.

#### Differential type of an Existential
The differential type of an existential type is tricky to define since our type system's only restriction on the `.Differential` type is that it also conforms to `IDifferentiable`. The differential type of any interface `IInterface : IDifferentiable` is therefore the interface type `IDifferentiable`. This is problematic since Slang generally requires a static `anyValueSize` that must be a strict upper bound on the sizes of all conforming types (since this size is used to allocate space for the union). Since `IDifferentiable` is defined in the core module `core.meta.slang` and can be used by the user, it is impossible to define a reliable bound. 
We instead provide a new **any-value-size inference** pass (`slang-ir-any-value-inference.h`/`slang-ir-any-value-inference.cpp`) that assembles a list of types that conform to each interface in the final linked IR and determines a relevant upper bound. This allows us to ignore types that conform to `IDifferentiable` but aren't used in the final IR, and generate a tighter upper bound. 

**Future work:**
This approach, while functional, creates a locality problem since the size of `IDifferentiable` is the max of _all_ types that conform to `IDifferentiable` in visible modules, even though we only care about the subset of types that appear as `T.Differential` for `T : IInterface`. The reason for this problem is that upon performing an associated type lookup, the Slang IR drops all information about the base interface that the lookup starts from and only considers the constraint interface (in this case `Differential : IDifferentiable`). 
There are several ways to resolve this issue, including (i) a static analysis pass that determines the possible set of types at each use location and propagates them to determine a narrower set of types, or (ii) generic (or 'parameterized') interfaces, such as `IDifferentiable<T>` where each version can have a different set of conforming types.

<!--#### IDifferentiable Method lookups on an Existential
All other method lookups are performed using existential-type lookups on the existential parameter. The idea is that existential-typed parameters come with a witness-table component that can be accessed by invoking `kIROp_ExtractExistentialWitnessTable` on them. This allows us to look up the `dadd`/`dzero` methods on this witness table in the same way as we did for generic types.-->

Example:
```C
interface IInterface : IDifferentiable
{
    [Differentiable]
    This foo(float val);

    [Differentiable]
    float bar();
};

float myFunc(IInterface obj, float a)
{
    IInterface k = obj.foo(a);
    return k.bar();
}

// Reverse-mode differentiated version (in pseudo-code corresponding to IR, some of these will get lowered further)
void bwd_myFunc(
    inout DifferentialPair<IInterface> dpobj,
    inout DifferentialPair<float> dpa,
    float.Differential dOut) // T.Differential is Lookup('Differential', T_Witness_Table)
{
    // Primal pass..
    IInterface obj = dpobj.p;
    IInterface k = obj.foo(a);
    // .....

    // Backward pass
    DifferentialPair<IInterface> dpk = diffPair(k);
    bwd_bar(dpk, dOut);
    IDifferentiable dk = dpk.d; // Differential of `IInterface` is `IDifferentiable`

    DifferentialPair<IInterface> dp = diffPair(dpobj.p);
    bwd_foo(dpobj, dpa, dk);
}

```

#### Looking up `dadd()` and `dzero()` on Existential Types
There are two distinct cases for lookup on an existential type. The more common case is the closed-box existential type represented simply by an interface. Every value of this type contains a type identifier & a witness table identifier along with the value itself.  The less common case is when the function calls are performed directly on the value after being cast to the concrete type.

**`dzero()` for "closed" Existential type: The `NullDifferential` Type**
For concrete and even generic types, we can initialize a derivative accumulator variable by calling the appropriate `Type.dzero()` method. This is unfortunately not possible when initializing an existential differential (which is currently of type `IDifferentiable`), since we must also initialize the type-id of this existential to one of the implementations, but we do not know which one yet since that is a run-time value that only becomes known after the first differential value is generated.

To get around this issue, we declare a special type called `NullDifferential` that acts as a "none type" for any `IDifferentiable` existential object. 

**`dadd()` for "closed" Existential types: `__existential_dadd`**
We cannot directly use `dadd()` on two existential differentials of type `IDifferentiable` because we must handle the case where one of them is of type `NullDifferential` and `dadd()` is only defined for differentials of the same type. 
We handle this currently by synthesizing a special method called `__existential_dadd` (`getOrCreateExistentialDAddMethod` in `slang-ir-autodiff.cpp`) that performs a run-time type-id check to see if one of the operand is of type `NullDifferential` and returns the other operand if so. If both are non-null, we dispatch to the appropriate `dadd` for the concrete type.

**`dadd()` and `dzero()` for "open" Existential types**
If we are dealing with values of the concrete type (i.e. the opened value obtained through `ExtractExistentialValue(ExistentialParam)`). Then we can perform lookups in the same way we do for generic type. All existential parameters come with a witness table. We insert instructions to extract this witness table and perform lookups accordingly. That is, for `dadd()`, we use `Lookup('dadd', ExtractExistentialWitnessTable(ExistentialParam))` and place a call to the result.

## `struct DifferentialPair<T:IDifferentiable>`
The second major component is `DifferentialPair<T:IDifferentiable>` that represents a pair of a primal value and its corresponding differential value. 
The differential pair is primarily used for passing & receiving derivatives from the synthesized derivative methods, as well as for block parameters on the IR-side.
Both `fwd_diff(fn)` and `bwd_diff(fn)` act as function-to-function transformations, and so the Slang front-end translates the type of `fn` to its derivative version so the arguments can be type checked.

### Pair type lowering.
The differential pair type is a special type throughout the AST and IR passes (AST Node: `DifferentialPairType`, IR: `kIROp_DifferentialPairType`) because of its use in front-end semantic checking and when synthesizing the derivative code for the functions. Once the auto-diff passes are complete, the pair types are lowering into simple `struct`s so they can be easily emitted (`DiffPairLoweringPass` in `slang-ir-autodiff-pairs.cpp`). 
We also define additional instructions for pair construction (`kIROp_MakeDifferentialPair`) and extraction (`kIROp_DifferentialPairGetDifferential` & `kIROp_DifferentialPairGetPrimal`) which are lowered into struct construction and field accessors, respectively.

### "User-code" Differential Pairs
Just as we use special IR codes for differential pairs because they have special handling in the IR passes, sometimes differential pairs should be _treated as_ regular struct types during the auto-diff passes.
This happens primarily during higher-order differentiation when the user wishes to differentiate the same code multiple times. 
Slang's auto-diff approaches this by rewriting all the relevant differential pairs into 'irrelevant' differential pairs (`kIROp_DifferentialPairUserCode`) and 'irrelevant' accessors (`kIROp_DifferentialPairGetDifferentialUserCode`, `kIROp_DifferentialPairGetPrimalUserCode`) at the end of **each auto-diff iteration** so that the next iteration treats these as regular differentiable types. 
The user-code versions are also lowered into `struct`s in the same way.

## Type Checking of Auto-Diff Calls (and other _higher-order_ functions)
Since `fwd_diff` and `bwd_diff` are represented as higher order functions that take a function as an input and return the derivative function, the front-end semantic checking needs some notion of higher-order functions to be able to check and lower the calls into appropriate IR.

### Higher-order Invocation Base: `HigherOrderInvokeExpr`
All higher order transformations derive from `HigherOrderInvokeExpr`. For auto-diff there are two possible expression classes `ForwardDifferentiateExpr` and `BackwardDifferentiateExpr`, both of which derive from this parent expression.

### Higher-order Function Call Checking: `HigherOrderInvokeExprCheckingActions`
Resolving the concrete method is not a trivial issue in Slang, given its support for overloading, type coercion and more. This becomes more complex with the presence of a function transformation in the chain. 
For example, if we have `fwd_diff(f)(DiffPair<float>(...), DiffPair<double>(...))`, we would need to find the correct match for `f` based on its post-transform argument types.

To facilitate this we use the following workflow:
1. The `HigherOrderInvokeExprCheckingActions` base class provides a mechanism for different higher-order expressions to implement their type translation (i.e. what is the type of the transformed function). 
2. The checking mechanism passes all detected overloads for `f` through the type translation and assembles a new group out of the results (the new functions are 'temporary')
3. This new group is used by `ResolveInvoke` when performing overload resolution and type coercion using the user-provided argument list.
4. The resolved signature (if there is one) is then replaced with the corresponding function reference and wrapped in the appropriate higher-order invoke.

**Example:**

Let's say we have two functions with the same name `f`: (`int -> float`, `double, double -> float`)
and we want to resolve `fwd_diff(f)(DiffPair<float>(1.0, 0.0), DiffPair<float>(0.0, 1.0))`.

The higher-order checking actions will synthesize the 'temporary' group of translated signatures (`int -> DiffPair<float>`, `DiffPair<double>, DiffPair<double> -> DiffPair<float>`). 
Invoke resolution will then narrow this down to a single match (`DiffPair<double>, DiffPair<double> -> DiffPair<float>`) by automatically casting the `float`s to `double`s. Once the resolution is complete, 
we return `InvokeExpr(ForwardDifferentiateExpr(f : double, double -> float), casted_args)` by wrapping the corresponding function in the corresponding higher-order expr

## Attributed Types (`no_diff` parameters)

Often, it will be necessary to prevent gradients from propagating through certain parameters, for correctness reasons. For example, values representing random samples are often not differentiated since the result may be mathematically incorrect.

Slang provides the `no_diff` operator to mark parameters as non-differentiable, even if they use a type that conforms to `IDifferentiable`

```C
float myFunc(float a, no_diff float b)
{
    return a * b;
}

// Resulting fwd-mode derivative:
DiffPair<float> myFunc(DiffPair<float> dpa, float b)
{
    return diffPair(dpa.p * b, dpa.d * b);
}
```

Slang uses _OpAttributedType_ to denote the IR type of such parameters. For example, the lowered type of `b` in the above example is `OpAttributedType(OpFloat, OpNoDiffAttr)`. In the front-end, this is represented through the `ModifiedType` AST node. 

Sometimes, this additional layer can get in the way of things like type equality checks and other mechanisms where the `no_diff` is irrelevant. Thus, we provide the `unwrapAttributedType` helper to remove attributed type layers for such cases.

## Derivative Data-Flow Analysis
Slang has a derivative data-flow analysis pass that is performed on a per-function basis immediately after lowering to IR and before the linking step (`slang-ir-check-differentiability.h`/`slang-ir-check-differentiability.cpp`). 

The job of this pass is to enforce that instructions that are of a differentiable type will propagate a derivatives, unless explicitly dropped by the user through `detach()` or `no_diff`. The reason for this is that Slang requires functions to be decorated with `[Differentiable]` to allow it to propagate derivatives. Otherwise, the function is considered non-differentiable, and effectively produces a 0 derivative. This can lead to frustrating situations where a function may be dropping non-differentiable on purpose. Example:
```C
float nonDiffFunc(float x)
{
    /* ... */
}

float differentiableFunc(float x) // Forgot to annotate with [Differentiable]
{
    /* ... */
}

float main(float x)
{
    // User doesn't realise that the function that is supposed to be differentiable is not 
    // getting differentiated, because the types here are all 'float'.
    // 
    return nonDiffFunc(x) * differentiableFunc(x);
}
```

The data-flow analysis step enforces that non-differentiable functions used in a differentiable context should get their derivative dropped explicitly. That way, it is clear to the user whether a call is getting differentiated or dropped.

Same example with `no_diff` enforcement:
```C
float nonDiffFunc(float x)
{
    /* ... */
}

[Differentiable]
float differentiableFunc(float x)
{
    /* ... */
}

float main(float x)
{
    return no_diff(nonDiffFunc(x)) * differentiableFunc(x);
}
```

A `no_diff` can only be used directly on a function call, and turns into a `TreatAsDifferentiableDecoration` that indicates that the function will not produce a derivative.

The derivative data-flow analysis pass works similar to a standard data-flow pass:
1. We start by assembling a set of instructions that 'produce' derivatives by starting with the parameters of differentiable types (and without an explicit `no_diff`), and propagating them through each instruction in the block. An inst carries a derivative if there one of its operands carries a derivative, and the result type is differentiable.
2. We then assemble a set of instructions that expect a derivative. These are differentiable operands of differentiable functions (unless they have been marked by `no_diff`). We then reverse-propagate this set by adding in all differentiable operands (and repeating this process).
3. During this reverse-propagation, if there is any `OpCall` in the 'expect' set that is not also in the 'produce' set, then we have a situation where the gradient hasn't been explicitly dropped, and we create a user diagnostic.
