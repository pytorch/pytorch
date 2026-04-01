This document details auto-diff-related decorations that are lowered in to the IR to help annotate methods with relevant information.

## `[Differentiable]`
The `[Differentiable]` attribute is used to mark functions as being differentiable. The auto-diff process will only touch functions that are marked explicitly as `[Differentiable]`. All other functions are considered non-differentiable and calls to such functions from a differentiable function are simply copied as-is with no transformation.

Further, only `[Differentiable]` methods are checked during the derivative data-flow pass. This decorator is translated into `BackwardDifferentiableAttribute` (which implies both forward and backward differentiability), and then lowered into the IR `OpBackwardDifferentiableDecoration`

**Note:** `[Differentiable]` was previously implemented as two separate decorators `[ForwardDifferentiable]` and `[BackwardDifferentiable]` to denote differentiability with each type of auto-diff transformation. However, these are now **deprecated**. The preferred approach is to use only `[Differentiable]`

`fwd_diff` and `bwd_diff` cannot be directly called on methods that don't have the `[Differentiable]` tag (will result in an error). If non-`[Differentiable]` methods are called from within a `[Differentiable]` method, they must be wrapped in `no_diff()` operation (enforced by the [derivative data-flow analysis pass](./types.md#derivative-data-flow-analysis) )

### `[Differentiable]` for `interface` Requirements
The `[Differentiable]` attribute can also be used to decorate interface requirements. In this case, the attribute is handled in a slightly different manner, since we do not have access to the concrete implementations.

The process is roughly as follows:
1. During the semantic checking step, when checking a method that is an interface requirement (in `checkCallableDeclCommon` in `slang-check-decl.cpp`), we check if the method has a `[Differentiable]` attribute
2. If yes, we construct create a set of new method declarations, one for the forward-mode derivative (`ForwardDerivativeRequirementDecl`) and one for the reverse-mode derivative (`BackwardDerivativeRequirementDecl`), with the appropriate translated function types and insert them into the same interface.
3. Insert a new member into the original method to reference the new declarations (`DerivativeRequirementReferenceDecl`)
4. When lowering to IR, the `DerivativeRequirementReferenceDecl` member is converted into a custom derivative reference by adding the `OpBackwardDerivativeDecoration(deriv-fn-req-key)` and `OpForwardDerivativeDecoration(deriv-fn-req-key)` decorations on the primal method's requirement key.

Here is an example of what this would look like:

```C
interface IFoo
{
    [Differentiable]
    float bar(float);
};

// After checking & lowering
interface IFoo_after_checking_and_lowering
{
    [BackwardDerivative(bar_bwd)]
    [ForwardDerivative(bar_fwd)]
    float bar(float);

    void bar_bwd(inout DifferentialPair<float>, float);

    DifferentialPair<float> bar_fwd(DifferentialPair<float>);
};
```

**Note:** All conforming types must _also_ declare their corresponding implementations as differentiable so that their derivative implementations are synthesized to match the interface signature. In this sense, the `[Differentiable]` attribute is part of the functions signature, so a `[Differentiable]` interface requirement can only be satisfied by a `[Differentiable]` function implementation

### `[TreatAsDifferentiable]`
In large codebases where some interfaces may have several possible implementations, it may not be reasonable to have to mark all possible implementations with `[Differentiable]`, especially if certain implementations use hacks or workarounds that need additional consideration before they can be marked `[Differentiable]`

In such cases, we provide the `[TreatAsDifferentiable]` decoration (AST node: `TreatAsDifferentiableAttribute`, IR: `OpTreatAsDifferentiableDecoration`), which instructs the auto-diff passes to construct an 'empty' function that returns a 0 (or 0-equivalent) for the derivative values. This allows the signature of a `[TreatAsDifferentiable]` function to match a `[Differentiable]` requirement without actually having to produce a derivative.

## Custom derivative decorators
In many cases, it is desirable to manually specify the derivative code for a method rather than let the auto-diff pass synthesize it from the method body. This is usually desirable if:
1. The body of the method is too complex, and there is a simpler, mathematically equivalent way to compute the same value (often the case for intrinsics like `sin(x)`, `arccos(x)`, etc..)
2. The method involves global/shared memory accesses, and synthesized derivative code may cause race conditions or be very slow due to overuse of synchronization. For this reason Slang assumes global memory accesses are non-differentiable by default, and requires that the user (or the core module) define separate accessors with different derivative semantics.

The Slang front-end provides two sets of decorators to facilitate this:
1. To reference a custom derivative function from a primal function: `[ForwardDerivative(fn)]` and `[BackwardDerivative(fn)]` (AST Nodes: `ForwardDerivativeAttribute`/`BackwardDerivativeAttribute`, IR: `OpForwardDervativeDecoration`/`OpBackwardDerivativeDecoration`), and 
2. To reference a primal function from its custom derivative function: `[ForwardDerivativeOf(fn)]` and `[BackwardDerivativeOf(fn)]` (AST Nodes: `ForwardDerivativeAttributeOf`/`BackwardDerivativeAttributeOf`). These attributes are useful to provide custom derivatives for existing methods in a different file without having to edit/change that module. For instance, we use `diff.meta.slang` to provide derivatives for the core module functions in `hlsl.meta.slang`. When lowering to IR, these references are placed on the target (primal function). That way both sets of decorations are lowered on the primal function.

These decorators also work on generically defined methods, as well as struct methods. Similar to how function calls work, these decorators also work on overloaded methods (and reuse the `ResolveInoke` infrastructure to perform resolution)

### Checking custom derivative signatures
To ensure that the user-provided derivatives agree with the expected signature, as well as resolve the appropriate method when multiple overloads are available, we check the signature of the custom derivative function against the translated version of the primal function. This currently occurs in `checkDerivativeAttribute()`/`checkDerivativeOfAttribute()`. 

The checking process re-uses existing infrastructure from `ResolveInvoke`, by constructing a temporary invoke expr to call the user-provided derivative using a set of 'imaginary' arguments according to the translated type of the primal method. If `ResolveInvoke` is successful, the provided derivative signature is considered to be a match. This approach also automatically allows us to resolve overloaded methods, account for generic types and type coercion.

## `[PrimalSubstitute(fn)]` and `[PrimalSubstituteOf(fn)]`
In some cases, we face the opposite problem that inspired custom derivatives. That is, we want the compiler to auto-synthesize the derivative from the function body, but there _is_ no function body to translate.
This frequently occurs with hardware intrinsic operations that are lowered into special op-codes that map to hardware units, such as texture sampling & interpolation operations. 
However, these operations do have reference 'software' implementations which can be used to produce the derivative.

To allow user code to use the fast hardware intrinsics for the primal pass, but use synthesized derivatives for the derivative pass, we provide decorators `[PrimalSubstitute(ref-fn)]` and `[PrimalSubstituteOf(orig-fn)]` (AST Node: `PrimalSubstituteAttribute`/`PrimalSubstituteOfAttribute`, IR: `OpPrimalSubstituteDecoration`), that can be used to provide a reference implementation for the auto-diff pass.

Example:
```C
[PrimalSubstitute(sampleTexture_ref)]
float sampleTexture(TexHandle2D tex, float2 uv)
{
    // Hardware intrinsics
}

float sampleTexture_ref(TexHandle2D tex, float2 uv)
{
    // Reference SW implementation.
}

void sampleTexture_bwd(TexHandle2D tex, inout DifferentialPair<float2> dp_uv, float dOut)
{
    // Backward derivate code synthesized using the reference implementation.
}
```

The implementation of `[PrimalSubstitute(fn)]` is relatively straightforward. When the transcribers are asked to synthesize a derivative of a function, they check for a `OpPrimalSubstituteDecoration`, and swap the current function out for the substitute function before proceeding with derivative synthesis.
