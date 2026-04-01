This documentation is intended for Slang contributors and is written from a compiler engineering point of view. For Slang users, see the user-guide at this link: [https://shader-slang.com/slang/user-guide/autodiff.html](https://shader-slang.com/slang/user-guide/autodiff.html)

# Overview of Automatic Differentiation's IR Passes
In this document we will detail how Slang's auto-diff passes generate valid forward-mode and reverse-mode derivative functions. Refer to [Basics](./basics.md) for a review of the two derivative propagation methods and their mathematical connotations & [Types](./types.md) for a review of how types are handled under differentiation.

## Auto-Diff Pass Invocation
Note that without an explicit auto-diff instruction (`fwd_diff(fn)` or `bwd_diff(fn)`) from the user present anywhere in the code, none of the auto-diff passes will do anything. 

Auto-diff processing operates on a function-by-function basis. Most of the logic is contained in `AutoDiffPass::processReferencedFunctions`. Here is a high-level workflow:
1. Scanning reachable insts in the module looking for `IRForwardDifferentiate` or `IRBackwardDifferentiate` operations. These instructions are added onto a work-list. The subject of a differentiate inst may be a plain function (`IRFunc`), a specialize inst (`IRSpecialize(a : IRGeneric, ...)`) in case of a generic method, or a lookup inst (`IRLookupWitness(a : IRWitnessTableType)`) when differentiating a method of an interface.

2. Dispatch each differentiation request through the appropriate 'transcriber' class. A transcriber (implements `AutodiffTranscriberBase`) is responsible for accepting a differentiation request and resolving it by replacing it with a generated function or a call to an already existing function that computes its derivative. 

3. Once all currently available derivative insts have been dispatched, the follow-up work-list is checked for more transcription requests. This is a global list that all transcribers can add more follow-up work to. As an example, differentiating a function that calls another function will generate a follow-up task for this inner function, even though the latter never appears directly in a `IRForwardDifferentiate` or `IRBackwardDifferentiate` inst. 
At this step, there are 2 other variants that can appear `IRBackwardDifferentiatePrimal` and `IRBackwardDifferentiatePropagate` (though these can't be invoked by the user directly).

4. This process from (1.) is run in a loop. This is because we can have nested differentiation requests such as `IRForwardDifferentiate(IRBackwardDifferentiate(a : IRFuncType))`. The inner request is processed in the first pass, and the outer request gets processed in the next pass.

## Auto-Diff Passes for `IRForwardDifferentiate`
For forward-mode derivatives, we only require a single pass implemented wholly in `ForwardDiffTranscriber`. This implements the linearization algorithm, which roughly follows this logic:

1. Create a clone of the original function
2. Perform pre-autodiff transformations, the most  
    a. **Temp-Var-For-Mutable-Params** Using `IRVar` to load from 
    b. **Linkage-Removal**: This is simply so the cloned function can be eliminated by DCE after auto-diff is complete
    c. **Force-Inline**: Inline all `__unsafeForceEarlyInline` & `[ForceInline]` functions _prior_ to auto-diff, so their contents can be included in the differentiation pass (even if they aren't actually marked as `[Differentiable]`)

3. Create a new blank function for the fwd-mode func (usually named `s_fwd_<ORIGINAL-FUNC-NAME>`) with the function type derived by transforming the original function type (See [Types](./types.md) for more information).
4. Create new blocks into the new fwd-func for each block in the orig-func.
5. Go through instructions in each block and dispatch to the appropriate generator function to emit the derivative logic into the corresponding block in the fwd-func. Each generator method is responsible for cloning in the original instruction as well as emitting a corresponding derivative instruction. `mapPrimalInst` and `mapDifferentialInst` are used to keep track of prior results so that operands for new instructions can be looked up.

The generator for each instruction computes the forward-mode derivative of each *instruction* using the Jacobian method that is detailed in [Basics#Jacobian-Method](./basics.md#jacobian-method-generate-forward--and-reverse-mode-derivatives-from-first-principles). Since forward-mode derivatives can be composed in the same order as the original instructions, our generation process goes through instructions in each block in the order that they appear, creating differential insts which act as operands for future insts.

Here's an example of this in IR-form

```Rust
OpModule
{
    %ftype = OpFuncType (%float) (%float) (%float)
    %f = OpFunc : %ftype
    {
        %b = OpBlock
        {
            %a = OpParam : %float
            %b = OpParam : %float

            %1 = OpAdd %a %b : %float
            %2 = OpAdd %1 %1 : %float

            OpReturn %2
        }
    }

    // Generated function type
    %dpfloat = OpDifferentialPairType (%float) (%witness_that_float_is_idifferentiable)
    %ftype_fwd = OpFuncType (%dpfloat) (%dpfloat) (%dpfloat)

    // Generated function
    %f_fwd = OpFunc : %ftype_fwd
    {
        %b_fwd = OpBlock
        {
            %dpa = OpParam : %dpfloat // Convert params to differential pair types
            %dpb = OpParam : %dpfloat // Convert params to differential pair types

            // Split block inputs into primals and differentials
            %a = OpDifferentialPairGetPrimal %dpa : %float
            %da = OpDifferentialPairGetDifferential %dpa : %float

            %b = OpDifferentialPairGetPrimal %dpb : %float
            %db = OpDifferentialPairGetDifferential %dpb : %float

            // Clone the primal inst for %1
            %1_primal = OpAdd %a %b : %float

            // Generate the diff inst for %1
            // Here, we consider the 'mini-function' Add(a,b) = a + b, and use the Jacobian method
            // to get the result that the fwd-mode derivative should be:
            // DAdd((a, da), (b, db)) = da + db = Add(da, db)
            // 
            %1_diff = OpAdd %da %db : %float
            
            // Do the same for the next inst (%2): clone in the primal
            // by looking up primal versions of the operands.
            //
            %2_primal = OpAdd %1_primal %1_primal : %float

            // Then, generate the derivative inst by looking up the differential
            // versions of the operands.
            // 
            %2_diff = OpAdd %1_diff %1_diff : %float

            // Return both the primal and differential
            %2_pair = OpDifferentialPairMakePair %2_primal %2_diff : %dpfloat
            OpReturn %2_pair
        }
    }
}
```

**Multiple Differential Insts:**
In the above example, the derivative of each inst was a single inst. This is not always the case. 
For instance, `OpMul %a %b` translates to **three** insts:
```Rust
%1 = OpMul %a_diff %b_primal : %float
%2 = OpMul %a_primal %b_diff : %float
%3 = OpAdd %1 %2 : %float
```

**Combined Primal & Differential Insts:**
In some cases, there is not need to clone in the primal inst since both the primal and differential can be computed in a single inst. An example is `IRCall`, whose deriavative only needs a single call (though it needs plenty of insts to pair and unpair arguments)
```Rust
// Original inst
%1 = OpCall %func %a %b : %float

//
// Upon differentiation:

// Pack args into pairs
%a_pair = OpDifferentialPairMakePair %a_primal %a_diff : %dpfloat
%b_pair = OpDifferentialPairMakePair %b_primal %b_diff : %dpfloat

// Call into fwd-mode deriv which computes *both* primal and differential
// values.
//
%func_fwd = OpForwardDifferentiate %func : %functype_fwd
%1_pair = OpCall %func_fwd %a_pair %b_pair : %float

// Split into primal and differential so they can be used for future insts.
%1_primal = OpDifferentialPairGetPrimal %1_pair : %float
%1_diff = OpDifferentialPairGetDifferential %1_pair : %float

```


### Phi Arguments
Block arguments are handled the same way as function arguments (which in the Slang IR, are also simply block arguments of the first block), and are converted into pair type arguments, with `OpDifferentialPairGetPrimal` and `OpDifferentialPairGetDifferential` insts automatically added to extract the primal and differential parts of each argument.


## Auto-Diff Passes for `IRBackwardDifferentiate`

For reverse-mode derivatives, we need several passes that also includes differentiating the forward-mode derivative. Most of this logic is contained in `BackwardDiffTranscriberBase::transcribeFuncImpl`. These passes are inspired by the paper ["You Only Linearize Once: Tangents Transpose to Gradients"](https://arxiv.org/abs/2204.10923), which describes this approach in a functional language setting. These passes extend these ideas to work for a general-purpose imperative language structure.

### 1. Preparation
The reverse-mode derivative generation involves a lot of large scale control-flow manipulation, including a CFG reversal step that aims to construct a method that flows from the end of the function to the beginning in order to compose reverse-mode derivatives.
To avoid having to deal with too many corner cases (and the maintainability issues that come with it), we bring the function to a 'normal form' before running our differentiation steps. This greatly simplifies the logic of the future passes.

Another high-level goal of these transformations is to bring the control-flow graph to a **reversible** form. That is, we can represent the reverse of control-flow graph using existing Slang constructs (`IRIfElse`, `IRUnconditionalBranch`, `IRLoop` and `IRSwitch`). This is not necessarily true of any valid Slang IR, so we perform additional transformations.

Note: These transformations are always applied onto a temporary clone of the original function. The original function is never touched so as to not affect its use in non-autodiff contexts.

Specifically we:
1. Bring the function into **single-return form**: If there are multiple blocks with return statements (i.e. multiple exit points) in a function, we eliminate this by wrapping the complete function body in a trivial loop (i.e. a single-iteration loop) and replacing existing return statements with breaks (or multi-level breaks) into its break block, which serves as the unique exit point for the function. This pass is currently contained in `convertFuncToSingleReturnForm()`

2. Eliminate **continue** statements: Loop continue statements introduce a reversibility problem. Since the forward loop can have multiple exit point, the reverse loop needs to have multiple entry points. Slang's loops do not support this. So, we eliminate these statements wrapping the body of the loop in another trivial loop (i.e. single-iteration loop) and turning the **continue** statements into **break** statements. This also involves writing **break** statements in the original loop into **multi-level** breaks.
    
    Here is an example:
    ```C
    // Original loop
    for (uint i = 0; i < N; i++)
    {
        if (i > 5)
            continue;

        if (i > 9)
            break;
        
        x = x + i;
    }

    // After continue-elimination
    outer_for:
    for (uint i = 0; i < N; i++)
    {
        inner_for:
        for (;;)
        {
            if (i > 5)
                break;

            if (i > 9)
                break outer_for; // multi-level break
            
            x = x + i;

            break;
        }
    }
    ```

3. Eliminate **multi-level breaks**: Slang supports breaking out to an outer loop. Unfortunately, this operation is hard to reverse since Slang (and shading languages in general) do not support arbitrary `goto` statements. We eliminate multi-level breaks by assigning each nested loop a nesting index (a constant `uint` denoting the nesting level). All break statements are rewritten to break out to the immediate next level (i.e. a standard break) with a index parameter denoting the intended break level. This parameter is checked at each level and if the break index does not match the level index, we break again to the immediate upper level. This pass is currently contained in `eliminateMultiLevelBreakForFunc`

    Continuing the above example, here is the code after multi-level break elimination.
    ```C
    // After multi-level-break elimination
    uint level = -1;
    for (uint i = 0; i < N; i++)
    {
        for (;;)
        {
            if (i > 5)
            {
                level = 1;
                break;
            }

            if (i > 9)
            {
                level = 0;
                break;
            }
            
            x = x + i;

            level = 1;
            break;
        }

        if (level != 1) // Level check immediately after breaking out of each loop.
            break;
    }
    ```

4. Eliminate **break** statements (enclosed in `normalizeCFG()`): Break statements also pose the same problem as continue statements (i.e. multiple exit points require the reverse loop to have multiple entry points, and Slang does not have a primitive for this). We eliminate break statements by introducing a boolean break flag which is set to `false` to indicate a break instead of using the break statement. Each *region* is enclosed in a if-else statement that checks the break flag and skips to the end if necessary.

    Break elimination proceeds with the following steps;

    Here is the above example code after break elimination.
    ```C
    // After break elimination
    
    uint level = -1;
    bool bflag_0 = true; // for outer loop (true => keep-going, false => break)

    for (uint i = 0; (i < N) && bflag_0; i++) // Insert flag into the loop condition (&& with the current condition)
    {
        bool bflag_1 = true; // for inner loop (true => keep-going, false => break)

        for (;bflag_1;) // Insert flag into the loop condition
        {
            if (i > 5)
            {
                level = 1;
                bflag_1 = false; // break
            }

            // Region after any break statement is enclosed in a 
            // if-else check.
            // 
            if (bflag_1)
            {
                if (i > 9)
                {
                    level = 0;
                    bflag_1 = false; // break
                }

                // Another if-else enclosure, this time for the second
                // break.
                // 
                if (bflag)
                {
                    x = x + i;
                    level = 1;
                }

                bflag_1 = false;
            }
        }

        if (level != 1)
        {
            bflag_0 = false;
        }
    }
    ```

    **Extra evaluation of the condition block:** The CFG normalization passes always attempt to preserve the equivalence of the original function while manipulating the control-flow constructs (i.e. ensure that the transformed code always computes the same thing). However, there is one corner-case exception: after break-elimination, the loop condition code can be evaluated 1 additional time, since we don't directly break out of the loop, but go through an extra loop condition check. This becomes important during the checkpointing step, when arrays are allocated to hold loop variables. The array bounds must account for an additional loop iteration to avoid correctness problems.


### 2. Linearization with Inst-Tagging
This is the same as generating the forward-derivative function, and is in-fact handled in the same way, by invoking `ForwardDiffTranscriber`. The **inst-tagging** part of this pass is not necessary for forward-mode auto-diff (simply discarded after the auto-diff pass), but is essential for reverse-mode.

**Inst-Tagging:** This pass also **tags** every instruction and block with either `IRPrimalInstDecoration`, `IRDifferentialInstDecoration` or `IRMixedDifferentialnstDecoration`, depending on whether an instruction contains/computes/reads/writes a primal value, a differential value or both. 

This assignment is according to the following rules:
1. The result of `.getDifferential()` from an inst of `IRDifferentialPairType` is a *differential* inst and `.getPrimal()` is a primal inst **NOTE:** This does not apply to `IRDifferentialPairUserCodeType`, all of whose operations yield a *primal* inst.
2. Further, any inst which contains a differential inst as an operand **AND** whose output value may be affected by this operand is a differential inst (e.g. if `isDifferentialInst(a) = true` then `isDifferentialInst( IRMul(a, b) ) = true`)
3. If an inst contains multiple outputs, *some* of which are differential and the others are primal, then these are *mixed-differential* insts. E.g. (a value of `IRDifferentialPairType` contains both a primal and differential value, and similarly a call of the form `IRCall(IRForwardDifferentiate(inner_fn))(...)` results in a mixed differential type since the primal part is not affect by differential inputs)
4. All other insts are *primal* by default.
5. Blocks are marked differential or primal if they contain **ONLY** differential or primal insts (respectively). Otherwise they are marked mixed-differential. The vast majority of blocks are mixed-differential.

Correct tag information is critical for the next steps to correctly transform the forward-mode derivative into the reverse-mode derivative function.

Here's the same forward-mode example, but with insts tagged accordingly
```Rust
OpModule
{
    // Generated function type
    ...

    // Generated function
    ...
        [OpMixedDifferentiaInstDecoration]
        %b_fwd = OpBlock
        {
            // Block params are mixed differentials since they carry both
            // primal and differential values
            // 
            [OpMixedDifferentialInstDecoration]
            %dpa = OpParam : %dpfloat 
            [OpMixedDifferentialInstDecoration]
            %dpb = OpParam : %dpfloat 

            [OpPrimalInstDecoration]
            %a = OpDifferentialPairGetPrimal %dpa : %float

            [OpDifferentialInstDecoration]
            %da = OpDifferentialPairGetDifferential %dpa : %float

            [OpPrimalInstDecoration]
            %b = OpDifferentialPairGetPrimal %dpb : %float

            [OpDifferentialInstDecoration]
            %db = OpDifferentialPairGetDifferential %dpb : %float

            [OpPrimalInstDecoration]
            %1_primal = OpAdd %a %b : %float

            [OpDifferentialInstDecoration]
            %1_diff = OpAdd %da %db : %float

            [OpPrimalInstDecoration]
            %2_primal = OpAdd %1_primal %1_primal : %float

            [OpDifferentialInstDecoration]
            %2_diff = OpAdd %1_diff %1_diff : %float

            // Return both the primal and differential
            [OpMixedDifferentialInstDecoration]
            %2_pair = OpDifferentialPairMakePair %2_primal %2_diff : %dpfloat

            [OpDifferentialInstDecoration]
            OpReturn %2_pair
        }
    ...
}
```

### 3. Unzipping
Implemented by `DiffUnzipPass`, this pass is responsible for **separating** primal instructions from differential instructions (as denoted by their decorations), by creating a full set of duplicate blocks that start **after** the last block, i.e. return block (the return statement is removed).

This separation is possible because the computation of a differential inst may include primal operands but a primal inst can never use a differential operand. 

The unzipping pass uses the decorations from the linearization step to figure out which instructions need to be moved.

The separation process uses the following high-level logic:
1. Create two clones of all the blocks in the provided function (one for primal insts, one for differential insts), and hold a mapping between each original (mixed) block to each primal and differential block. The return statement of the current final block is **removed**. 
2. Process each instruction of each block: instructions marked as **primal** are moved to the corresponding **primal block**, instructions marked **differential** are moved to the corresponding **differential block**.
3. Instructions marked **mixed** need op-specific handling, and so are dispatched to the appropriate splitting function. For instance, block parameters that are holding differential-pair values are split into parameters for holding primal and differential values (the exception is function parameters, which are not affected). Similarly, `IRVar`s, `IRTerminatorInst`s (control-flow) and `IRCall`s are all split into multiple insts.
4. Except for `IRReturn`, all other control-flow insts are effectively duplicated so that the control-flow between the primal blocks and differential blocks both follow the original blocks' control-flow. The main difference is that PHI arguments are split (primal blocks carry primal values in their PHI arguments, and differential blocks carry diff values) between the two. Note that condition values (i.e. booleans) are used by both the primal and differential control-flow insts. However, since booleans are always primal values, they are always defined in the primal blocks.


**Block-Tagging:** Blocks are now tagged primal or differential depending on whether they are holding primal or differential insts. This is important for the next step (transposition) to figure out which blocks need to be transposed.

**Out-of-Scope Accesses:** After unzipping, the resulting IR is often **not valid**. If the control-flow is straight line (i.e. no branching or loops), the resulting IR is valid. However, if there is control-flow, then instructions can use operands whose definition does not dominate the use. This invalid IR is currently allowed to persist until the end of the auto-diff passes, when the checkpointing step occurs (i.e. Running IR validation will fail in between these steps)


Here is an example of unzipped code:

```Rust
OpModule
{
    // Generated function type
    ...

    // Unzipped code
    ...
        // The first block of a function is still mixed differential, and exclusively holds 
        // function parameter definitions (no other instructions)
        // 
        [OpMixedDifferentialDecoration]
        {
            [OpMixedDifferentialDecoration]
            %dpa = OpParam : %dpfloat
            [OpMixedDifferentialDecoration]
            %dpb = OpParam : %dpfloat
        }

        // Primal version of b containing only primal instructions
        [OpPrimalInstDecoration]
        %b_primal = OpBlock
        {
            [OpPrimalInstDecoration]
            %a_primal = OpDifferentialPairGetPrimal %dpa : %dpfloat 
            [OpPrimalInstDecorarion]
            %b_primal = OpDifferentialPairGetPrimal %dpa : %dpfloat 

            [OpPrimalInstDecoration]
            %1_primal = OpAdd %a_primal %b_primal : %float

            [OpPrimalInstDecoration]
            %2_primal = OpAdd %1_primal %1_primal : %float

            [OpBackwardDerivativePrimalReturnDecoration %2_primal]
            OpUnconditionalBranch %b_diff
        }

        // Differential version of b containing only differential instructions
        // with some exceptions. 
        // 
        [OpDifferentialInstDecoration]
        %b_diff = OpBlock
        {
            [OpDifferentialInstDecoration]
            %a_diff = OpDifferentialPairGetDifferential %dpa : %dpfloat 
            [OpDifferentialInstDecorarion]
            %b_diff = OpDifferentialPairGetDifferential %dpa : %dpfloat 

            [OpDifferentialInstDecoration]
            %1_diff = OpAdd %a_diff %b_diff : %float

            [OpDifferentialInstDecoration]
            %2_diff = OpAdd %1_diff %1_diff : %float

            // Return both the primal and differential
            [OpMixedDifferentialInstDecoration]
            %2_pair = OpDifferentialPairMakePair %2_primal %2_diff : %dpfloat

            [OpDifferentialInstDecoration]
            OpReturn %2_pair
        }

    ...
}
```

### 4. Transposition

The next step involves converting each differential instruction into its transpose. Effectively, we are re-writing each forward-mode derivative into its reverse-mode equivalent.

Recall from auto-diff [basics](./basics.md), that both the forward and reverse mode derivatives can be derived from the Jacobian matrix of any operation. The main difference is whether we multiply the derivatives of the inputs with the Jacobian or multiply the Jacobian with the derivatives w.r.t the outputs. These two operations are the transpose of each other, in that the reverse-mode derivative can be thought of as multiplying with the transpose of the Jacobian.

We perform this transposition on a per-instruction level.

Here is an example of a transposition of a multiplication operation:
```Rust
[OpPrimalInstDecoration]
%b = OpLoad %var_b // %b is a primal value

[OpDifferentialInstDecoration]
%da = OpLoad %var_da // %da is a differential value

// The operation we want to transpose
[OpDifferentialInstDecoration]
%1d = OpMul %da %b : %float

[OpDifferentialInstDecoration]
OpStore %1d %var_result
```

This multiplication can be represented as a tiny matrix multiplication between a singleton vector `[%da]` and singleton matrix `[%b]`. 
It's transpose will be the multiplication of the transpose of that matrix (which is the value itself `[%b]`) with a derivative w.r.t its output `%1d`, i.e. it becomes `%da = OpMul %1d %b`. Note that we now have to provide `%1d` as an **input**, and receive `da` was an output.

The resulting code is then:
```Rust
[OpPrimalInstDecoration]
%b = OpLoad %var_b : %float // primal values are unaffected (at this stage, they are in primal blocks)

// Reverse-mode code: (_rev) appended to all variables & insts to keep them distinct from the fwd-mode code.
[OpDifferentialInstDecoration]
%1d_rev = OpLoad %var_result_rev : %float

// The operation we want to transpose
[OpDifferentialInstDecoration]
%da_rev = OpMul %1d_rev %b : %float

[OpDifferentialInstDecoration]
OpStore %da_rev %var_da_rev
```

Notice that the three differential instructions are effectively run backwards **and** transposed. Loads become stores, 
the `OpMul` is transposed into another `OpMul`, and stores become loads. This backwards transposition is because the differential outputs become differential inputs, and thus, we need to process the future instructions first so that the new operands are defined before bring used for the new instruction.

This reverse order of operations also applies to control-flow. The rule of thumb is: if the forward-mode pass takes a particular path through the code, for a given set of primal values, the reverse-mode must "re-trace" the same path through the code, but in reverse by starting at the end.

We synthesize a CFG that satisfies this property through the following steps:
1. Clone the provided unzipped forward-mode function (and all blocks + instructions) to serve as the reverse-mode function.
2. Remove all **differential** blocks and create a set of corresponding reverse-mode blocks for each **differential** block removed (**primal** blocks are simply left alone), while holding a map between corresponding blocks. Initially, they are empty. 
3. Using the provided unzipped forward-mode function as a reference, process each differential block by walking each instruction from the _last_ (terminator) inst, and dispatching to the appropriate op-specific `transposeXYZ()` method to emit the appropriate transposed instructions into the corresponding reverse-mode block. 

    There are several concerns that must be taken care of:
    1. **Multiple Derivaive Outputs:** Unlike forward-mode auto-diff, where an inst producing a single value, would only need a single derivative (corresponding to that value), reverse-mode auto-diff can produce multiple derivatives from an inst. For instance `%dc = IRAdd(%da, %db)` produces two derivatives: `%da_rev = %dc_rev` and `%db_rev = %dc_rev`. Thus, the `transposeXYZ()` implementation for any instruction can return a set of derivative insts for each relevant input differential value.

    2. **Insts Used in Multiple Places (Derivative Accumulation):** If an inst is used in multiple places, and receives a reverse-mode derivative from several of those places, these results need to be **added up** to get the correct derivative. 
    
        Consider this forward-mode example

        ```Rust
        [OpDifferentialInstDecoration]
        %db = OpAdd %da, %da : %float

        [OpDifferentialInstDecoration]
        %dc = OpAdd %db, %da : %float
        ```

        It's reverse-mode derivative will look like this:

        ```Rust
        %db_rev = %dc_rev // %db only has one differential since it only consumed in one place.

        // reverse-mode differential for %da from trnaposing the first instruction
        [OpDifferentialInstDecoration]
        %da_rev_1 = OpAdd %db_rev, %db_rev : %float

        // reverse-mode differential for %da from transposing the second instruction
        [OpDifferentialInstDecoration]
        %da_rev_2 = %dc_rev

        // add them together to get the final derivative for %da
        [OpDifferentialInstDecoration]
        %da_rev = OpAdd %da_rev_1 %da_rev_2 : %float
        ```

        Derivative accumulation is achieved through two ways:
        
        **Within** a block, we keep a list all the reverse derivative insts for each inst and only **materialize** the total derivative when it is required as an operand. This is the most efficient way to do this, because we can apply certain optimizations for composite types (derivative of an array element, vector element, struct field, etc..).
        
        **Across** blocks, we use an accumulator variable that is inserted into a top-level block in the function, and add to this variable whenever a transposition operation generates a new inst. This can sometimes produce sub-optimal code for aggregate/large data types, but at the moment, the accumulator method is necessary because insts can receive derivatives from conditionally executed blocks.

        While this example uses `OpAdd` to demonstrate accumulation, in practice, we use the derivative type system (See [Types](./types.md) for more) to look up the derivative addition function (`dadd`) to add two values of an arbitrary differential type. In practice, the `OpAdd` is replaced by `OpCall %float_dadd %da_rev1 %da_rev_2`. Similarly, for accumulator variables, we must initialize them to zero for the accumulation to work correctly, and we lookup the `dzero` interface method to initialize it in a type-specific way.

    3. **Deferred Materialization for Derivatives of Composite Types:**
        Non-primitive types, such as vectors, arrays, structs, etc. whose elements are used in several places in the forward-mode code, can result in sub-optimal reverse-mode code. Here is an example (in Slang source-style):
        ```C
        float f_fwd(DifferentialPair<float3> input)
        {
            float3 dinput = input.getDifferential();
            float a = dinput.x + dinput.y;
            float b = a + dinput.z;

            return b;
        }

        // Transposed code (naively, without deferred materialization)
        void f_rev(inout DifferentialPair<float3> input, float d_output)
        {
            // transpose of (return b;)
            float db_rev = d_output;
            
            // transpose of (float b = a + dinput.z)
            float da_rev = db_rev;
            float3 dinput_rev_1 = float3(0.f, 0.f, da_rev);

            // transpose of (float a = dinput.x + dinput.y)
            float3 dinput_rev_2 = float3(0.f, da_rev, 0.f);
            float3 dinput_rev_3 = float3(da_rev, 0.f, 0.f);

            // Accumulate [dinput_rev_1, dinput_rev_2, dinput_rev_3]
            float3 dinput = dinput_rev_1 + dinput_rev_2 + dinput_rev_3

            input = DifferentialPair<float3>(
                input.getPrimal(),
                dinput);
        }
        ```

        Note that, this approach to inst-by-inst transposition can use a lot more stack space than is necessary (`dinput_rev_1`, `dinput_rev_2` and `dinput_rev_3` all only have a single non-0 entry). This is a known complexity issue with naive inst-by-inst transposition: hypothetically, an size-$N$ vector/array would end up allocating $O(N^2)$ memory even if only $N$ elements are non-0. 
        In our Slang implementation, we circumvent this (to an extent) by deferring materialization. Rather than create each component `dinput_rev_i` as soon as we see an inst use, we hold the derivative with a special flavor value for lookups (say `Swizzle` or `GetElement`). When the total value `dinput_rev` is necessary, we process components of each flavor type at once and create a single derivative from all the components. 

        Here is the same example, with deferred materialization:
        ```C
        // Transposed code (naively, without deferred materialization)
        void f_rev(inout DifferentialPair<float3> input, float d_output)
        {
            // transpose of (return b;)
            float db_rev = d_output;
            
            // transpose of (float b = a + dinput.z), hold {flavor=Swizzle, component=.z, derivInst=db_rev} in list.
            float da_rev = db_rev;

            // transpose of (float a = dinput.x + dinput.y), 
            // hold {flavor=Swizzle, component=.x, derivInst=da_rev} and {flavor=Swizzle, component=.y, derivInst=da_rev} in list.

            // Materialize when required (for constructing return pair)
            float3 dinput = float3(db_rev, da_rev, da_rev);

            input = DifferentialPair<float3>(
                input.getPrimal(),
                dinput);
        }
        ```

        Note that this only really works for accumulation *within* a single block/control-flow region. For across regions, we still have to materialize when we exit a region, so this memory problem can still manifest for control-flow heavy functions, where each region must allocate enough space for its contribution to the full derivative, even if only a small subset is non-0.



```C
float a[10] = /*...*/;
for (int i = 0; i < 10; i++)
{
    a[i] = f(a[i]);
}
```

```C

// Entry block
%t = OpBlock
{
    IRLoop %c %br %c 0
}

// Condition
%c = OpBlock
{
    %i = OpParam : %float
    %a = OpParam : %Array(%float, 10)
    
    %2 = OpLesser(%i, 10) : %bool

    %OpIfElse(%2, %b, %br, %br)
}

// Loop body.
%b = OpBlock 
{
    %a_i = OpGetElement(%a, %i) : %float
    %f_a_i = OpCall(f, %a_i) : %float

    %a_next = OpUpdateElement(%a, %i, %f_a_i) : %Array(%float, 10)

    %i_next = OpAdd(%i, 1)

    OpUnconditionalBranch(%c, %i_next, %a_next)
}

// Break block
%br = OpBlock
{
    //...
}
```

After AD passes, this results in the following code:
```C

//// Primal context pass.

// Entry block
%t_rev = OpBlock
{
    // Context storage for all loop phi variables (n_iters + 1)
    %ctx_a = IRVar : %array(%array(%float, 10), 11) // Catastrophically large amount of storage.
    %ctx_i = IRVar : %array(%float, 11)

    OpLoop %c %br %c 0
}

// Condition
%c_rev = OpBlock
{
    %i = OpParam : %float
    %a = OpParam : %array(%float, 10)

    // Context store operations.
    %ctx_i_ptr = OpGetElementPtr(%ctx_i, %i) : %ptr(%int)
    OpStore(%ctx_i_ptr, %i)
    %ctx_a_ptr = OpGetElementPtr(%ctx_a, %i) : %ptr(%array(%float, 10))
    OpStore(%ctx_a_ptr, %a)
    
    %2 = OpLesser(%i, 10) : %bool

    %OpIfElse(%2, %b, %br, %br)
}

// Loop body.
%b = OpBlock 
{ /*...*/ }

// Break block
%br = OpBlock
{ /*...*/ }

//// Backprop pass

// Entry block
%t_rev = OpBlock
{
    // Count down from the end
    OpLoop %c_rev %br_rev %c_rev 9 

    // Variable to hold the derivative of %a
    %var_da_rev = OpVar : %ptr(%array(%float, 10))
}

// Condition
%c_rev = OpBlock
{
    // rev-mode loop counter (runs backwards from limit to 0)
    %dc = OpParam : %int
    
    %2 = OpLesser(%i, 10) : %bool

    OpIfElse %2 %b %br %br
}

// Loop body.
%b_rev = OpBlock 
{
    // Context load operations.
    %ctx_i_ptr = OpGetElementPtr(%ctx_i, %dc) : %ptr(%int)
    %i_saved = OpLoad(%ctx_i_ptr) : %int

    %ctx_a_ptr = OpGetElementPtr(%ctx_a, %dc) : %ptr(%array(%float, 10))
    %a_saved = OpLoad(%ctx_a_ptr) : %array(%float, 10)

    %a_i = OpGetElement(%a_saved, %i_saved) : %float
    %a_pair_i = OpMakeDifferentialPair(%a_i, 0) : %diff_pair(%float)

    %da_rev_ptr = OpGetElementPtr(%var_da_rev, %i_saved) : %ptr(%float)
    %df_output = OpLoad(%da_rev_ptr) : %float

    // Call rev-mode of f to propagate derivative of output of f to input of f. (Assume f has no context requirement)
    %var_a_pair_i = OpVar : %ptr(%diff_pair(%float))
    OpStore(%var_a_pair_i, %a_pair_i)
    OpCall(f_rev, %a_pair_i, %df_output) : %float 

    // Load derivative for a_i
    %a_pair_i_loaded = OpLoad(%var_a_pair_i, %a_pair_i)
    %da_rev_i = OpDifferentialPairGetDifferential(%a_pair_i_loaded) : %float

    // Create derivative array for backpropagation (this happens during gradient materialization)
    %da_rev_local_var = OpVar : %ptr(%array(%float, 10))
    %da_rev_init_zero = OpMakeArray(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) : %array(%float, 10)
    OpStore(%da_rev_local_var, %da_rev_init_zero)

    %da_rev_var_i = OpGetElementPtr(%da_rev_local_var, %dc) : %ptr(%float)
    %curr_dval = OpLoad(%da_rev_var_i) : %float
    %acc_dval = OpAdd(%curr_dval, %da_rev_i) : %float
    OpStore(%da_rev_var_i, %acc_dval)

    // Add derivative array to the global var.
    %curr_dval_a = OpLoad(%var_da_rev) : %array(%float, 10)
    %new_dval_a = OpLoad(%da_rev_local_var) : %array(%float, 10)
    %acc_dval_a = OpCall('array_dadd', %curr_dval_a, %new_dval_a) : %array(%float, 10)
    OpStore(%var_da_rev, %acc_dval_a)

    %dc_next = OpAdd(%dc, -1)

    OpUnconditionalBranch(%c_rev, %dc_next)
}

// Break block
%br_rev = OpBlock
{ /*...*/ }
```

4. Construct the reverse control-flow (`reveseCFGRegion()`) by going through the reference forward-mode blocks, and cloning the control-flow onto the reverse-mode blocks, but in reverse. This is achieved by running `reverseCFGRegion()` recursively on each sub-region, where a *region* is defined as a set of blocks with a single entry block and a single exit block. This definition of a region only works because we normalized the CFG into this form.

    The reversal logic follows these general rules:
    1. **Unconditional Branch**: For an unconditional branch from `A->B` we simply have to map the reverse version of B with that of A. i.e. `rev[B] -> rev[A]`
    2. **If-Else**: For an if-else of the form `A->[true = T->...->T_last->M, false = F->...->F_last->M]`, we construct `rev[M]->[true = rev[T_last]->...->rev[T_last]->rev[A], false = rev[F_last]->...->rev[F]->rev[A]]`. That is, we reverse each sub-region, and start from the merge block and end at the split block.
    Note that we need to identify `T_last` and `F_last` i.e. the last two blocks in the true and false regions. We make the last block in the region an additional return value of `reverseCFGRegion()`, so that when reversing the true and false sub-regions, we also get the relevant last block as an additional output. Also note that additional empty blocks may be inserted to carry derivatives of the phi arguments, but this does not alter the control-flow.
    3. **Switch-case**: Proceeds in exactly the same way as `if-else` reversal, but with multiple cases instead of just 2.
    4. **Loop**: After normalization, all (non-trivial) loops are of the form: `A->C->[true = T->...->T_last->C, false=B->...->M]`. We reverse this loop into `rev[M]->...rev[B]->rev[C]->[true=rev[T_last]->...->rev[T]->rev[C], false=rev[A]]`. The actual reversal logic also handles some corner cases by inserting additional blank blocks to avoid situations where regions may share the same merge block.

    Finally, we process the first and last blocks (entry and return blocks) by inserting a void return (reverse-mode derivative functions are always of void result type)

At this stage, the reverse-mode generation is almost complete. The control-flow and the derivative logic is present, but we still have to resolve out-of-scope accesses from the new differential blocks into the primal block.

### 5. Checkpointing/Recomputation (also called 'primal-hoisting')
This step legalizes the out-of-scope accesses of primal insts from within differential blocks. This is to prepare us for the next step (i.e. [extraction](#6-extraction)) that splits the function into two by moving the primal blocks into a separate primal-context-generator function, and the differential blocks into the backward-propagation function. 

Before we can perform this extraction, we must find any primal values being used in differential blocks and handle them in one of two ways:
**Store** (put the values in a static struct) or **Recompute** (clone the necessary instructions to recompute when necessary). We first _classify_ all necessary instructions into one of the two buckets before processing each use accordingly.

1. **Classify uses into each set:** Note that rather than proceeding on an inst-by-inst basis, we classify **uses** of insts. The same inst can be used in several places, and we may decide to store one use and recompute another (in some cases, this could be the optimal result). 
The classification process uses a work-list approach that roughly looks like the following:
    1. Add all uses of **primal** insts in an inst within a **differential** block to the work list. This is our initial set of uses that require classification. 
    2. Query the active policy object (which for now is hardcoded) to obtain the classification based on heuristics & user decorations (Specifically `[PreferRecompute]` and `[PreferCheckpoint]` decorations influence the classification policy)
    3. For uses that should be **recomputed**, we have to now make the same decision one their **operands**, in order to make them available for the recomputation insts. Thus, their operands are added to the work list.
    4. For uses that should be **stored**, there is no need to consider their operands, since the computed value will be explicitly stored and loaded later.
    5. Once the worklist is empty, go over all the **uses** and their classifications, and convert them into a list of **insts** that should be stored or recomputed. Note that if an inst has uses with both classifications, then it can appear in both lists.

2. **Process 'Store' (i.e. checkpoint) insts:** Store them into a single variable (of a struct type that is synthesized as necessary), and then loaded from in the differential blocks. This allows us to simply turn this variable into an output parameter from the context function and an input parameter for the backprop function.
When storing values this way, we must consider that instructions within loops can have different values each iteration. Thus, we must use an array to store each value, and this array's size must be statically known since we wish to synthesize a static struct type to hold all the stored values. Thus, we enforce the requirement of a `[MaxIters(N)]` decoration and attempt to infer a loop iteration limit if one is not provided.

    Here's an example of a case where we decide to checkpoint _all_ relevant uses:

    ```C
    // Example function without loops post-transposition step (BEFORE hoisting)
    void f_rev(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = 0;

        if (x < 0.5)
        {
            float t1 = x * x;
            p = t1 * t1 + x;
        }

        if (x > 10.f)
        {
            float t2 = x * x * x;
            p = t2 * t2 + x;
        }

        //
        // Reversed differential blocks start here (will be extracted into a separate function in Step 6: Extraction)
        //

        float dp_rev = d_out;
        float dx_rev = 0.f; // accumulator var for 'x.d'
        if (x > 10.f)
        {
            float dt2_rev = t2 * dp_rev; // access of a primal value 't2' from a differential block.
            dx_rev += dp_rev;
            dp_rev = 0.f; // dp_rev's value gets reset to 0 after use.

            dx_rev += x * x * dt2_rev;
            dx_rev += x * dt2_rev * x;
            dx_rev += dt2_rev * x * x;
        }

        if (x < 0.5)
        {
            float dt1_rev = t1 * dp_rev; // access of a primal value 't1' from a differential block.
            dx_rev += dp_rev;

            dx_rev += x * dt1_rev;
            dx_rev += dt1_rev * x;
        }

        dpx = DifferentialPair<float>(x, dx_rev);
    }

    // The same function after the primal hoisting's checkpointing step. In this example, we
    // assume all relevant uses are being checkpointed.
    // 
    void f_rev_hoisted(DifferentialPair<float> dpx, float d_out)
    {
        // Insert vars for checkpointed insts at the top-level
        float t1_storage;
        float t2_storage;

        //
        // Primal blocks
        //

        float x = dpx.getPrimal();
        float p = 0;

        if (x < 0.5)
        {
            float t1 = x * x;
            t1_storage = t1; // Cache values immediately after they are created.
            p = t1 * t1 + x;
        }

        if (x > 10.f)
        {
            float t2 = x * x * x;
            t2_storage = t2; // Cache values immediately after they are created.
            p = t2 * t2 + x;
        }

        //
        // Reversed differential blocks
        //

        float x = dpx.getPrimal();

        float dp_rev = d_out;
        float dx_rev = 0.f; // accumulator var for 'x.d'
        if (x > 10.f)
        {
            float dt2_rev = t2_storage * dp_rev; // Use stored value.
            dx_rev += dp_rev;

            dx_rev += x * x * dt2_rev;
            dx_rev += x * dt2_rev * x;
            dx_rev += dt2_rev * x * x;
        }

        if (x < 0.5)
        {
            float dt1_rev = t1_storage * dp_rev; // Use stored value.
            dx_rev += dp_rev;

            dx_rev += x * dt1_rev;
            dx_rev += dt1_rev * x;
        }

        dpx = DifferentialPair<float>(x, dx_rev);
    }
    ```
    Another example with a function `g` that does contain loops:

    ```C
    // Example function with a loop, post-transposition step (BEFORE hoisting)
    void g_rev(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = 0;

        for (uint i = 0; i < 10; i++)
        {
            p = x * p;
        }

        //
        // Reversed differential blocks
        //

        float dx_rev = 0.f;
        float dp_rev = d_out;
        for (uint i = 9; i > 0; i--)
        {
            dx_rev += p * dp_rev; // primal value 'p' accessed from differential blocks
            dp_rev = x * dp_rev;
        }

        return DifferentialPair<float>(x, dx_rev);
    }

    // After hoisting, note that we checkpoint 'p' in this case by using an array.
    void g_rev_hoisted(DifferentialPair<float> dpx, float d_out)
    {
        // Insert array to hold states of 'p'
        float p_storage[11];

        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = 0;

        // Insert storage for all states of p, including the initial value upon loop entry
        p_storage[0] = p;
        for (uint i = 0; i < 10; i++)
        {
            p = x * p;
            // Use the loop induction variable 'i' to figure out which index to store p in.
            p_storage[i+1] = p;
        }

        //
        // Reversed differential blocks
        //

        float dx_rev = 0.f;
        float dp_rev = d_out;
        for (uint i = 9; i >= 0; i--)
        {
            // Load appropriate value of p from storage
            float p = p_storage[i];
            dx_rev += p * dp_rev; 
            dp_rev = x * dp_rev;
        }

        return DifferentialPair<float>(x, dx_rev);
    }
    ```

    **Indexed Region Processing:** In order to be able to allocate the right array and use the right indices, we need information about which blocks are part of which loop (and loops can be nested, so blocks can be part of multiple loops). To do this, we run a pre-processing step that maps all blocks to all relevant loop regions, the corresponding index variables and the inferred iteration limits (maximum times a loop can run). Note that if an instruction appears in a nested block, we create a multi-dimensional array and use multiple indices.

    **Loop State Variables:** Certain variables cannot be classified as recompute. Major examples are loop state variables which are defined as variables that are read from and written to within the loop. In practice, they appear as phi-variables on the first loop block after SSA simplification. Their uses _must_ be classified as 'store', because recomputing them requires duplicating the primal loop within the differential loop. This is because the differential loop runs backwards so the state of a primal variable at loop index $N$ cannot be recomputed when the loop is running backwards ($N+1 \to N \to N-1$), and involves running the primal loop up to $N$ times within the current iteration of the differential loop. In terms of complexity, this turns an $O(N)$ loop into an $O(N^2)$ loop, and so we disallow this.
    It is possible that the resulting $O(N^2)$ loop may end up being faster in practice due to reduced memory requirements, but we currently lack the infrastructure to robustly allow such loop duplication while keeping the user informed of the potentially drastic complexity issues.

3. **Process 'Recompute' insts:** Insert a copy of the primal instruction into a corresponding 'recomputation' block that is inserted into the differential control-flow so that it dominates the use-site. 

    **Insertion of Recompute Blocks:** In order to accommodate recomputation, we first preprocess the function, by going through each **breakable (i.e. loop) region** in the differential blocks, looking up the corresponding **primal region** and cloning all the primal blocks into the beginning of the differential region. Note that this cloning process does not actually clone the instructions within each block, only the control-flow (i.e. terminator) insts. This way, there is a 1:1 mapping between the primal blocks and the newly created **recompute blocks**, This way, if we decide to 'recompute' an instruction, we can simply clone it into the corresponding recompute block, and we have a guarantee that the definition and use-site are within the same loop scope, and that the definition comes before the use.
    
    **Legalizing Accesses from Branches:** Our per-loop-region recompute blocks ensure that the recomputed inst is always within the same region as its uses, but it can still be out-of-scope if it is defined within a branch (i.e. if-else). We therefore still run a light-weight hoisting pass that detects these uses, inserts an `IRVar` at the immediate dominator of the def and use, and inserts loads and stores accordingly. Since they occur within the same loop region, there is no need to worry about arrays/indices (unlike the 'store' case).
    
    **Marking Recompute Blocks:** These blocks are marked with `OpRecomputeBlockDecoration` to identify them as containing primal instructions, even though they are within differential regions. This helps us remove any unused blocks if none of the instructions end up being recomputed.
    
    Here is an example of recomputation demonstrated in Slang source-style (although this takes place in IR-form)
    ```C
    // Example function without loops post-transposition step. 
    void f_rev(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = 0;

        if (x < 0.5)
        {
            float t1 = x * x;
            p = t1 * t1 + x;
        }

        if (x > 10.f)
        {
            float t2 = x * x * x;
            p = t2 * t2 + x;
        }

        //
        // Reversed differential blocks start here (will be extracted into a separate function in Step 6: Extraction)
        //

        float dp_rev = d_out;
        float dx_rev = 0.f; // accumulator var for 'x.d'
        if (x > 10.f)
        {
            float dt2_rev = t2 * dp_rev; // access of a primal value 't2' from a differential block.
            dx_rev += dp_rev;
            dp_rev = 0.f; // dp_rev's value gets reset to 0 after use.

            dx_rev += x * x * dt2_rev;
            dx_rev += x * dt2_rev * x;
            dx_rev += dt2_rev * x * x;
        }

        if (x < 0.5)
        {
            float dt1_rev = t1 * dp_rev; // access of a primal value 't1' from a differential block.
            dx_rev += dp_rev;

            dx_rev += x * dt1_rev;
            dx_rev += dt1_rev * x;
        }

        dpx = DifferentialPair<float>(x, dx_rev);
    }

    // The same function after the primal hoisting step. Note that the primal control flow has been cloned into the start of
    // the top-level differential region.
    // 
    void f_rev_hoisted(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = 0;

        if (x < 0.5)
        {
            float t1 = x * x;
            p = t1 * t1 + x;
        }

        if (x > 10.f)
        {
            float t2 = x * x * x;
            p = t2 * t2 + x;
        }

        //
        // Reversed differential blocks start here (will be extracted into a separate function in Step 6: Extraction)
        //

        // Recompute blocks are inserted at the beginning of each differential region.
        float x_recompute = dpx.getPrimal();
        if (x_recompute < 0.5)
        {
            // Only the t1 instruction is cloned in since it is used by the differential blocks.
            float t1_recompute = x_recompute * x_recompute;
        }

        if (x_recompute > 10.f)
        {
            // Only the t2 instruction is cloned in since it is used by the differential blocks.
            float t2_recompute = x_recompute * x_recompute * x_recompute;
        }

        float dp_rev = d_out;
        float dx_rev = 0.f; // accumulator var for 'x.d'
        if (x_recompute > 10.f)
        {
            float dt2_rev = t2_recompute * dp_rev; // invalid access of 't2_recompute' (it's inside a branch)
            dx_rev += dp_rev;

            dx_rev += x_recompute * x_recompute * dt2_rev;
            dx_rev += x_recompute * dt2_rev * x_recompute;
            dx_rev += dt2_rev * x_recompute * x_recompute;
        }

        if (x < 0.5)
        {
            float dt1_rev = t1 * dp_rev; // invalid access of 't1_recompute' (it's inside a branch)
            dx_rev += dp_rev;

            dx_rev += x_recompute * dt1_rev;
            dx_rev += dt1_rev * x_recompute;
        }

        dpx = DifferentialPair<float>(x, dx_rev);
    }

    // Same function after branch-access-legalization (run after the primal-hoisting step):
    void f_rev_hoisted_and_legalized(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks:
        //

        float x = dpx.getPrimal();
        float p = 0;

        float t1; // Var inserted/moved to immediate dominator block (branch-access-legalization)
        if (x < 0.5)
        {
            t1 = x * x;
            p = t1 * t1 + x;
        }

        float t2; // Var inserted/moved to immediate dominator block (branch-access-legalization)
        if (x > 10.f)
        {
            t2 = x * x * x;
            p = t2 * t2 + x;
        }

        //
        // Reversed differential blocks:
        //

        float dp_rev = d_out;
        float dx_rev = 0.f; // accumulator var for 'x.d'
        if (x > 10.f)
        {
            float dt2_rev = t2 * dp_rev;
            dx_rev += dp_rev;

            dx_rev += x * x * dt2_rev;
            dx_rev += x * dt2_rev * x;
            dx_rev += dt2_rev * x * x;
        }

        if (x < 0.5)
        {
            float dt1_rev = t1 * dp_rev;
            dx_rev += dp_rev;

            dx_rev += x * dt1_rev;
            dx_rev += dt1_rev * x;
        }
    }
    ```

    For completeness, here is another example of a function `g` which contains a loop to demonstrate how recomputation works when there are
    multiple loop regions.

    ```C
    // Example function with a loop, post-transposition step (BEFORE hoisting)
    void g_rev(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = x;

        if (x < 0.5)
        {
            float k = 2.f * x;
            p = p * k;
        }

        for (uint i = 0; i < 10; i++)
        {
            if (x > 0.5)
            {
                float t = 2.f * i;
                p = p + x * t;
            }
        }

        //
        // Reversed differential blocks
        //

        float dt_rev = 0.f;
        float dp_rev = 0.f;
        for (uint i = 9; i >= 0; i++)
        {
            if (x > 0.5)
            {
                dx_rev += t * dp_rev; // Use of primal value 't' in differential blocks.
            }
        }

        if (x < 0.5)
        {
            dp_rev = dp_rev * k;       // Use of primal value 'k' in differential blocks.
            float dk_rev = p * dp_rev; // Use of primal value 'p' in differential blocks.
            dx_rev += dk_rev * 2.f;
        }

        dx_rev += dp_rev;

        return DifferentialPair<float>(x, dx_rev);
    }

    // The same function after hoisting and branch-access-legalization. 
    // Notice that recompute blocks are inserted into the top-level
    // as well as each loop region in the differential blocks.
    // 
    void g_rev_hoisted_and_legalized(DifferentialPair<float> dpx, float d_out)
    {
        //
        // Primal blocks (will be extracted into a separate function in Step 6: Extraction)
        //

        float x = dpx.getPrimal();
        float p = x;

        if (x < 0.5)
        {
            float k = 2.f * x;
            p = p * k;
        }

        for (uint i = 0; i < 10; i++)
        {
            if (x > 0.5)
            {
                float t = 2.f * i;
                p = p + x * t;
            }
        }

        //
        // Reversed differential blocks
        //

        // ----- Recompute blocks inserted for top-level
        float p_recompute = x; // Inst recomputed.
        float k_recompute; 
        if (x < 0.5)
        {
            k_recompute = 2.f * x; // Inst recomputed.
        }
        // -----

        float dt_rev = 0.f;
        float dp_rev = 0.f;
        for (uint i = 9; i >= 0; i++)
        {
            // ---- Recompute blocks inserted for loop region.
            float t_recompute;
            if (x > 0.5)
            {
                t_recompute = 2.f * i; // Inst recomputed.
            }
            // ----

            if (x > 0.5)
            {
                dx_rev += t_recompute * dp_rev; 
            }
        }

        if (x < 0.5)
        {
            dp_rev = dp_rev * k_recompute;       
            float dk_rev = p_recompute * dp_rev; 
            dx_rev += dk_rev * 2.f;
        }

        dx_rev += dp_rev;

        return DifferentialPair<float>(x, dx_rev);
    }
    ```

### 6. Extraction 
The final step involves _splitting_ the function immediately after the primal block to create two functions: a **primal context function** that computes the primal value normally, but also outputs a context object with relevant intermediate values, and a **backward propagation function** that computes the backward derivative and consumes this context object for the required intermediate values.

The first 5 steps have set us up for this final step, so it is not particularly complex. We follow this high-level logic:

1. Create an empty function for the primal context function. The type of this function is the same as the primal function, but with an additional `out` parameter for the intermediate context, whose type is undecided at this stage. We use a temporary function-specific type called `OpBackwardDerivativeIntermediateContextType(func)` as a placeholder. 
2. Move primal blocks to the primal context function. Re-create the return inst (the return value is temporarily remembered using a decoration during the rest of the AD process). Also, the first block (reserved for function parameters) is also duplicated and processed to have primal parameters in the primal function and pair parameters in the differential function.
3. Lower all `OpBackwardDerivativeIntermediateContextType` types into concrete struct types by creating a field for each 'stored' inst from Step 5. This lowering process happens **at the end of the current AD pass after all relevant methods have completed Step 5**. We need Step 5 (hoisting) to be complete for all relevant methods because the context struct for a given function can include context structs of other functions that are called from it. Our context-type lowering therefore proceeds recursively by lowering the context for inner functions as necessary. The lowering process also removes the temporary vars that were created to hold the store insts, and replaces them with a stores and loads from the context struct.
   
   **Recursive Functions are Disallowed:** Since we lower all intermediate types into a static struct type, recursive calls cannot currently be supported from differentiable functions. The context struct for a method may include itself, creating an impossible scenario.

Here is one of the examples above (`g`) after checkpointing:

```C
// Example function before the extraction step.
void f_rev_hoisted(DifferentialPair<float> dpx, float d_out)
{
    // Insert vars for checkpointed insts at the top-level
    float t1_storage;
    float t2_storage;

    //
    // Primal blocks
    //

    float x = dpx.getPrimal();
    float p = 0;

    if (x < 0.5)
    {
        float t1 = x * x;
        t1_storage = t1; // Cache values immediately after they are created.
        p = t1 * t1 + x;
    }

    if (x > 10.f)
    {
        float t2 = x * x * x;
        t2_storage = t2; // Cache values immediately after they are created.
        p = t2 * t2 + x;
    }

    //
    // Reversed differential blocks
    //

    float x = dpx.getPrimal();

    float dp_rev = d_out;
    float dx_rev = 0.f; // accumulator var for 'x.d'
    if (x > 10.f)
    {
        float dt2_rev = t2_storage * dp_rev; // Use stored value.
        dx_rev += dp_rev;

        dx_rev += x * x * dt2_rev;
        dx_rev += x * dt2_rev * x;
        dx_rev += dt2_rev * x * x;
    }

    if (x < 0.5)
    {
        float dt1_rev = t1_storage * dp_rev; // Use stored value.
        dx_rev += dp_rev;

        dx_rev += x * dt1_rev;
        dx_rev += dt1_rev * x;
    }

    dpx = DifferentialPair<float>(x, dx_rev);
}

// After extraction: lowered intermediate context for f
struct f_Intermediates
{
    float t1;
    float t2;
};


// After extraction: primal context function
float s_primal_ctx_f(float x, out f_Intermediates ctx)
{
    //
    // Primal blocks
    //

    float x = dpx.getPrimal();
    float p = 0;

    if (x < 0.5)
    {
        float t1 = x * x;
        ctx.t1 = t1; // Cache values immediately after they are created.
        p = t1 * t1 + x;
    }

    if (x > 10.f)
    {
        float t2 = x * x * x;
        ctx.t2 = t2; // Cache values immediately after they are created.
        p = t2 * t2 + x;
    }

    return p;
}

// After extraction: backward propagation function.
void s_bwd_f(DifferentialPair<float> dpx, float d_out, f_Intermediates ctx)
{
    float x = dpx.getPrimal();

    float dp_rev = d_out;
    float dx_rev = 0.f; // accumulator var for 'x.d'
    if (x > 10.f)
    {
        float dt2_rev = ctx.t2 * dp_rev; // Use stored value.
        dx_rev += dp_rev;

        dx_rev += x * x * dt2_rev;
        dx_rev += x * dt2_rev * x;
        dx_rev += dt2_rev * x * x;
    }

    if (x < 0.5)
    {
        float dt1_rev = ctx.t1 * dp_rev; // Use stored value.
        dx_rev += dp_rev;

        dx_rev += x * dt1_rev;
        dx_rev += dt1_rev * x;
    }

    dpx = DifferentialPair<float>(x, dx_rev);
}
```

Having separate methods for the primal and backward passes is necessary when reverse-mode differentiating a method that calls out to other differentiable functions.
Here is an example of differentiating a method that calls out to multiple methods, to get an idea for why we need the primal context method to be separate

```C
float outer(float x)
{
    float y = f(x);
    float z = g(y);
    float w = h(z);

    return w;
}

// It's complete reverse mode derivative looks like the following:
void outer_rev(DifferentialPair<float> dpx, float d_output)
{
    // Compute the primal values in the forward direction, while producing relevant context. 
    f_Intermediates f_ctx;
    g_Intermediates g_ctx;
    h_Intermediates h_ctx;

    float y = s_primal_ctx_f(x, f_ctx);
    float z = s_primal_ctx_g(y, g_ctx);
    float w = s_primal_ctx_h(z, h_ctx);

    // Note that at this point, we are holding intermediate context variables for f, g and h.

    // Consume the context while evaluating the propagating the derivatives backwards.
    DifferentialPair<float> dpz = {z, 0.f};
    s_bwd_h(dpz, d_output, h_ctx);

    DifferentialPair<float> dpy = {y, 0.f};
    s_bwd_g(dpy, dpz.getDifferential(), g_ctx);

    DifferentialPair<float> _dpx = {x, 0.f};
    s_bwd_f(dpx, dpy.getDifferential(), f_ctx);

    dpx = _dpx;
}
```
