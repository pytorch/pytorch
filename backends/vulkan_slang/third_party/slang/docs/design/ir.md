The Design of Slang's Intermediate Representation (IR)
======================================================

This document details some of the important design choices for Slang's IR.

Goals and Non-Goals
-------------------

The IR needs to balance many goals which can sometimes come into conflict.
We will start by enumerating these goals (and related non-goals) explicitly so that we can better motivate specific design choices.

* Obviously it must be simple to lower any source code in Slang code to the IR. It is however a non-goal for the lowering process to be lossless; we do not need to recover source-level program structure from the IR.

* The IR must be amenable to standard dataflow analyses and optimizations. It should be possible to read a paper on a compiler algorithm or technique and apply it to our IR in a straightforward manner, and with the expected asymptotic efficiency.

* As a particular case of analysis and optimization, it should be possible to validate flow-dependent properties of an input function/program (e.g., whether an `[unroll]` loop is actually unrollable) using the IR, and emit meaningful error messages that reference the AST-level names/locations of constructs involved in an error.

* It should be possible to compile modules to the IR separately and then "link" them in a way that depends only on IR-level (not AST-level) constructs. We want to allow changing implementation details of a module without forcing a re-compile of IR code using that module (what counts as "implementation details") is negotiable.

* There should be a way to serialize IR modules in a round-trip fashion preserving all of the structure. As a long-term goal, the serialized format should provide stability across compiler versions (working more as an IL than an IR)

* The IR must be able to encode "generic" (type-parameterized) constructs explicitly, and to express transformations from generic to specialized (or dynamic-dispatch) code in the IR. In particular, it must be possible for a module to make use of generic defined in another (separately-compiled) module, with validation performed before linking, and specialization performed after.

* The IR must be able to express code that is close to the level of abstraction of shader intermediate languages (ILs) like SPIR-V and DXIL, so that we can minimize the amount of work required (and the number of issues that can arise) when translating the IR to these targets. This can involve lowering and legalization passes to match the constraints of those ILs, but it should not require too much work to be done outside of the IR.

* It should be possible to translate code in the IR back into high-level-language code, including things like structured control-flow constructs.

* Whenever possible, invariants required by the IR should be built into its structure so that they are easier to maintain.

* We should strive to make the IR encoding, both in memory and when serialized, as compact as is practically possible.

Inspirations
------------

The IR design we currently use takes inspiration from three main sources:

* The LLVM project provides the basic inspiration for the approach to SSA, such as using a typed IR, the decision to use the same object to represent an instruction and the SSA value it produces, and the push to have an extremely simple `replaceAllUsesWith` primitive. It is easy to forget that it is possible to design a compiler with different design decisions; the LLVM ones just happen to both be well-motivated and well-known.

* The Swift IL (SIL) provides the inspiration for our approach for encoding SSA "phi nodes" (blocks with arguments), and also informs some of how we have approached encoding generics and related features like existential types.

* The SPIR-V IL provides the inspiration for the choice to uniformly represent types as instructions, for how to encode "join points" for structured control flow, and for the concept of "decorations" for encoding additional metadata on instructions.


Key Design Decisions
--------------------

### Everything is an Instruction

The Slang IR strives for an extremely high degree of uniformity, so almost every concept in the IR is ultimately just an instruction:

* Ordinary add/sub/mul/etc. operations are instructions, as are function calls, branches, function parameters, etc.

* Basic blocks in functions, as well as functions themselves are "parent instructions" that can have other instructions as children

* Constant values (e.g., even `true` and `false`) are instructions

* Types are instructions too, and can have operands (e.g., a vector type is the `VectorType` instruction applied to operands for the element type and count)

* Generics are encoded entirely using ordinary instructions: a generic is encoded like a function that just happens to do computation at the type level

* It isn't true right now, but eventually decorations will also be instructions, so that they can have operands like any other instruction

* An overall IR module is itself an instruction so that there is a single tree that owns everything

This uniformity greatly simplifies the task of supporting generics, and also means that operations that need to work over all instructions, such as cloning and serialization, can work with a single uniform representation and avoid special-casing particular opcodes.

The decision to use an extremely uniform design, even going as far to treat types as "ordinary" instructions, is similar to SPIR-V, although we do not enforce many of the constraints SPIR-V does on how type and value instructions can be mixed.

### Instructions Have a Uniform Structure

Every instruction has:

* An opcode
* A type (the top-level module is the only place where this can be null)
* Zero or more operands
* Zero or more decorations
* Zero or more children

Instructions are not allowed to have any semantically-relevant information that is not in the above list.
The only exception to this rule is instructions that represent literal constants, which store additional data to represent their value.

The in-memory encoding places a few more restrictions on top of this so that, e.g., currently an instruction can either have operands of children, but not both.

Because everything that could be used as an operand is also an instruction, the operands of an instruction are stored in a highly uniform way as a contiguous array of `IRUse` values (even the type is contiguous with this array, so that it can be treated as an additional operand when required).
The `IRUse` type maintains explicit links for use-def information, currently in a slightly bloated fashion (there are well-known techniques for reducing the size of this information).

### A Class Hierarchy Mirrored in Opcodes

There is a logical "class hierarchy" for instructions, and we support (but do not mandate) declaring a C++ `struct` type to expose an instruction or group of instructions.
These `struct` types can be helpful to encode the fact that the program knows an instruction must/should have a particular type (e.g., having a function parameter of type `IRFunction*` prevents users from accidentally passing in an arbitrary `IRInst*` without checking that it is a function first), and can also provide convenience accessors for operands/children.

Do make "dynamic cast" operations on this class hierarchy efficient, we arrange for the instruction opcodes for the in-memory IR to guarantee that all the descendents of a particular "base class" will occupy a contiguous range of opcodes. Checking that an instruction is in that range is then a constant-time operation that only looks at its opcode field.

There are some subtleties to how the opcodes are ordered to deal with the fact that some opcodes have a kind of "multiple inheritance" thing going on, but that is a design wart that we should probably remove over time, rather than something we are proud of.

### A Simpler Encoding of SSA

The traditional encoding of SSA form involves placing "phi" instructions at the start of blocks that represent control-flow join points where a variable will take on different values depending on the incoming edge that is taken.
There are of course benefits to sticking with tradition, but phi instructions also have a few downsides:

- The operands to phi instructions are the one case where the "def dominates use" constraint of SSA appears to be violated. I say "appears" because officially the action of a phi occurs on the incoming edge (not in the target block) and that edge will of course be dominated by the predecessor block. It still creates a special case that programmers need to be careful about. This also complicates serialization in that there is no order in which the blocks/instructions of a function can be emitted that guarantees that every instruction always precedes all of its uses in the stream.

- All of the phi instructions at the start of the block must effectively operate in parallel, so that they all "read" from the correct operand before "writing" to the target variable. Like the above special case, this is only a problem for a phi related to a loop back-edge. It is of course possible to always remember the special interpretation of phi instructions (that they don't actually execute sequentially like every other instruction in a block), but its another special case.

- The order of operands to a phi instruction needs to be related back to the predecessor blocks, so that one can determine which value is to be used for which incoming edge. Any transformation that modifies the CFG of a function needs to be careful to rewrite phi instructions to match the order in which predecessors are listed, or else the compiler must maintain a side data structure that remembers the mapping (and update it instead).

- Directly interpreting/executing code in an SSA IR with phi instructions is made more difficult because when branching to a block we need to immediately execute any phi instructions based on the block from which we just came. The above issues around phis needing to be executed in parallel, and needing to track how phi operands relate to predecessor blocks also add complexity to an interpreter.

Slang ditches traditional phi functions in favor of an alternative that matches the Swift IL (SIL).
The idea doesn't really start in Swift, but rather in the existing observation that SSA form IR and a continuation-passing style (CPS) IR are semantically equivalent; one can encode SSA blocks as continuation functions, where the arguments of the continuation stand in for the phi instructions, and a branch to the block becomes just a call.

Like Swift, we do not use an explicit CPS representation, but instead find a middle ground of a traditional SSA IR where instead of phi instructions basic blocks have parameters.
The first N instructions in a Slang basic block are its parameters, each of which is an `IRParam` instruction.

A block that would have had N phi instructions now has N parameters, but the parameters do not have operands.
Instead, a branch instruction that targets that block will have N *arguments* to match the parameters, representing the values to be assigned to the parameters when this control-flow edge is taken.

This encoding is equivalent in what it represents to traditional phi instructions, but nicely solves the problems outlined above:

- The phi operands in the successor block are now arguments in the *predecessor* block, so that the "def dominates use" property can be enforced without any special cases.

- The "assignment" of the argument values to parameters is now encoded with a single instruction, so that the simultaneity of all the assignments is more clear. We still need to be careful when leaving SSA form to obey those semantics, but there are no tricky issues when looking at the IR itself.

- There is no special work required to track which phi operands come from which predecessor block, since the operands are attached to the terminator instruction of the predecessor block itself. There is no need to update phi instructions after a CFG change that might affect the predecessor list of a block. The trade-off is that any change in the *number* of parameters of a block now requires changes to the terminator of each predecessor, but that is a less common change (isolated to passes that can introduce or eliminate block parameters/phis).

- It it much more clear how to give an operational semantics to a "branch with arguments" instead of phi instructions: compute the target block, copy the arguments to temporary storage (because of the simultaneity requirement), and then copy the temporaries over the parameters of the target block.

The main caveat of this representation is that it requires branch instructions to have room for arguments to the target block. For an ordinary unconditional branch this is pretty easy: we just put a variable number of arguments after the operand for the target block. For branch instructions like a two-way conditional, we might need to encode two argument lists - one for each target block - and an N-way `switch` branch only gets more complicated.

The Slang IR avoids the problem of needing to store arguments on every branch instruction by banning *critical edges* in IR functions that are using SSA phis/parameters. A critical edge is any edge from a block with multiple successors (meaning it ends in a conditional branch) to one with multiple predecessors (meaning it is a "join point" in the CFG).
Phi instructions/parameters are only ever needed at join points, and so block arguments are only needed on branches to a join point.
By ruling out conditional branches that target join points, we avoid the need to encode arguments on conditional branch instructions.

This constraint could be lifted at some point, but it is important to note that there are no programs that cannot be represented as a CFG without critical edges.

### A Simple Encoding of the CFG

A traditional SSA IR represents a function as a bunch of basic blocks of instructions, where each block ends in a *terminator* instruction.
Terminators are instructions that can branch to another block, and are only allowed at the end of a block.
The potential targets of a terminator determine the *successors* of the block where it appears, and contribute to the *predecessors* of any target block.
The successor-to-predecessor edges form a graph over the basic blocks called the control-flow graph (CFG).

A simple representation of a function would store the CFG explicitly as a graph data structure, but in that case the data structure would need to be updated whenever a change is made to the terminator instruction of a branch in a way that might change the successor/predecessor relationship.

The Slang IR avoids this maintenance problem by noting an important property.
If block `P`, with terminator `t`, is a predecessor of `S`, then `t` must have an operand that references `S`.
In turn, that means that the list of uses of `S` must include `t`.

We can thus scan through the list of predecessors or successors of a block with a reasonably simple algorithm:

* To find the successors of `P`, find its terminator `t`, identify the operands of `t` that represent successor blocks, and iterate over them. This is O(N) in the number of outgoing CFG edges.

* To find the predecessors of `S`, scan through its uses and identify users that are terminator instructions. For each such user if this use is at an operand position that represents a successor, then include the block containing the terminator in the output. This is O(N) in the number of *uses* of a block, but we expect that to be on the same order as the number of predecessors in practice.

Each of these actually iterates over the outgoing/incoming CFG *edges* of a block (which might contain duplicates if one block jumps to another in, e.g, multiple cases of a `switch`).
Sometimes you actually want the edges, or don't care about repeats, but in the case where you want to avoid duplicates the user needs to build a set to deduplicate the lists.

The clear benefit of this approach is that the predecessor/successor lists arise naturally from the existing encoding of control-flow instructions. It creates a bit of subtle logic when walking the predecessor/successor lists, but that code only needs to be revisited if we make changes to the terminator instructions that have successors.

### Explicit Encoding of Control-Flow Join Points

In order to allow reconstruction of high-level-language source code from a lower-level CFG, we need to encode something about the expected "join point" for a structured branch.
This is the logical place where control flow is said to "reconverge" after a branch, e.g.:

```hlsl
if(someCondition) // join point is "D"
{
	A;
}
else
{
	B;
	if(C) return;
}
D;
```

Note that (unlike what some programming models would say) a join point is *not* necessarily a postdominator of the conditional branch. In the example above the block with `D` does not postdominate the block with `someCondition` nor the one with `B`. It is even possible to construct cases where the high-level join point of a control-flow construct is unreachable (e.g., the block after an infinite loop).

The Slang IR encodes structured control flow by making the join point be an explicit operand of a structured conditional branch operation. Note that a join-point operand is *not* used when computing the successor list of a block, since it does not represent a control-flow edge.
This is slightly different from SPIR-V where join points ("merge points" in SPIR-V) are encoded using a metadata instruction that precedes a branch. Keeping the information on the instruction itself avoids cases where we move one but not the other of the instructions, or where we might accidentally insert code between the metadata instruction and the terminator it modifies.
In the future we might consider using a decoration to represent join points.

When using a loop instruction, the join point is also the `break` label. The SPIR-V `OpLoopMerge` includes not only the join point (`break` target) but also a `continue` target. We do not currently represent structured information for `continue` blocks.
The reason for this is that while we could keep structured information about `continue` blocks, we might not be able to leverage it when generating high-level code, because the syntactic form of a `for` loop (the only construct in C-like languages where `continue` can go somewhere other than the top of the loop body) only allows an *expression* for the continue clause and not a general *statement*, but we cannot guarantee that after optimization the code in an IR-level "continue clause" would constitute a single expression.
The approach we use today means that the code in "continue clause" might end up being emitted more than once in final code; this is deemed acceptable because it is what `fxc` already does.

When it comes time to re-form higher-level structured control flow from Slang IR, we use the structuring information in the IR to form single-entry "regions" of code that map to existing high-level control-flow constructs (things like `if` statements, loops, `break` or `continue` statements, etc.).
The current approach we use requires the structuring information to be maintained by all IR transformations, and also currently relies on some invariants about what optimizations are allowed to do (e.g., we had better not introduce multi-level `break`s into the IR).

In the future, it would be good to investigate adapting the "Relooper" algorithm used in Emscripten so that we can recover valid structured control flow from an arbitrary CFG; for now we put off that work.
If we had a more powerful restructuring algorithm at hand, we could start to support things like multi-level `break`, and also ensure that `continue` clauses don't lead to code duplication any more.

## IR Global and Hoistable Value Deduplication

Types, constants and certain operations on constants are considered "global value" in the Slang IR. Some other insts like `Specialize()` and `Ptr(x)` are considered as "hoistable" insts, in that they will be defined at the outer most scope where their operands are available. For example, `Ptr(int)` will always be defined at global scope (as direct children of `IRModuleInst`) because its only operand, `int`, is defined at global scope. However if we have `Ptr(T)` where `T` is a generic parameter, then this `Ptr(T)` inst will be always be defined in the block of the generic. Global and hoistable values are always deduplicated and we can always assume two hoistable values with different pointer addresses are distinct values.

The `IRBuilder` class is responsible for ensuring the uniqueness of global/hoistable values. If you call any `IRBuilder` methods that creates a new hoistable instruction, e.g.  `IRBuilder::createIntrinsicInst`, `IRBuilder::emitXXX` or `IRBuilder::getType`, `IRBuilder` will check if an equivalent value already exists, and if so it returns the existing inst instead of creating a new one.

The trickier part here is to always maintain the uniqueness when we modify the IR. When we update the operand of an inst from a non-hoistable-value to a hoistable-value, we may need to hoist `inst` itself as a result. For example, consider the following code:
```
%1 = IntType
%p = Ptr(%1)
%2 = func {
   %x = ...;
   %3 = Ptr(%x);
   %4 = ArrayType(%3);
   %5 = Var (type: %4);
   ...
}
```

Now consider the scenario where we need to replace the operand in `Ptr(x)` to `int` (where `x` is some non-constant value), we will get a `Ptr(int)` which is now a global value and should be deduplicated:
```
%1 = IntType
%p = Ptr(%1)
%2 = func {
   %x = ...;
   //%3 now becomes %p.
   %4 = ArrayType(%p);
   %5 = Var (type: %4);
   ...
}
```
Note this code is now breaking the invariant that hoistable insts are always defined at the top-most scope, because `%4` becomes is no longer dependent on any local insts in the function, and should be hoisted to the global scope after replacing `%3` with `%p`. This means that we need to continue to perform hoisting of `%4`, to result this final code:
```
%1 = IntType
%p = Ptr(%1)
%4 = ArrayType(%p); // hoisted to global scope
%2 = func {
   %x = ...;
   %5 = Var (type: %4);
   ...
}
```

As illustrated above, because we need to maintain the invariants of global/hoistable values, replacing an operand of an inst can have wide-spread effect on the IR.

To help ensure these invariants, we introduce the `IRBuilder.replaceOperand(inst, operandIndex, newOperand)` method to perform all the cascading modifications after replacing an operand. However the `IRInst.setOperand(idx, newOperand)` will not perform the cascading modifications, and using `setOperand` to modify the operand of a hoistable inst will trigger a runtime assertion error.

Similarly, `inst->replaceUsesWith` will also perform any cascading modifications to ensure the uniqueness of hoistable values. Because of this, we need to be particularly careful when using a loop to iterate the IR linked list or def-use linked list and call `replaceUsesWith` or `replaceOperand` inside the loop.

Consider the following code:

```
IRInst* nextInst = nullptr;
for (auto inst = func->getFirstChild(); inst; inst = nextInst)
{
     nextInst = inst->getNextInst(); // save a copy of nestInst
     // ...
     inst->replaceUsesWith(someNewInst); // Warning: this may be unsafe, because nextInst could been moved to parent->parent!
}
```

Now imagine this code is running on the `func` defined above, imagine we are now at `inst == %3` and we want to replace `inst` with `Ptr(int)`. Before calling `replaceUsesWith`, we have stored `inst->nextInst` to `nextInst`, so `nextInst` is now `%4`(the array type). Now after we call `replaceUsesWith`, `%4` is hoisted to global scope, so in the next iteration, we will start to process `%4` and follow its `next` pointer to `%2` and we will be processing `func` instead of continue walking the child list!

Because of this, we should never be calling `replaceOperand` or `replaceUsesWith` when we are walking the IR linked list. If we want to do so, we must create a temporary workList and add all the insts to the work list before we make any modifications. The `IRInst::getModifiableChildren` utility function will return a temporary work list for safe iteration on the children. The same can be said to the def-use linked list. There is `traverseUses` and `traverseUsers` utility functions defined in `slang-ir.h` to help with walking the def-use list safely.

Another detail to keep in mind is that  any local references to an inst may become out-of-date after a call to `replaceOperand` or `replaceUsesWith`. Consider the following code:
```
IRBuilder builder;
auto x = builder.emitXXX(); // x is some non-hoistable value.
auto ptr = builder.getPtrType(x);  // create ptr(x).
x->replaceUsesWith(intType); // this renders `ptr` obsolete!!
auto var = builder.emitVar(ptr); // use the obsolete inst to create another inst.
```
In this example, calling `replaceUsesWith` will cause `ptr` to represent `Ptr(int)`, which may already exist in the global scope. After this call, all uses of `ptr` should be replaced with the global `Ptr(int)` inst instead. `IRBuilder` has provided the mechanism to track all the insts that are removed due to deduplication, and map those removed but not yet deleted insts to the existing inst. When using `ptr` to create a new inst, `IRBuilder` will first check if `ptr` should map to some existing hoistable inst in the global deduplication map and replace it if possible. This means that after the call to `builder.emitVar`, `var->type` is not equal to to `ptr`.

### Best Practices

In summary, the best practices when modifying the IR is:
- Never call `replaceUsesWith` or `replaceOperand` when walking raw linked lists in the IR. Always create a work list and iterate on the work list instead. Use `IRInst::getModifiableChildren` and `traverseUses` when you need to modify the IR while iterating.
- Never assume any local references to an `inst` is up-to-date after a call to `replaceUsesWith` or `replaceOperand`. It is OK to continue using them as operands/types to create a new inst, but do not assume the created inst will reference the same inst passed in as argument.


