# torch._rust CFG converter, executor and optimizer: design

Each section is a design question: what we decided and why, or, for open
questions, the options and a recommendation. Implementation gaps and pending
work live in TODO.md.

# Decided

## Scope: CPython 3.12, in Rust

**Decision:** The converter, executor and optimizer live in `torch/_rust`, in
Rust, and support CPython 3.12 only.

**Why:** One bytecode and frame layout to transcribe exactly (the executor's
opcode semantics come from 3.12.11's `bytecodes.c`), so behavior can be
checked against CPython's own test suite. Other versions wait until the design
settles (see "When to support Python 3.13/3.14?").

## IR: SSA with block parameters

**Decision:** Blocks hold SSA instructions with the value stack removed;
merges use block parameters, not phi instructions. An instruction is its
value: `ValueId(n)` is `cfg.instrs[n]`.

**Why:** Block params make every edge's data flow explicit, which outlining
and versioning rely on (a block's inputs are exactly its params). Instructions
as values give O(1) use-to-definition.

## Values are local to a version; one global instruction table

**Decision:** Values don't cross block boundaries: only block params and edge
args do (`verify::check_locality`). Instructions stay in the global
`Cfg::instrs` table rather than moving into each `Version`.

**Why version-local, not dominance-based global SSA:**

- A block can have several versions, or fall back to its origin. A value
  defined in one version doesn't exist when another version or the origin (a
  separate `Cfg`) ran, so a dominated block couldn't name it; versions instead
  agree on numbered outputs that the terminator reads.
- Blocks stay independently replaceable: `optimize_block` swaps one block's
  body, and specialization will add versions, without looking at any other
  block.
- The executor's liveness is per version, and a block's leftover values drop
  when it exits, so lifetimes match CPython's.

The cost is extra params and edge args (e.g. a `with`'s `__exit__` threaded
through every block). Values flow freely once blocks are merged into one
version (outlining).

**Why a global table:** ValueIds are stable keys for the observation tables
(types, effects, callees), which carry over unchanged when an optimized Plan
replaces the old one. A `Cfg` clones in one copy, and an origin gets its own
`Cfg` with fresh ids. The verifier gives the locality guarantee that
version-owned instrs would give by construction. Tombstones accumulate; a
compaction pass can reclaim them if it ever matters.

## Exceptions: unwind segments, raising terminators

**Decision:** A version carries an ordered list of unwind segments (from an
instr index on, raising ops go to this handler with these args), instead of
splitting blocks at raising ops. `Raise`/`Reraise` are terminators.

**Why:** Splitting after every raising op made nearly every bytecode its own
block. With segments, a try/with body without other control flow stays one
block while each raising op still passes exact SSA values to its handler.

## Terminators only transfer control

**Decision:** Ops that do work are body ops even where they decide control
flow: `FOR_ITER` lowers to `IterNext` + `Branch`, `SEND` to `Send` + `Branch`,
`JUMP_BACKWARD` to `EvalBreaker` + `Jump`.

**Why:** Terminators are shared by all versions of a block, so they must not
have effects. Value-producing edges become ordinary edges, with no landing
blocks or implicit received-value params.

## Versions, guards and origins

**Decision:** A block has one shape (params, terminator, handlers) and any
number of versions, each with a guard. The first version whose guard holds
runs; otherwise the block's origin (an outlined `Cfg` holding the original
code) runs and leaves with the block's outputs.

**Why:** Specialized code always has the unspecialized code to fall back to,
at block granularity, and the origin is also what a later optimization starts
again from.

**Decision:** Guards run only on entry to a block: a version's body contains
no checks. A condition that isn't known on entry ends the region instead.

**Why:** Entry guards let the body be optimized as a whole (down to a single
fused kernel), whereas a check mid-version means running arbitrary code
inside it.

**Decision:** A guard is a list of checks, on a block param, on a read (a
param's attribute, a global, a dict item, ...) or on global state, as
Dynamo's guards are checks on sources, not code. The block keeps the reads,
each once. Only reads a check needs, the compute takes (its graph's inputs)
or the block only passes on move there; reads that feed Python code stay
in the body where the code reads them.

**Why:** Guards copied in as code (each elided callee's reads and checks,
repeated) were hundreds of instructions ahead of the compute, hard to read
and slow; as checks on shared reads they are what they check. Moving every
read known on entry bought nothing (an input costs the same read) and
moved reads ahead of the eval breaker in `Resume`.

## The executor runs on CPython's frames

**Decision:** The executor runs a frame's CFG inside CPython's own frame:
fast locals stay in the frame, SSA values are the value stack, values are
released at their last use, and a suspended generator's stack is saved into
the frame in CPython's exact layout.

**Why:** Tracebacks, `f_locals`, generators, the GC and object lifetimes all
behave as in CPython, which the comparison against CPython's test suite
checks. Either engine can resume the other's generators.

## Generators stay frames, not state machines

**Decision:** CPython's generator is a suspended frame, a one-shot
continuation; the executor keeps that model (a resume table maps a suspension
point back into the CFG). Optimized versions can't contain a `Yield`
(`outlinable`), and a `Yield` is an exit from any optimized region.

**Why:** Rewriting generators as state machines buys little (the frame
already is the state) and costs compatibility: `GeneratorType` checks,
`gi_frame`, `throw`/`close` semantics, `yield from` fast paths. Across a
suspension only the yield's stack args survive, so promoted locals are written
to the frame before a yield and reloaded after, which is noise next to a
suspend and resume. The case worth doing is inlining a non-escaping generator
into the loop consuming it (after function inlining).

## Torch ops are seen at PreDispatch, through a native hook

**Decision:** `PythonFallbackKernel` gains a thread-local native hook,
`at::impl::set_pre_dispatch_hook`, that the executor installs to record torch
ops at the PreDispatch key.

**Why:** PreDispatch is where Dynamo/export capture: before autograd and
decomposition. The only existing way to intercept it was a Python pre-dispatch
mode, a stack that accepts only Proxy/Functional/SchemaCheck modes; a second
fallback on another key isn't possible (one per key, and the free keys sit
below autograd); a Python `TorchDispatchMode` is post-autograd and a Python
round trip per op.

## Side effects are recorded per site, by the innermost activation

**Decision:** Each executor activation records the Python frames entered and
the torch ops run at its current site (instr or block). A frame is a direct
call if its function is the Call's callable (or a bound method's
`__func__`), indirect otherwise; generator resumes are their own kind.

**Why:** A call to a Python function then shows up once at the caller, and
the callee's effects at the callee's own sites, which is what an optimizer
deciding about one call needs.

## Type lattice: intervals, bounded unions, shallow containers

**Decision:** Ints are intervals, str and bool keep a constant until two
differ, tensors keep class/dtype/device/requires_grad and an interval per
dimension; a union holds up to four kinds, then Any. Containers record their
kind and length only (tuples per position, one level deep), as a snapshot at
definition. Observing never runs Python code.

**Why:** Interval dims are what shape specialization needs. Shallow
containers keep observation bounded: element facts come from the values read
out of a container, which is also how Dynamo only learns about the parts of an
input that the code touches. Scanning every element at every observation made
the CPython suite about 5x slower, with many modules timing out. Running
Python during observation would change behavior (`__instancecheck__`,
properties, a str's cached UTF-8).

## Locals and arguments aren't recorded at entry

**Decision:** No observation of a frame's locals or arguments when it starts.

**Why:** What the code reads is observed where it's read (the loads), and a
later lazy "evaluate when read" scheme covers locals the same way.

## Optimizing a block outlines it into itself; Plans are swapped

**Decision:** `optimize_block` outlines a block into itself (the origin keeps
the original, the version gets the optimized body) and swaps in a new `Plan`.
Frames already running keep their Plan; observations carry over by value id.
Blocks holding a generator resume point, blocks with an origin and blocks with
several versions are refused.

**Why:** Replacing the Plan atomically means no running frame sees a half
updated CFG. Refusing resume-point blocks means a suspended generator always
resumes at a point that exists in the new Plan.

## Optimization follows Dynamo's assumptions, only in optimized code

**Decision:** The optimizer optimizes what Dynamo would, and makes Dynamo's
assumptions there: inside an optimized stretch, frame locals may be stale and
old values released late. Code that isn't optimized keeps exact CPython
semantics.

**Why:** The goal is to optimize the same code Dynamo does, so we take the
same bets in the same places. Anywhere Dynamo would graph-break, we keep the
frame exact.

## Optimized code is pure blocks between glue that loads and stores

**Decision:** Optimizing a block splits it at its barriers (ops not judged
movable, where Dynamo would graph-break) into glue blocks and pure blocks. A
stretch of movable ops becomes a pure block that takes the values it needs as
params and produces its results as outputs; its first load of each local
moves to the glue before it, later loads reuse the value last loaded or
stored, and its stores move to the glue after it (dropped when only a return
follows). Only pure blocks get versions, guards and replaced ops. If a pure
block raises while stores are pending, it unwinds to a landing block that
writes them and re-raises (with the original lasti) into the original handler.

**Why:**

- It's Dynamo's structure: a graph is a pure function of its inputs, and the
  bytecode around it loads locals, calls it and stores its outputs.
- Guards are checks on block params, which is what the IR's `Guard` already
  was. The values a guard needs are exactly the pure block's inputs.
- Pure blocks hold no frame access and no barriers, so op replacement and,
  later, codegen work on a self-contained function.
- Landing blocks keep frame writes out of the IR's exception machinery (an
  earlier version put pending stores on unwind segments): a handler or a
  traceback sees the locals CPython would have left, and the normal path pays
  nothing.
- Moving stores past every op broke code that reads the frame or depends on
  when an old value is released (`% locals()`, finalizers, weakrefs,
  `gc.collect` counts): 377 differences over the CPython suite, all where
  Dynamo graph-breaks. With barriers the suite matched the unoptimized
  executor. Never moving stores was also clean, but gives up store motion in
  the code Dynamo optimizes.
- Blocks are renumbered afterwards (the IR numbers blocks in RPO), and
  block-site effects move with their blocks.

## Operators become aten ops, guarded like Dynamo's inputs

**Decision:** In a pure block's version, an operator (binary, compare,
unary) whose every visit ran one aten op, with a plain tensor as its left
operand, is replaced by that op (`a + b` by `aten::add.Tensor(a, b)`), called
through the overload's C++ binding. The version's guard checks each operand
that is a param: `plain_tensor` (exactly `torch.Tensor` or `nn.Parameter`)
or `int`/`float`/`bool` exactly, plus `no_torch_function_mode`. An operand
computed in the block counts only if it's a constant or another replaced
op's result.

**Why:**

- These are the guards that decide whether the operator dispatches straight
  to that aten op: a tensor subclass can override the operator or
  `__torch_function__`, a scalar subclass can override the operator (or the
  arg types can pick another overload), and an active `TorchFunctionMode`
  would see the operator but not a direct aten call. Dtype, device, shape,
  grad mode and autocast don't change which op runs (they act below the point
  where both paths meet), so they aren't needed until code is generated.
- An aten op on plain tensors returns a plain tensor, so replaced results
  need no checks of their own; nothing can check other values in the middle
  of a version, so operands from non-replaced ops block replacement.
- Reflected operators (`2 - t` runs `t.__rsub__(2)`, recorded as
  `rsub(t, 2)`) don't map operands to aten args uniformly (`__rpow__` keeps
  the order, `__rtruediv__` runs two ops), so a tensor must be on the left.
- Calls (`torch.relu(x)`) aren't replaced yet: their callee comes from a
  global or attribute load inside the block, which needs an identity guard on
  an object (see the guards question below).
- The C++ binding with Python scalars gives the same type promotion as the
  operators (checked across dtypes), so scalars need no explicit
  wrapped-number conversion.

## Which ops are movable: decided by identity, as Dynamo does

**Decision:** Movable ops are, from observations: local ops, globals and ops
that can't raise; arithmetic, compare, attribute and subscript ops whose
operands were only ever builtin scalars, str, tensors, tuples or lists
(modules for attributes) and that entered no Python frame; calls that always
called the same callable, entered no Python frame, and that
`torch._dynamo.trace_rules.lookup` classifies as `TorchInGraphFunctionVariable`.

**Why:** Dynamo decides by the callable's identity, not by what the call did:
a C function that runs torch ops isn't safe by default (`locals()`, `vars()`
and `gc.collect` enter no Python frame and run no torch op either). Using
`trace_rules` keeps us aligned with what Dynamo captures. Effects remain a
sanity check (no Python frame entered).

## Callees are held so they can still be freed

**Decision:** Each Call records its callee: functions, classes and unbound
builtins through weakrefs, method descriptors of static types strongly; any
other callable (a builtin bound to an object, a callable instance, a
descriptor of a heap type) isn't identified, so the call stays a barrier.

**Why:** Holding callees strongly kept user objects alive and broke CPython
tests: a builtin bound through a weakref proxy kept its array alive, a class
kept its instances' cycle alive, and module builtins kept a freshly imported
module alive. Method descriptors can't be weakly referenced, but a static
type's descriptor can never be freed anyway.

# Open

## What else do guards check, and what happens on a miss?

Guards check what aten replacement needs (param types, torch-function mode).
Not yet guarded: the observations `movable` relied on (a later run with a
different operand type or callee in a pure block sees stale locals there),
callee identity for replacing calls, and dtypes/devices/shapes, which code
generation will need. On a miss the origin always runs.

**Recommendation (Claude):** Guard every pure block's params on the types its
movable ops relied on, and callees by identity (globals and module
attributes the block loads, checked at entry as Dynamo's `ID_MATCH` does,
which turns those loads into guarded constants). Add dtype/device/shape
checks with codegen. On a miss, run the origin, which keeps observing; after
a few misses re-optimize from the joined observations as another version,
with a cap on versions per block (like Dynamo's cache size limit) after which
the block stays unoptimized.

## When and what should be optimized?

Nothing decides today: there's the manual `optimize_block` and the testing
knob `_set_cfg_optimize`.

**Recommendation (Claude):** A per-code-object call counter; after a small
number of runs (enough for observations to settle, say 2-8), optimize every
block that ran, at once. Per-block hotness can come later, when optimizing
gets expensive enough to matter.

## Python callees: inline, or summarize?

Calls to Python functions are barriers. Dynamo inlines them; the alternative
is a summary from the callee's own observations ("never touches its caller's
frame").

**Recommendation (Claude):** Inline, for Dynamo parity: it lets SSA and store
motion cross calls, and summaries are fragile (`sys._getframe`, callees of
callees). Barriers until inlining exists.

## What should observation cost once code is optimized?

**Decision:** A Cfg stops recording once the optimizer has seen every block of
it; optimized versions never record, origins always do, and a site keeps 8
distinct effect sequences before it's marked as varying. The op hook is
switched off while nothing records.

**Why:** Nothing reads what's recorded after optimization, and recording plus
the per-op boxed redispatch through the hook were a third of the executor's
time on small training steps. Origins keep recording so misses can drive
re-optimization later.

## What does the CFG eventually compile to?

The target shapes what the optimizer should produce.

**Recommendation (Claude):** Following Dynamo: FX graphs for the stretches
`movable` accepts, handed to Inductor, with the executor running everything
else. Bytecode generation adds little, since the executor already runs the
CFG.

## When to support Python 3.13/3.14?

The converter and executor are transcribed from 3.12's bytecode and frame
internals; 3.13 changes both.

**Recommendation (Claude):** After the optimizer's shape settles (guards,
regions, codegen target), so the 3.13 port follows a stable design.

## What happens to the CPython comparison harness?

It's the main correctness check for the executor and the optimizer
(`CFG_OPTIMIZE=1`).

**Decision:** It's checked in as `torch/_rust/tools/compare_cpython.py`, with
its module list next to it, reading `Lib/test` from a CPython checkout given
with `--cpython`. Not a CI job yet.

**Why:** A full run takes about 30 minutes on 16 workers and needs a CPython
checkout at the interpreter's tag.
