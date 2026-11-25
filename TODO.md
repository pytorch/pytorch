# torch._rust CFG converter / executor: known issues and pending work

## Build and tooling

- Buck build: not updated for `csrc/frame_shim.c`, `csrc/exec_ops.c`,
  `csrc/op_hook.cpp`, the `cc` build dependency, or the link against Dynamo's
  exported `THP_PyFrame_*` helpers in libtorch_python. Untested.
- Clippy needs the install-tree headers. To run it:
  `TORCH_RUST_INCLUDE_DIR=$PWD/build/install/cp312-cp312-linux_x86_64/torch/include cargo clippy --profile=release --manifest-path torch/_rust/Cargo.toml --target-dir build/rust --all-targets -- -D warnings`
  (use an absolute path for the include dir).
- `frame_shim.c` uses `pthread_getattr_np` for the native stack guard, so it's
  Linux-only. macOS and Windows aren't handled.
- Before shipping, every public entry point needs a Python API: today
  callers use the extension's functions directly
  (`call_with_cfg_execution`, `_set_cfg_optimize`, `_specialize`,
  `_reoptimize`, `cfg_observed`, ...), with no Python-level module, type
  stubs or docstrings, and test-only hooks mixed in with real ones. Wrap
  the public ones in Python (documented and typed), keep the extension
  private, and leave testing hooks underscored.

## Executor fidelity gaps (CPython runs these parts)

- PEP 695 typing intrinsics (unary 7-11, binary 2-3) run through
  one-instruction trampoline code objects.
- Eval breaker: runs `_Py_HandlePending` through a trampoline, at every
  `RESUME` (`Op::Resume`, including after each yield), after calls, and at
  `JUMP_BACKWARD` (`Op::EvalBreaker`).

## Executor limitations (fail hard by design)

- Tracing, profiling and `sys.monitoring` are refused with `RuntimeError`.
  Supporting them would need line/call/return events emitted from the
  executor.
- Invalid bytecode is rejected at conversion with `NotImplementedError`.
  CPython raises `SystemError` only when the bad instruction runs (see
  test_code `test_invalid_bytecode`).
- A code object is rejected if liveness finds a value other than the yield's
  stack args live across a `Yield` (`live.bad_yield`).
- CPython's adaptive specialization is not done (test_dis checks for it).
- The hook is installed interpreter-wide and gated by a per-thread counter.
  Other threads pay the cost of the hook check.
- Threads: frames of the same code running concurrently on several
  threads haven't been looked at (recording races, Plan swaps, the
  optimizer's state); Dynamo serializes compiles with a `compile_lock`.
  Free-threaded 3.13t is untested too.

## Torch op hook (pre-autograd, csrc/op_hook.cpp)

- Enabled under `call_with_cfg_execution`, but it runs only while a frame is
  recording (or ops are being printed): each op is recorded as a side effect
  (below). `torch._rust.set_print_torch_ops(True)` also prints them; that
  switch is process-global, but the hook is per thread.
- Ops inside a reported op aren't reported, but still go through the hook
  (a boxed redispatch each, only while recording) so their writes and
  random tags are attributed to it (see step 4 under Optimizer).
- Reports every top-level op on the thread, including ops run by torch's own
  Python internals (e.g. Dynamo's fake-tensor ops when eager `torch.cond`
  compiles itself). They land on whichever executor frame is innermost.
- Ops run on other threads (autograd engine, intra-op pools) aren't recorded.
- Turning `set_print_torch_ops` on while inside `call_with_cfg_execution`
  with nothing recording doesn't restart a paused hook until the next
  recorder change.
- Higher-order ops dispatch in Python and never reach the C++ PreDispatch
  fallback, so we see the aten ops inside them, not the HOP.
- `make_fx(pre_dispatch=True)` (or export) run inside the executor clears the
  PreDispatch TLS include bit when its own mode stack empties, which silently
  turns our hook off for the rest of the call.
- `op_hook.cpp` includes the cxx-generated `torch_rust/src/bindings/ffi.rs.h`
  (Cargo's path); Buck's generated-header path may differ.
- ATen side: `at::impl::set_pre_dispatch_hook` in
  `aten/src/ATen/core/PythonFallbackKernel.{h,cpp}`. `torch/_ops.py`
  HigherOrderOperator.dispatch now skips PreDispatch when no pre-dispatch mode
  is active (it used to fail, since that key was never on without a mode).
- `pytorch-fastcompile` quick mode doesn't refresh the libtorch that the
  editable install loads (`build/install/...`); ATen changes need `--full`.

## Side effects (exec/effects.rs)

- Recorded per site (an instr, or a block outside its instrs) of each Cfg,
  origins included, in the code object's `Compiled`; printed by
  `torch._rust.cfg_side_effects(fn)`. Each distinct sequence a visit produced
  is kept with a count, so a site with varying effects grows a list without
  bound (see QUESTIONS.md).
- Effects are strings (`module.qualname`, op name plus formatted args). An
  optimizer will want the function/code object and the op handle instead;
  holding strong refs from co_extra needs care (a recursive function would
  keep its own code object alive).
- Direct vs indirect: a frame is direct if its function is the Call's
  callable (or a bound method's `__func__`); anything else entered during the
  instr is indirect (C code, type slots, `__torch_function__`, ...).
- Frames CPython runs for us (trampolines, generator prologues) don't record,
  so their effects land on the caller's site. So does the eval breaker after a
  call (signal handlers, pending calls), on the block site.
- A Cfg records until every block of it has been offered to the optimizer
  (`_set_cfg_optimize`'s whole-code pass; `optimize_block` of one block keeps
  it recording). Optimized versions never record; origins always do, so guard
  misses are still seen. A site keeps 8 distinct effect sequences, then is
  marked as varying, which the optimizer treats as unknown.
- Nothing re-optimizes yet, so what origins record on misses is unused.

## Optimizer (cfg/optimize.rs)

- `torch._rust.optimize_block(fn, b)` optimizes block `b` of the CFG the
  executor runs for `fn`: the block is outlined into itself (its origin keeps
  the original) and its version is the optimized body.
  `torch._rust._set_cfg_optimize(True)` optimizes every block of a code's CFG
  after its first frame returns (for testing).
- The executor swaps in the result as a new `Plan` (prepared Cfg plus generator
  resume points); running frames keep the Plan they started with. Values keep
  their ids, so effects and types carry over; the new origin starts with none.
- Blocks holding a generator resume point, blocks that already have an origin,
  and blocks with several versions are refused. Resuming a generator under a
  replaced Plan relies on resume-point blocks never changing.
- Optimizing is manual (from Python); when and what to optimize: see
  QUESTIONS.md.
- Optimizing splits a block into glue and pure blocks (see QUESTIONS.md):
  loads before, stores after, landing blocks for raises with pending stores.
  Inside a pure block, as under Dynamo, the frame's locals are stale and old
  values are released late. A stretch with no real work (only constants and
  locals) isn't split out.
- Aten replacement covers binary, compare and unary operators with a plain
  tensor on the left. Not yet: reflected operators, calls (need identity
  guards), subscripts (`t[0]` is `select.int(t, 0, 0)`, not a 1:1 mapping),
  ops recorded as more than one aten op.
- Movable (`observed_ops` in exec/mod.rs), from observations only: local ops,
  globals, ops that can't raise; arithmetic/compare/attribute/subscript ops
  whose operands were only ever builtin scalars, str, tensors, tuples or lists
  (modules for attributes) and that entered no Python frame; calls that always
  called the same callable, entered no Python frame, and that
  `torch._dynamo.trace_rules.lookup` classifies as
  `TorchInGraphFunctionVariable`. No guards back these yet (see
  QUESTIONS.md).
- Callees (`effects::Callee`) are identified only for callables holding no
  user data, held so they can still be freed: functions, classes and unbound
  builtins through weakrefs, method descriptors of static types strongly
  (descriptors can't be weakly referenced; one of a heap type isn't
  identified). A builtin bound to an object (e.g. through a
  weakref proxy) or a callable instance isn't identified, so such calls stay
  barriers; Dynamo handles bound tensor methods as tensor methods, which we
  could too by recording `(type(self), name)`.
- Python callees are barriers (inline or summarize: see QUESTIONS.md).
- `_set_cfg_optimize(True)` optimizes every block of each code after its
  first frame returns; the CPython comparison runs with it.
- Next, in order (from the autograd coverage run and the perf comparison,
  see Testing):
  0. Done: a call of a Python function Dynamo's trace rules put in its
     graphs (`F.relu`, `F.layer_norm`) is movable and pinned like a builtin
     when every visit entered only its frame and that function's own Cfg
     recorded no other frame and only torch ops that write nothing and
     draw no random numbers (`leaf_ops` in
     exec/mod.rs). Optimizing runs outside the executor: before, the
     optimizer's own Python (trace_rules.lookup, OpOverload.tags) was
     recorded as effects of the call whose callee had just returned.
  1. Done: calls to in-graph torch functions that ran one torch op keep
     their call but pin the callee (`Op::Pinned`): guarded on the
     global/module-attribute loads still producing it (the loads become
     reads of the guard with an `Is` check, `guard F.relu is
     torch.nn.functional.relu`; every optimized pure block's reads and
     checks are lifted, elided or not; a callee loaded after something
     that writes isn't pinned, since its check couldn't go in the guard)
     when the loads are in the block, or
     on the callable param's identity (`Check::Is`) when a barrier put them
     in glue. Not turned into aten ops:
     a Python binding's args don't map 1:1 onto a schema (namedtuple returns,
     int -> int[], `input`/`self`, defaults), and Dynamo keeps them as calls
     too; autograd's tracer sees the aten ops anyway. With it the compute
     blocks of the coverage models train (see Autograd).
  2. Inline Python callees whose callee is always the same function
     (guarded on identity), so pure regions span module boundaries
     (`model(x)` -> `Module.__call__` -> `_call_impl` -> `forward` ->
     `F.linear`). That Python between frames is the executor's remaining
     1.4-1.8x over eager; torch.compile's eager backend sits at eager speed
     because Dynamo traced it away. It also gives autograd regions worth
     training. Open: tracebacks through inlined frames, generators, code
     reading its own frame (`locals()`, `sys._getframe`), where to stop.
     Observations are per code object, so the module path's shared callees
     look polymorphic (`_call_impl`'s `forward_call` callee is "several",
     `Module.__getattr__`'s branches go both ways); specializing them needs
     per-call-site profiles. Inlining alone won't close the gap to eager
     (see Testing: plain-Python interpretation cost and generators dominate).
     Plan (approved 2026-11-15): elide a callee's observed path into the
     caller's pure block, its branches becoming checks in the version's
     guard, with a failed check running the block's origin (a raise once
     reran it too; see guards below); per-call-site clones for polymorphic callees. Gaps it
     must close first: nothing re-optimizes a code after its first pass;
     spliced code can't use frame-relative ops (co_names/co_consts,
     f_globals, set_instr origins); `record_callee` records no callee for
     callable instances (module calls); `LoadAttr` on non-module objects
     isn't movable; autograd's `Where` and the reporting entry points find
     a Cfg by code object, which a clone wouldn't share.
     Step 1 so far: callees are recorded for callable instances (their
     class's Python `__call__`, "via __call__") and for attribute loads from
     instances whose class has a Python `__getattr__` ("via __getattr__",
     entered only when the site's effects show the frame), so a module call
     enters `_wrapped_call_impl` directly. Making instance attribute and dict
     reads movable was tried and backed out: the reads land in pure blocks,
     which autograd then refuses ("runs LoadMethod"), and read-only pure
     blocks only add guards. Hoisting them out naively changes which
     exception wins when a read and an earlier op both fail, and the locals
     a handler sees; they need guards on reads (step 3), and are only
     movable for elision until then.
     Step 2 so far: `torch._rust._reoptimize(fn)` optimizes again from
     the first Plan's Cfg (the code as converted), whose value ids the
     observations are kept by; observations of values only later Cfgs had,
     block-site effects (kept in the latest numbering) and what origins
     recorded on misses (their own tables) are dropped, and so are training
     versions. Optimizing gives each instr of the top-level blocks that
     entered one Python function directly on every visit (a call, an
     instance call, a `__getattr__` fallback) a clone of that function's
     Compiled (`Prepared::clones`, carried to later Plans; depth <= 4; no
     generators), handed to the frame through `NEXT_CLONE` (code object and
     calling frame must match; taken once, cleared after the instr). Every
     Compiled is now Arc'd (co_extra holds one), and autograd's `Where`
     holds the Compiled instead of finding it by code object. Reporting
     takes a `clone=[instr, ...]` path (`cfg_observed`,
     `_cfg_autograd_blocks`, `_cfg_misses`, `_reoptimize`, `_cfg_clones`).
     Nothing triggers re-optimization yet, and clones only cost (conversion,
     recording) until elision uses them.
     Step 3 so far (behind `torch._rust._set_cfg_elide(True)`, applied by
     `_reoptimize`): a version's guard can check reads (see "Guards"
     below), and a call whose clone
     observed one path through a plain function (positional, keyword and
     default args; no *args/**kwargs, cells or free variables; ops that
     don't read the frame's names; nested elidable calls) is elided
     (cfg/inline.rs): callee locals become values, globals are read with
     `LoadGlobalIn` through the pinned globals dict, consts and names past
     the code's own come from `Cfg::const_pins` and extra names, defaults
     are pinned behind an `Is` check on `__defaults__`, and reads and checks
     that don't depend on compute are lifted into the guard. The callee's
     observations (aten ops, pinned calls) carry over, so its operators and
     torch calls become aten ops and pins. `is_leaf` refuses callees that
     write anything but their own locals. A block whose version has inputs
     trains (3d): the bootstrap checks the guard and reads the inputs, then
     traces the version's body (not the origin) on the params and inputs;
     the training version shares the guard, adds checks on the inputs, and
     runs `Autograd` on params ++ inputs. In a version with elided calls,
     an operator whose operand was read in it (a global tensor) is replaced
     too, behind a check on that read in the guard. When a training
     version's check on an input fails, the plain version runs. The module path elides (3e): instance
     calls (a `TypeAttr` check on `__call__`), bound methods (`__func__` checked,
     `__self__` bound), `CALL_FUNCTION_EX` of a tuple and empty dict the
     elision built, `*args`/`**kwargs` binding, cells as values,
     `__getattr__` fallbacks (a `Getattr` check: generic `__getattribute__`,
     nothing by that name on the class or in the instance dict), reads that
     ran no Python and no torch op (instance attributes, dict lookups), and
     in-graph calls found through a load chain in the callee's globals that
     ran nothing observable (`torch._C._is_tracing()`: pinned, and a call
     of a pinned callee with no args counts as a read).
     `has_torch_function*` of plain tensors folds to False (`use_aten`), and
     an Expect of a known bool is dropped. With `_set_cfg_optimize` and
     elision on, a code with clones is optimized again at its 10th return
     (3f; clones nest 16 deep). An elided `nn.Linear` call trains as one
     block. Single-threaded ms/step (agent_space/inline/stepbench.py,
     executor, elision off -> on): mlp 0.71 -> 0.58 (autograd on 0.74 ->
     0.67), block 0.81 -> 0.78 (0.96 -> 0.92), attn 1.50 -> 0.98 (1.37 ->
     1.02); eager 0.30-0.39, 0.47-0.53, 0.68-0.72. Fixed on the way:
     `CALL_FUNCTION_EX`'s callee was recorded from args[1] (the tuple)
     instead of args[0], so `f(*args)` sites had no callee. Elision didn't
     rewrite handler args that named the elided call's result (found when
     guards made more elisions succeed).
     Gaps: guarded reads still run on each entry (shared, but nothing folded
     under dict or type version guards); `Getattr` checks materialize the
     instance dict; nothing guards the class of an instance read in elided
     code (the read still runs, so it's correct, but a property added later
     would run Python inside the guard, and run again in the origin if a
     later check missed); the re-optimization trigger is a fixed count; a block
     that bootstraps but never sees a backward (forward-only loops) runs the
     tracer on every call; a constant branch is trusted after one visit.
     Guards (replacing guard prefixes, which replaced restarts and step 4's
     undo log): guards run only on entry (QUESTIONS.md, "Versions, guards
     and origins") and are checks on reads, not code. Elision and
     specialization copy code in with `Expect` ops, each holding a `Check`
     (`Value(True)`, `Is`, `Exact`, `TypeAttr`, `Getattr`, ...), where its path
     depends on something; `guard::lift_guard` then moves into
     `Block::reads` the reads (whose args are params or lifted reads) that a
     check needs, that the block's compute (aten ops, pinned calls) takes,
     or that the block only passes on, leaving reads that feed Python code
     (`foo` in `foo()`) where they were (deduplicated, in
     the order they first ran; a read or check taking a constant gets its
     own copy; an empty dict stays in the body, since it's a new object
     each time: shared, it made two elided calls' `**kwargs` one dict,
     which test_optparse found) and every check into the version's
     `Guard::checks`, each with its target (a read after an instr that writes Python state, see
     `guard::writes`, stays in the body: lifting reads past `del lst[0]`
     broke test_optparse once every pure block lifted), and the body
     takes the reads it uses as `Version::inputs`. A check left in the body
     (on something the body computes) refuses the elision or the region.
     A guard is one list of checks, each on a block param, a read, or (grad
     mode, torch-function modes and the like) nothing, all printed as
     `guard` lines; an `Expect` before lifting holds the same `Check`. The
     verifier holds reads to `guard::guard_read` (what lifting produces),
     with a `Project` right after its `LoadMethod`, and checks each check's
     target. The guard's order is the order checks were added (cheap param
     and global checks mostly first; a training version's are appended
     last, after reads it doesn't need first). On entry, `select` runs the block's reads in order only as far as the
     next check needs (a read runs after the checks that came before it, as
     in the code), shared by the block's versions; a failed check tries the
     next version, a raising read is a miss (counted by `_cfg_misses`, the
     origin runs, nothing to undo). The printer shows a guard as `guard
     <expr>` lines, with reads used more than once named (`$k = ...`). The
     nested-module example's training version is 96 checks over 40 shared
     reads, then one Autograd op (it was ~440 interleaved instrs). After the
     guard, a raise unwinds through the version's unwind segments to the
     handler the code would have reached, without the elided callees' frames
     in the traceback. Elision keeps only the calls whose checks all end up
     in the guard (`optimize_blocks` tries each call on a copy of the Cfg,
     so it's quadratic in a block's elidable calls); a branch on what the
     version computes leaves the call a call. Before, a version could
     restart (a raise or a failed in-body check reran its origin from the
     start), and step 4 undid the tensor writes, RNG draws and attribute
     stores a rerun would repeat; with no reruns, all of that went, and
     mutating ops, random ops and attribute stores on plain objects sit in
     pure blocks as they are (a raise after them leaves them done, as
     CPython does). The bootstrap no longer reruns a block whose traced run
     raised either (it repeated an in-place op); it still reruns the origin
     when the run finished but its node couldn't be made, which would
     repeat effects (rare: wrapping the outputs failed). Not done yet:
     folding a guarded read into the body as a constant (an `Is` check
     makes it a `Pinned`, a `Value` check a `Lit`), and dict and type
     version tags; a guarded
     read only some versions check still runs if an earlier version's
     checks needed it.
     Nested modules (agent_space/defer/nested.py: Outer.forward calling
     two Inner modules, each calling nn.Linear, with branches on Python
     config and on tensor data): `m(x)` elides into one ~430-instr block
     (four pinned calls and an aten add), once MAX_CLONE_DEPTH went from 8
     to 16: each module level costs three clones (`_wrapped_call_impl`,
     `_call_impl`, `forward`), so 8 stopped inside the second level and the
     inner Linear was never cloned. 16 covers about five levels of modules. The block
     trains: branch conditions on Python config (`self.act == "relu"`) are
     compares of reads, which `is_read` counts, so they join the guard
     (see Guards above). In elided code a compare isn't checked to be on
     exact builtins (as specialization does), so a custom `__eq__` runs in
     the guard,
     ahead of where it ran and again on a miss (like a custom `__bool__`
     below). A branch on
     tensor data (`if h.sum() > 0`) now leaves its call a call (it was an
     in-body Expect whose flips reran the whole forward). Gaps:
     `Module.__setattr__` is Python, so `self.x = y` in a module still ends
     the region (it would need `object.__setattr__` through `super()`
     treated as a store); dict stores (`STORE_SUBSCR`) aren't movable.
     BatchNorm still doesn't
     elide: `F.batch_norm` calls `_verify_batch_size` in training, whose
     loop elision can't copy (loops would need unrolling, as Dynamo does);
     `nn.Dropout` stops at `_VF.dropout` (`VFModule.__getattr__` calls
     `getattr` on a module, not pinned).
     Generic only (no nn.Module fast paths). Order: generic pieces (dict
     reads, instance callees, `__getattr__` fallback, instance attribute
     reads), re-optimization and per-site clones, elision of restartable
     paths, then effects in pure blocks (done, see Guards). Regions end where
     torch.compile graph-breaks (tensor-value branches, I/O, opaque calls).
- Specialization (cfg/specialize.rs, manual for now:
  `torch._rust._specialize(fn, block, clone=[])`): a region starting at a
  block, whose branches each went one way when observed (the branch
  condition's type is a constant bool), becomes one block whose version runs
  the path taken, with a check on each branch's condition in its guard;
  its origin is a copy of the whole region, run from its start on a miss.
  So each condition must be known on entry: reads move into the guard
  unless what ran before them may have changed what they read (a local
  stored or deleted earlier; any other read after code that can run
  Python), and a ToBool or Compare may move ahead of calls only when its
  args were always exact bools, ints, floats, strs or None, each checked in
  the guard. A
  region is single-entry (every block but the head has all its predecessors
  in it, the head none) and ends at the last block of the path; its other
  blocks may leave it only as that block does (a return, or a jump to the
  same block, rewritten to Leave in the origin) or by raising out of the
  frame. The owner block's outputs are what the exit block's terminator
  reads. Gaps: nothing calls it automatically, and it runs on unoptimized
  blocks only, while pure blocks need blocks without origins (split refuses
  them), so specialized regions don't get aten ops or training yet: next is
  specializing first and splitting inside the version, or specializing
  across already optimized blocks; a local read moves ahead of calls even
  though a write through `frame.f_locals` (PEP 667) could change it; a
  ToBool or Compare is a read (`is_read`), so in elided code a custom
  `__bool__` or `__eq__` runs in the guard; a branch is trusted after one visit, and a guard
  that keeps missing isn't noticed (respecialize or give up on it);
  handlers inside the region only as raise targets (a raising terminator in
  it must propagate out of the frame).
- Later: promote locals across a multi-block region (loops). The region's entry
  loads, its exits store, loop headers get params for promoted slots, and the
  unwind stores carry over unchanged. Needs outlining of regions, not just
  chains. A Yield stays a region exit (only its stack args survive a
  suspension).
- Guards and glue transitions of optimized blocks cost time even when the
  block does nothing worth optimizing (optimize + autograd on was slower than
  plain recording before recording stopped); blocks that gain nothing could
  stay unoptimized.
- Later: inline non-escaping generators into the loop consuming them (Yield
  becomes a jump into the loop body, `ForIter` a jump back after the yield),
  after function inlining.
- Loop unrolling passes (full and partial), driven by observed trip counts
  and guarded on them, so short stable loops (`for layer in self.layers`)
  become one straight-line pure block. Loops left rolled keep their header
  and back edge in glue with the body as its own pure block (see Autograd).
- Elision coverage vs Dynamo (reference: `torch/_dynamo/variables/`,
  `torch/_dynamo/polyfills/`). Dynamo traces through these; elision can't
  copy them into a region yet:
  - Wrapped and decorated functions: `functools.partial`, `@wraps`
    wrappers, `lru_cache`, `@contextmanager`.
  - Context managers (`no_grad`, `set_grad_enabled`, `autocast`, device
    contexts, `sdpa_kernel`); each needs to become a guarded state change
    inside the region.
  - try/except.
  - Classes and descriptors: `super()`, property/classmethod/staticmethod
    and other descriptors, dataclasses, namedtuples, Enum.
  - Builtins and containers: dict/list/set/deque methods, `len`,
    `isinstance`, `getattr`/`hasattr`, `zip`/`enumerate`/`range`,
    itertools, operator.
  - User iterators and f-strings.
  - `print`, `logging` and `warnings` calls, which Dynamo reorders or
    ignores.
- Python side effects inside a region compiled as one unit: attribute,
  list and dict stores, global and cell writes, and new objects escaping
  the region must be replayed after the graph, as Dynamo's
  `side_effects.py` does (a separate replay path per container kind, dead
  new objects pruned). Today they run in place because pure blocks run
  eagerly.
- Free tensors: globals, closure cells and module params read in a region
  become graph inputs or constants (Dynamo's `install_free_tensors`).
  `nn.Parameter` created inside a region.
- `.item()` and `.tolist()` inside a region instead of ending it
  (`capture_scalar_outputs`).

## Autograd (exec/autograd.rs, behind `_set_cfg_autograd(True)`)

- Implemented (phases 1-5 of the plan): grad guards (`GradEnabled`,
  `RequiresGrad`, `TensorMeta`, `Value`, `NoForwardAD`,
  `NoSavedTensorHooks`); aten graphs with literals, kwargs and multi-output
  ops run without a frame (exec/graph.rs); the joint graph traced below
  Autograd with a Python TorchDispatchMode (exec/trace.py, exec/joint.rs),
  which renames in-place ops on unviewed intermediates to their
  out-of-place overloads (see Functionalization below); a C++ CfgBackward node
  (csrc/autograd_node.cpp); the bootstrap (first grad-requiring run traces
  the block eagerly on detached copies); the training version swapped in at
  the first backward (Op::Autograd); and the min-cut partition
  (cfg/partition.rs).
- Not yet (phase 6): specializing the backward on which tangents are
  defined. Undefined gradients are materialized as zeros, as custom
  Functions do by default.
- Only pure blocks of aten ops and pinned calls (no other calls, attribute
  loads or globals) are eligible; others keep eager per-op autograd. Pure
  blocks are small today (a Python callee or a non-replaced operator ends
  one), mostly one op each, so a training version gains nothing over aten's
  own autograd node and costs a Python round trip per node (see Testing for
  timings); inlining is what makes them worth it. Coverage run
  (agent_space/autograd/coverage.py, using `_cfg_autograd_blocks`) over an
  MLP, MLP with dropout, residual block, attention, conv + batch norm and
  nn.TransformerEncoderLayer: every block doing tensor work trains
  (`Linear.forward`'s `F.linear`, `F.relu`, `layer_norm`, `batch_norm`, conv,
  max_pool, the residual adds and attention's blocks), except dropout's
  (`bernoulli_` is random). Before pinned calls, one block trained.
- Bootstrap params: tensors (guarded on metadata), scalars and tuples/lists
  of ints (guarded by value), NULLs the version's guard checks (`Check::Null`,
  from values only ever seen NULL), pinned callees (`Check::Is`), and params
  no op in the block reads (only passed on, returned as they are). Anything
  else makes the block give up.
- An operator after a pinned call whose args include something other than
  plain tensors and builtin scalars (a tuple, say: `torch.reshape(x, s) *
  2`) isn't replaced with its aten op, since a tuple could hold a tensor
  subclass that makes the call's result one; the block then stays
  ineligible.
- If the bootstrap's traced run raises, the block gives up bootstrapping
  and the exception goes on as raised. A block with writes or random ops
  bootstraps and joint.rs refuses it (mutating an input, random ops);
  before entry-only guards such an op was a barrier, so blocks around an in-place op
  could train separately and now can't (splitting training regions at
  them would get that back). A
  TorchDispatchMode raises KeyError('__builtins__') the first time one is
  used under a frame whose globals lack `__builtins__` (also in eager
  PyTorch), which this covers.
- Functionalization. joint.rs handles only an in-place op on a tensor the
  graph made that nothing views (renamed to its out-of-place overload, later
  uses read the new value) and refuses every other write. Since entry-only
  guards made in-place and random ops movable, they sit inside pure
  blocks, and those blocks don't train: BatchNorm (running stats are
  inputs), dropout (`bernoulli_` is random), `x += 1` on an input. Needed,
  as AOTAutograd does it:
  - Writes to inputs: trace the out-of-place op and write the final value
    back into the input after the graph (`copy_` in the glue, bumping its
    version counter as eager would). Inputs mutated before being read for
    backward need the pre-mutation value saved.
  - Writes through views: replay them as scatters into the base
    (`slice_scatter`, `as_strided_scatter`, ...) and regenerate the view
    from the base, which needs view tracking in the trace (today every
    tensor arg of a view op is conservatively marked aliased).
  - Inputs aliasing each other (the same storage passed twice): guard on
    it, or fall back.
  - `out=` overloads and ops writing several args.
  - RNG: random ops take a seed and offset as graph inputs (philox, as
    AOTAutograd's functionalized RNG ops do), so the backward's recompute
    draws the same numbers and the generator's state advances as eager's
    would; CPU's generator isn't philox, so CPU may need a different scheme
    or no recompute of random ops.
  Codegen will need the same for graphs without autograd.
- A block that gives up only stops bootstrapping in the Prepared it ran
  under; a Plan swap copies the cleared flags by block id, which holds only
  while swaps keep block ids (add_training does; optimize renumbers).
- AOTAutograd behaviors not yet covered (`torch/_functorch/_aot_autograd/`):
  - Inference vs training graph: when inputs require grad but no output
    does, AOT builds an inference graph. Inference graphs run under
    `no_grad` and training graphs under `enable_grad`.
  - Output aliasing (`collect_metadata_analysis.py`,
    `gen_alias_from_base`):
    - Each output is classified: an alias of an input or of an
      intermediate, an intermediate base, or an `_unsafe_view`.
    - Aliases are rebuilt outside the node.
    - Views coming from one `unbind`/`split` call are treated as not
      aliasing each other.
    - `mark_non_differentiable` is called on outputs that don't require
      grad.
  - Input mutations beyond the Functionalization list below:
    - Metadata-only mutations (`set_`, `resize_`, `as_strided_`, `t_`).
    - Mutations under `no_grad` or hidden from autograd.
    - The version-counter bump on write-back.
    - Gradient through the write-back: the mutated input that requires
      grad gets a tangent, and its pre-mutation value is cloned.
    - Mutations that cancel out are ignored.
  - Duplicate inputs and aliased inputs: duplicates are deduplicated behind
    a `DuplicateInputs` guard, and mutated aliased inputs get a synthetic
    base (`AOTDedupeWrapper`, `AOTSyntheticBaseWrapper`).
  - The node's own handling:
    - Inputs whose gradient is always None are detached.
    - Saved views are detached to break reference cycles.
    - Grads arrive in a boxed list so each can be freed early.
    - Donated buffers.
  - Mutations during backward (Float8-style buffer updates), and a tangent
    mutated by the backward.
  - Backward environment:
    - Autocast in backward (`backward_pass_autocast`).
    - Contextvars re-entered.
    - An error when the deterministic setting differs from the forward's.
    - An error when the backward runs under vmap or grad.
  - Effect tokens (`with_effects`, `torch/_higher_order_ops/effects.py`)
    keep side-effecting ops (print, torchbind) ordered in a graph.
  - Mutable custom ops become `auto_functionalized`; the first call checks
    that a custom op doesn't return an alias.
- Partitioner gaps vs `torch/_functorch/partitioners.py`:
  - CSE before partitioning.
  - Ban-recompute heuristics: reductions, compute-heavy ops, distance from
    the backward.
  - Params are free to save.
  - Collectives are always saved, with the decision synced across ranks.
  - The backward is reordered to mimic the autograd engine.
  - A memory budget (knapsack), CPU offload of activations, and SAC
    policies (`CheckpointPolicy`).
- A training version's exceptions go to the block's handler only when the
  original version raises into a single handler taking params; otherwise
  (or when the trace isn't supported) the block gets no training version and
  stops bootstrapping, keeping eager per-op autograd.
- Saved values hold Python references and their versions are checked by
  hand at backward, not through SavedVariable, so saved_tensors_hooks and
  pack/unpack aren't supported (guarded by `NoSavedTensorHooks`).
- The CfgBackward node has no `compiled_args`, so compiled autograd isn't
  supported. A backward on a CUDA engine thread would run the Plan swap
  (Python imports, op resolution) there; only CPU has been tested.
- In-place ops on returned views raise (`IN_CUSTOM_FUNCTION` creation meta,
  from `_wrap_outputs`) where eager wouldn't.
- Mutations of leaves that require grad, `retain_grad`, hooks on tensors,
  and `.grad` reads/writes inside optimized code aren't considered.
- The joint keeps `aten::detach` ops autograd inserted when saving; a
  simplification pass could drop them.
- Use cases to cover, roughly in priority order:
  1. Compiled forward, eager `loss.backward()` outside. Dynamo/AOTAutograd
     compile the backward ahead of time from a joint graph and wrap both in
     an autograd.Function the engine calls. The common case.
  2. Inference (`no_grad`/`inference_mode`): no backward, but grad mode must
     be guarded so the same code is specialized per mode.
  3. `.backward()` inside the compiled function. Dynamo graph-breaks there;
     compiled autograd (`torch._dynamo.compiled_autograd`) traces the
     backward at runtime instead, capturing hooks, accumulation and DDP/FSDP
     comms. We run the whole frame, so `.backward()` isn't a break for us:
     we could hand the engine one compiled backward for all we captured.
  4. Optimizer step (`optimizer.step()`, `zero_grad`), usually compiled as
     its own graph of foreach/fused updates.
  5. `torch.autograd.grad` (returns grads instead of filling `.grad`), often
     with `create_graph=True`: gradient penalties, physics-informed losses.
  6. Higher-order gradients: double backward, Hessian-vector products,
     meta-learning. The backward graph must itself be differentiable.
  7. `torch.func` transforms (`grad`, `vjp`, `jvp`, `vmap`, `jacrev`,
     `jacfwd`); forward-mode AD (dual tensors) is a separate mechanism.
  8. User `autograd.Function`s with their own forward and backward.
  9. Activation checkpointing (`torch.utils.checkpoint`): backward
     recomputes parts of the forward.
  10. Hooks and accumulation: `register_hook`, post-accumulate-grad hooks
      (FSDP/DDP, optimizer-in-backward), micro-batch accumulation.
  11. Several compiled regions joined by eager code, each with its own
      backward; partial `requires_grad` and `detach()` (AOTAutograd
      specializes on which inputs need grad).
  12. AMP training with `GradScaler` (scaled loss, unscale, inf check).

### Design: lazy backward (implemented, see above)

- Why: Dynamo/AOTAutograd trace the joint graph and partition it when the
  forward compiles (only Inductor's backward codegen is lazy). So they pay
  for a joint graph even for code that never runs backward, and must guess
  the incoming gradients (tangents): strides/memory format (wrong guesses
  are coerced with a copy every step), subclass metadata (DTensor
  placements; `__coerce_same_metadata_as_tangent__`), which outputs get a
  gradient (missing ones are materialized as zeros), and how backward is
  called (no double backward).
- Everything is in Rust: joint graph construction, functionalization and
  the min-cut partitioner are rewritten, not AOTAutograd reused. Forward and
  backward are both CFGs in our IR, run by the executor.
- Pure blocks not yet differentiated run eager ops under
  `saved_tensors_hooks`, like non-reentrant checkpointing: pack keeps only a
  token for the block, unpack recomputes the block from its saved inputs.
  This only works for pure blocks (re-running glue would repeat side
  effects), and pure blocks only exist after observation, so the first
  observed runs still save eager's activations; the warmup stance avoids
  that. Saving everything (plain eager) stays an option.
- Must be saved or captured even then: RNG state (seed/offset, restored on
  recompute, like checkpoint's `preserve_rng_state`); outputs of collectives
  (can't run twice); optionally outputs of nondeterministic kernels
  (atomics, unstable sort/topk ties) so gradients match the forward exactly.
  A pure block reads no other outside state by construction.
- Inputs modified in place after the forward make recompute raise (version
  counters) where eager may not. Dynamo's partitioner and checkpointing have
  the same gap.
- On the first backward of a block: observe the actual tangents, build the
  joint graph from the forward CFG, run the min-cut partitioner, and swap in
  a new forward (saving the chosen set) and backward as a new Plan.
- Each forward's autograd node holds the Plan it ran with, so forwards still
  pending (micro-batch accumulation) backward through their own plan:
  recompute for old ones, the min-cut pair for new ones.
- Guards: the forward on grad mode and which inputs require grad; the
  backward on tangent layout/subclass, which outputs got gradients, and
  `create_graph`. A backward guard miss builds another backward variant from
  the saved set, or falls back to recompute from saved inputs when the saved
  set isn't enough (e.g. double backward).
- Mechanics: a custom autograd node with an edge per input requiring grad,
  set as the outputs' `grad_fn`; outputs aliasing inputs need view handling
  (AOTAutograd regenerates them as views).
- Wins over Dynamo: inference without `no_grad` never builds a joint graph;
  outputs that never get gradients drop their backward paths; tangents are
  exact; `.backward()` inside a frame we run isn't a graph break, so forward
  and backward can be compiled together.
- Decisions:
  - Per-op derivatives are recorded, not transcribed from derivatives.yaml:
    run the op's backward through torch's autograd (on fakes, or during the
    observed real backward) with our op hook listening, and turn the aten
    ops seen into IR. This is what AOTAutograd does with make_fx. To check:
    that PreDispatch sees ops the engine runs (grad mode off, no
    `create_graph`), and that the include bit reaches CUDA engine threads
    (the engine copies thread-local state). Hand-written Rust formulas for
    hot ops can come later.
  - Pure regions are always one basic block: chains merge, branches on
    guarded conditions are specialized, short stable loops are unrolled, and
    a branch on tensor data or a rolled loop is glue around single-block
    pure bodies. A rolled loop's body gets an autograd node per iteration,
    so the engine's graph is the per-iteration tape and no reversed loop is
    generated (cost: per-iteration overhead, no cross-iteration fusion).
  - The joint trace is recorded with a Python TorchDispatchMode (as make_fx
    does), which costs a Python round-trip per op but runs once per block.
    Long term, a native thread-local hook in `pythonFallback` (like
    `set_pre_dispatch_hook` at PreDispatch) would trace without Python.
  - So functionalization is per block: in-place ops become out-of-place with
    a new SSA value, writes through views become scatters into the base
    (`slice_scatter`, ...), and mutated inputs are written back in the glue.
  - Torch's autograd engine stays: it calls our C++ node, which runs our
    backward CFG and uses SavedVariable for version-counter checks. Only the
    torch.compile layers are rewritten; base PyTorch C++ changes only where
    compile3 needs a hook.

### Design: warmup stance (proposed)

- A context (`with torch._rust.warmup(): train_step(real_batch)`) in which
  the executor runs everything fake, so the min-cut forward/backward pair is
  built before the first real step, on the user's own model and inputs.
  Same objects, no copies, no memory or compute. Covers the whole step
  (loss, `.backward()`, optimizer), since returned fakes can't leave it.
- Run under `FakeTensorMode(allow_non_fake_inputs=True)` so real tensors
  (inputs, params, buffers, globals) are faked as ops see them, through its
  memoizing converter.
- Types must recognize FakeTensor and record it as the tensor it stands for
  (device, sizes, strides); today it's an opaque Python-dispatch tensor that
  exact-`torch.Tensor` guards reject.
- Side effects run for real, so warmup needs an undo log: stores into
  objects that existed before warmup (`StoreAttr`, subscript stores,
  globals, `.grad`, optimizer state created lazily on the first step,
  BatchNorm counters) are recorded and restored when warmup ends, so no
  fake leaks into the model. Other effects (I/O, logging, C extensions)
  just happen.
- Data-dependent ops (`.item()`, `bool(t)`, `.tolist()`, `nonzero`) can't
  produce a value: abort the frame's warmup there, roll back, and keep the
  observations made so far.
- Warmup must run enough steps to cross whatever triggers optimization.

## Tensor subclasses (not supported yet)

- Aten replacement guards on exact `torch.Tensor`/`nn.Parameter`, so any
  subclass takes the miss path and runs the origin's original ops.
- Types only recognize tensors by exact type (`tensor_by_type`), and
  Python-dispatch tensors (`__torch_dispatch__`, FakeTensor) only report
  dtype and requires_grad.
- Needed: `__torch_function__` subclasses (run their override, or inline it
  as Dynamo does), and traceable wrapper subclasses (DTensor, NJT,
  `__tensor_flatten__`/`__tensor_unflatten__`), which Dynamo/AOTAutograd
  desugar into their inner tensors.
- Other values Dynamo handles specially that a region can't hold yet:
  Python `random` (its state captured in the graph), NumPy
  (`trace_numpy`), sparse and nested tensors, TorchScript/torchbind and
  opaque objects, and async collective tensors (`trigger_wait`).

## Other torch.compile features not supported yet

Guards and semantics (needed before optimized code can be trusted):

- Global-state guards: default dtype, deterministic algorithms, autocast
  state, grad mode, active TorchFunction/TorchDispatch modes, dispatch-key
  TLS. Only "no TorchFunctionMode active" is checked today.
- Guards on globals and attributes a pure block depends on (dict versions,
  object identity), as Dynamo installs. Movable calls are unguarded.
- Mutation and aliasing: in-place ops on inputs, views, shared storage.
  AOTAutograd functionalizes and replays input mutations after the graph.
  Pure blocks run eagerly so this is fine today, but codegen needs it (see
  Functionalization under Autograd).
- Tensor metadata: strides, contiguity, storage offset and layout in types
  and guards (Dynamo guards on strides).
- RNG: random ops in an optimized region need the RNG state handled (philox
  functionalization in AOTAutograd; see Functionalization under Autograd).
- Program changes after optimization must still behave as in CPython. Not
  covered yet:
  - A class changes (a method is added or replaced, or its bases or
    `__dict__` change).
  - A function's `__code__`, `__closure__` or `__kwdefaults__` is rebound
    (only `__defaults__` is checked today).
  - A weakref target dies.
  - An attribute appears on, or disappears from, an instance.
  - A partly consumed iterator is passed in.
  - The same tensor is passed twice, or inputs overlap in storage.
  - `.grad` is replaced, the current stream or the default device changes,
    or the region is called under vmap or grad.

  The mechanism is guards: see the kinds in `GuardBuilder`
  (`torch/_dynamo/guards.py`).
- Entering an optimized region should be cheap. Our guards run tens of
  reads and checks per call (each read once, but nothing folded), while Dynamo evaluates guards in a C++
  tree, shares checks across cache entries, and skips subtrees whose dict
  tags match.
- Code that sees many different inputs mustn't keep recompiling:
  - Compiles are bounded per block and per Module instance (Dynamo's
    `cache_size.py`), with a defined fallback at the bound (stay
    unoptimized, or error).
  - Users can trade safety for speed by dropping guards
    (`guard_filter_fn`, the `skip_guard_on_*_unsafe` presets).

Compiling:

- Varying sizes and scalars shouldn't compile again for each new value:
  - Today a Python scalar is guarded by its value, and types only record
    size intervals.
  - Mechanisms: symbolic sizes, unspecialized ints and floats
    (`specialize_int`, `specialize_float`, float tensorification),
    automatic dynamic after a recompile, and unbacked symints for
    data-dependent ops (`.item()`, `nonzero`).
  - The decision should carry over to the next run (PGO,
    `torch/_dynamo/pgo.py`).
  - User controls: `mark_dynamic(min, max)`, `mark_unbacked`,
    `mark_static`, `maybe_mark_dynamic`, `mark_static_address`,
    `torch._check*`, `guard_or_false`, and overrides per source through
    env vars (`torch/compiler/config.py`).
- A backend interface: lower a pure block to an FX graph so Inductor and
  other `backend=` choices plug in (code generation: see QUESTIONS.md).
- Tracing with fake tensors instead of running ops, which needs meta/fake
  impls, including for custom ops (`torch.library.custom_op`).
- Higher-order ops (`cond`, `while_loop`, `checkpoint`, `flex_attention`)
  as units rather than the aten ops inside them (see the op hook section).

Code generation (target still open, see QUESTIONS.md; Inductor,
`torch/_inductor/`, is the reference for what it must provide):

- Fewer kernels: fuse pointwise ops, reductions, and epilogues and
  prologues around matmul and conv.
- Fast kernels for the hardware: autotune matmul and conv templates and
  kernel configs, and cache the results.
- Less peak memory: buffer planning and reuse, in-place updates, and
  ordering for peak memory.
- Layouts: channels-last for convs and padding for alignment, while output
  strides still match eager's.
- Low launch overhead: CUDA graphs (below), static kernel launchers.
- Results that match eager: eager's RNG numbers, optional emulation of
  eager's precision casts, a deterministic mode, and debug asserts on sizes
  and NaNs.
- Inference: params frozen into constants, then constant folding (conv-bn,
  linear-binary).
- An op set to generate from: decomposition tables (core aten, where
  CompositeImplicit ops decompose), and a fallback to aten for ops it can't
  generate.
- User kernels in a region: Triton kernels (`triton_op`, `wrap_triton`)
  and custom ops, through their fake impls and layout constraints.
- Devices: CPU, CUDA, ROCm, XPU, MPS. Only CPU has been tried.
- Compile latency: compile in parallel or in subprocess workers so that
  optimizing doesn't stall the step.
- Finding a broken pass: bisect which pass breaks accuracy
  (`compiler_bisector`).

Model-level:

- nn.Module: parameters and buffers as graph inputs, guards on `training`,
  forward/pre-forward/backward hooks.
- Autocast/AMP regions.
- Distributed: DDP bucketing hooks, FSDP, functional collectives, DTensor.
  Also:
  - Ranks must agree on guard misses (guard collectives) and on
    dynamic-shape decisions (compiler collectives); otherwise they diverge
    and hang.
  - `compile_on_one_rank`.
- Compiled autograd and compiled optimizer steps (`torch.optim`).
- CUDA streams/events in captured code; `max-autotune`.
- CUDA graphs (`mode="reduce-overhead"`): capture an optimized region once,
  replay it to cut launch overhead. Needs static input addresses (copy
  inputs into static buffers, or guard on pointers for params/buffers), a
  private memory pool shared between forward and backward graphs, warmup
  runs before capture, one graph per shape set (dynamic shapes), and output
  invalidation (a replay overwrites the previous run's outputs; Inductor's
  cudagraph trees detect stale uses). Anything that syncs with the CPU,
  allocates outside the pool, or runs CPU ops can't be captured, and RNG
  needs graph-safe philox state.

Controls and tooling:

- API surface: `fullgraph=True` (error on graph break),
  `torch.compiler.disable`, `allow_in_graph`, `torch.compiler.is_compiling()`,
  recompile limits.
- Diagnostics: graph-break and recompile reasons (`torch._dynamo.explain`),
  `TORCH_LOGS` and `trace_structured` for tlparse, the minifier.
- Caching compiled artifacts across processes (like the FX graph and
  AOTAutograd caches), and `torch.export`-style ahead-of-time output.
- `torch.profiler` / `record_function` inside executed code (separate from
  the `sys.monitoring` refusal).
- `set_stance` (`force_eager`, `eager_on_recompile`, `fail_on_recompile`,
  `eager_then_compile`).
- `suppress_errors`: an internal error in the optimizer falls back to
  unoptimized code. `TORCH_COMPILE_DISABLE`.
- `torch.compiler.reset` and `remove_from_cache`.
- `disable(recursive=False)` and skip rules (`trace_rules` skipfiles):
  which code the executor runs and which it leaves to CPython. Torch and
  stdlib internals may be cheaper left to CPython.
- `assume_constant_result`, `nonstrict_trace`, `leaf_function`,
  `substitute_in_graph`, `disallow_in_graph`.
- `nested_compile_region` / `invoke_subgraph`: compile a repeated layer
  once (compare per-site clones, which compile it per call site).
- `error_on_graph_break` and `graph_break()` called from user code.
- What `is_compiling` / `is_exporting` return under the executor.
- More diagnostics:
  - Region-end reasons with stable IDs, like Dynamo's graph-break
    registry.
  - A guard-miss reason log (`recompiles`).
  - Compile IDs, `CompilationMetrics`, `dynamo_timed`.
  - `comptime` and `fake_tensor_crossref`.

## Type lattice (exec/types.rs)

- Every value-producing instr's result and every block param is observed when
  it's defined; the object's Ty is joined into the value's. Observing never
  runs Python code: only exact builtin types are looked into, and tensors are
  recognized by type (`tensor_by_type`, not isinstance).
- A mutable container is a snapshot at definition: `%0 = BUILD_LIST 0` stays
  `list[0]` even though the list grows later; each later load records the
  length at that point.
- Containers are recorded by kind and length only; tuples of up to 16
  elements per position, one level deep. Element facts come from the values
  read out of a container (subscripts, iteration, unpacking), which are
  observed where they're read. An earlier full scan of every container at
  every observation made the CPython suite ~5x slower overall, with many
  modules timing out.
- Classes are identified by address without a reference, so a freed class's
  address reused by another conflates them.
- Tensors: dtype, device, requires_grad, per-dim size intervals. No strides,
  layout or contiguity yet. Python-dispatch tensors (subclasses with
  `__torch_dispatch__`, FakeTensor) only get dtype and requires_grad, since
  their device and sizes can be computed in Python; symbolic sizes are unknown.
- Strings keep a value up to 64 chars (read by code point:
  PyUnicode_AsUTF8AndSize caches a UTF-8 copy in the str, which
  sys.getsizeof sees); floats are type only.

## Testing

- The CPython test-suite comparison harness is
  `torch/_rust/tools/compare_cpython.py` (usage in its docstring). The
  18-module set the commits' test plans use:
  `test_gc test_weakref test_listcomps test_optparse test_finalization
  test_array test_struct test_exceptions test_generators test_scope test_with
  test_raise test_contextlib test_dataclasses test_functools test_math
  test_float test_int`.
- Modules not yet covered by the sweep:
  - test_threading, which times out in both modes in the harness;
  - multiprocessing, network/ssl, GUI and tracing modules;
  - packages with their own sub-test layout (test_asyncio, test_importlib,
    test_email, test_unittest, test_sqlite3, test_ctypes).
- Torch-level coverage, not run yet:
  - OpInfo through optimized and trained blocks (modeled on
    `test/inductor/test_torchinductor_opinfo.py`).
  - Dynamo's own tests (`test/dynamo`) under the executor. Dynamo already
    runs CPython's tests too (`test/cpython/v3_13`).
  - The torchbench, HuggingFace and timm suites (`benchmarks/dynamo/`) for
    accuracy, coverage and speed.
  - Expected-failure lists with one file per test
    (`test/dynamo_expected_failures/`).
  - A `CompileCounter`-style helper.
- Load-independent perf (2026-11-15): agent_space/bench/run.sh MODEL MODE
  counts user-space instructions per training step with `perf stat` (only
  around the measured loop, through perf's control fifo) and prints the min
  of 15 x 20 steps, pinned to one core; instruction counts repeat to 0.01M
  on the loaded dev machine, wall times swing. Million instructions per
  step, single-threaded (executor elide/autograd as off/on bits):
  mlp eager 0.75, executor 00 1.59, 01 1.75, 10 1.53, 11 1.66, compile
  eager 0.75, inductor 0.68; block 1.22, 2.18, 2.46, 2.13, 2.34, 1.23,
  1.01; attn 1.84, 3.02, 3.48, 2.89, 2.88, 1.84, 1.16; conv 1.34, 2.43,
  2.68, 2.36, 2.45, 1.33, 1.17; transformer 3.48, 5.74, 6.11, 5.62, 5.96,
  3.36, 2.16 (agent_space/bench/grid1.txt). So the executor runs 1.6-2.1x
  eager's instructions; elision saves 2-5%; autograd on still adds
  instructions everywhere except elided attn (the plan's bar: no worse).
  With guards as checks on shared reads (2026-11-16,
  agent_space/hoist/grid_guards.txt; eager and compile unchanged), the
  executor's 00/01/10/11: mlp 1.52, 1.74, 1.41, 1.59; block 2.11, 2.38,
  1.99, 2.20; attn 2.94, 3.39, 2.70, 2.69; conv 2.33, 2.66, 2.19, 2.46;
  transformer 5.55, 6.10, 5.28, 5.78. Elision now saves 5-8% (it was 2-5%);
  autograd on still adds instructions except on elided attn.
  torch.compile's eager backend matches eager's count but takes 10-25%
  longer.
- Perf comparisons: agent_space/autograd/bench.py (one mode per process,
  median of 7 x 100 steps) against eager and torch.compile with the eager
  and inductor backends. Run with OMP_NUM_THREADS=1 on a shared machine:
  with the intra-op pool, load from other jobs swung eager by up to 2x.
  Inductor's on-disk caches make its first call 0.5-8s warm vs ~30s cold
  (TORCHINDUCTOR_FORCE_DISABLE_CACHES=1). Single-threaded, after recording
  stopped, the executor is 1.4-1.8x eager on those models. With pinned calls
  every compute block trains, and autograd on costs 0.06-0.4 ms/step more
  than off (mlp 0.62 -> 0.89, attn 1.03 -> 1.30, transformer 1.82 -> 2.22):
  one-op training versions replace aten's C++ backward nodes with
  CfgBackward round trips through Python and the executor.
- Plain Python is 4-13x slower per bytecode under the executor than under
  CPython, with recording off and optimizing off (agent_space/inline/
  interp.py, single-threaded): a loop iteration with an attribute load and
  two adds 273ns vs 70ns, a call to a two-arg function 420ns vs 59ns, a
  generator resume 535ns vs 41ns. Optimizing makes the loop slower (336ns:
  guards and glue transitions). In a call-heavy profile, `run_blocks`'s own
  dispatch is 25% of samples and per-frame setup (eval_frame,
  tr_frame_enter, Bufs take/give, clear_at_exit, take_exit,
  tr_tracing_active) about 10%. This, not frame boundaries, is most of the
  executor's gap to eager on the training-step benchmarks: the MLP step's
  overhead splits into forward +0.03ms, `zero_grad` +0.04ms and
  `model.parameters()` +0.04ms (both generator-heavy: 22 `named_modules` and
  16 `Tensor.__hash__` frames per step), and the loss/backward Python
  wrappers +0.08ms (agent_space/inline/parts.py).
- Each generator call enters CPython's
  evaluator for its prologue (`COPY_FREE_VARS`/`MAKE_CELL`/`RETURN_GENERATOR`):
  `_Py_MakeCoro` isn't exported, so CPython builds the generator object. No
  user code runs there. Owning this would mean transcribing `genobject.c`
  internals.

## Blocks, versions and outlining

- A block has one shape (params, a pure terminator over output indices, a
  handler list) and any number of versions (guard, straight-line body,
  outputs, unwind segments). The first version whose guard holds runs;
  otherwise the block's `origin` (an outlined `Cfg`) runs and leaves with the
  outputs. Values are local to a block: only params and edge args cross.
- Outlining (`cfg/outline.rs`) only handles chains (jump to a block with no
  other way in), merged into one version or, for testing the miss path, into
  just the origin. It's off by default: `bytecode_to_cfg(outline=...)` and
  `torch._rust._set_cfg_outline(...)` turn it on. Not yet: multi-exit regions
  (needs a `Switch` terminator), specialization, regions holding a resume
  point (a generator can't suspend inside an origin).
- Raising is a terminator (`Raise`/`Reraise`), so a block raises whichever
  version runs. Its handler is `TermUnwind`, which has no `Outer` yet: an
  origin always ends in `Leave` and the owner raises. A region containing a
  raising block (not just ending in one) will need it.
- Simplify no longer substitutes non-reserved values across blocks, so stack
  values (e.g. a `with`'s `__exit__`) travel through params at every block
  boundary. Constants could be rematerialized in the block instead.
- `Block::origin` is an `Arc<Cfg>` (the `Compiled` holding it is shared via
  `co_extra`).

## Converter

- CPython 3.12 only (3.13/3.14: see QUESTIONS.md).
- Asynchronous exceptions are only modeled where the eval breaker runs
  (`Resume`, `EvalBreaker`, plus the executor's checks after calls). `MemoryError` isn't
  modeled in can-raise.
- Code generation: see QUESTIONS.md.
