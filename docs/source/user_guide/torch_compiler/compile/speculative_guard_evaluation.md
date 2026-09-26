# Speculative Guard Evaluation

Status: experimental design for maintainer feedback. The implementation is
disabled by default. Inductor opts in only for a narrow subset of CUDA
inference graphs.

## Motivation

On every compiled-frame invocation, Dynamo selects a cache entry by evaluating
its guards before executing the entry's transformed bytecode and compiled
artifact. For latency-sensitive inference, this serializes CPU validation with
device execution and can leave an otherwise idle GPU waiting.

Qwen3-8B measurements on an NVIDIA A100 found approximately 2 ms in the
profiled Dynamo cache-lookup range. A mutation-free prefill experiment reduced
median synchronized latency from 20.01 ms to 16.76 ms by launching first and
evaluating guards while the device was active. A single-token decode experiment
reduced hit latency from 14.15 ms to 12.95 ms. These are latency results for an
idle-device setup; continuously queued serving can already hide much of this
CPU work.

The proposal does not remove guard evaluation. It changes the ordering for a
backend-certified artifact:

```text
normal:      guard lookup -> transformed bytecode -> device artifact
speculative: safe launch  -> guard lookup ---------> commit -> cached bytecode
                                      failure -----> abort -> normal path
```

## Scope

This is an opt-in mechanism for backends that can provide a transactionally
isolated invocation. Dynamo does not infer whether an arbitrary compiled graph
is safe to launch. The initial implementation:

- is disabled by default;
- requires a fullgraph compilation;
- excludes export, packaging, inlined `__torch_function__` subclass dispatch,
  backward state, and Dynamo-recorded external Python side effects;
- preserves normal cache lookup and guard-complete-hook behavior; and
- lets Inductor opt in only when AOTAutograd and Inductor metadata satisfy the
  restrictions described below.

The general idea of running transformed Python concurrently with guards is out
of scope. Python execution would contend for the GIL, CUDA work cannot be
cancelled once submitted, and transformed bytecode can expose side effects
before validation finishes.

## Backend contract

An opted-in backend callable exposes a private
`_torchdynamo_speculation_descriptor` attribute. Dynamo stores it on the
corresponding cache entry. The interface has two phases:

```python
class SpeculationDescriptor(Protocol):
    def launch(
        self, frame_locals: dict[str, object]
    ) -> SpeculationTicket | None: ...


class SpeculationTicket(Protocol):
    def commit(self) -> object: ...
    def abort(self) -> None: ...


class BytecodeSpeculationTicket(Protocol):
    def commit_to_cached_code(self) -> None: ...
    def finish(self) -> None: ...
    def abort(self) -> None: ...
```

`launch()` must synchronously validate every property required to execute the
specific artifact safely. This includes tensor ABI, device and stream state,
aliasing, captured-object lifetime, and backend-internal specialization not
represented by the outer Dynamo cache entry. Returning `None` declines
speculation and continues through the normal path.

A returned ticket must keep inputs and private outputs alive. Work must remain
unobservable until commit. A direct ticket's `commit()` returns the exact Python
frame result, including output adaptation. A bytecode ticket instead uses
`commit_to_cached_code()` to arm the installed backend callable. Dynamo then
runs the normal cached bytecode, whose backend call consumes the speculative
output, and calls `finish()` to verify consumption. This second form preserves
Dynamo's arbitrary Python output reconstruction without asking Inductor to
reverse engineer it.

`abort()` must drain or correctly order discarded work and restore every
speculative mutation before the normal path executes. None of the ticket
operations may invoke user code. Implementations must reject overlapping
launches unless their state is concurrency-safe.

An implementation that enqueues work and then raises from `launch()` must clean
up before raising because Dynamo has not received a ticket to abort.

## Runtime algorithm

The evaluator first follows existing guardless and precompile fast paths. When
a code object has an eligible descriptor-bearing entry, it selects the first
backend-compatible entry in normal cache order as the prediction.

The evaluator then:

1. Retains the predicted guard manager and descriptor.
2. Copies frame locals so the descriptor cannot change values observed by
   guards.
3. Calls `launch()` with Dynamo evaluation disabled recursively.
4. Re-fetches the code object's cache state in case `launch()` reset it.
5. Performs the existing complete cache lookup and guard-complete hook.
6. Commits only when lookup selected the predicted guard manager by identity.
7. For a bytecode ticket, runs the selected transformed bytecode and verifies
   that its backend call consumed the speculative result.
8. Otherwise aborts before compilation, eager execution, or another cached
   entry runs.

Invalidation clears the descriptor. A per-code-object descriptor count avoids
an additional cache scan for ordinary entries. Launch, lookup, hook, commit,
and abort exceptions preserve the primary exception and restore the eval-frame
callback. Kineto ranges identify speculative launch, commit, and abort.

## Side effects and Inductor integration

Dynamo can reject Python mutations that its bytecode generator would replay,
but it cannot prove that a backend artifact is transactional. Inductor and
AOTAutograd can have additional effects:

- mutation of graph inputs, parameters, buffers, or storage metadata;
- output/input aliasing;
- RNG-state advancement;
- collectives, effect tokens, host callbacks, and custom operators;
- CUDA Graph Trees static-buffer and output-liveness state; and
- asynchronous device assertions or faults.

The initial Inductor integration emits descriptors only for single-device CUDA
inference artifacts without input mutation, AOTAutograd effect tokens, RNG,
CUDA Graph Trees, graph partitions, or unsupported operators. Dynamic inputs
must be direct frame-local tensors with exactly matching shape, stride, storage
offset, dtype, device, layout, and tensor flags. Module parameters and buffers
are weakly captured from the first normal invocation and checked by both exact
tensor metadata and version counter before speculative use. A ticket holds
temporary strong references while work is in flight. If cached bytecode later
supplies different parameter objects, the adapter discards the speculative
result and invokes Inductor normally.

Admission has two stages. The AOT graph rejects unsupported namespaces,
mutation, nondeterminism, effectful tags, and explicit assertions. After
lowering, the generated artifact is rejected if any scheduler node contains a
device assertion or an indirect memory access. The second check covers
operators such as `aten.gather` whose schema is not tagged as dynamic-output
but whose generated Triton kernel performs data-dependent bounds checks.

The eligibility bit is stored with `CompiledFxGraph`, so local FX and
AOTAutograd cache hits retain the same decision. The one-shot result is
thread-local. On a hit, normal transformed bytecode preserves Python return
structure and consumes the result without launching Inductor twice. Each
launch records a CUDA event. Commit makes the current stream wait on that event;
abort synchronizes it before allowing a different graph or eager code to run.

The current safe subset rejects data-dependent output and indexing operations.
This matters for the motivating Qwen workload: its embedding lookup lowers
through `aten.index.Tensor`, whose generated CUDA kernel can issue a device
assert for an out-of-range token. A wrong-path asynchronous assertion cannot be
cancelled or rolled back after guards fail. A controlled experiment that
supplied the model-level invariant that token IDs were in range measured 19.61
ms baseline versus 17.61 ms speculative median latency, a 10.2% reduction. A
repeat after adding CUDA-event ordering measured 21.85 ms versus 19.72 ms, a
9.7% reduction. Its profile showed 2.40 ms in cache lookup, a 0.25 ms
speculative-launch CPU range, and a 0.063 ms commit range. After adding exact
metadata checks for every warmed parameter and buffer, two additional
30-sample alternating runs measured 25.88 ms versus 24.10 ms and 21.00 ms
versus 19.30 ms, reductions of 6.9% and 8.1%. Speculative launch took 0.27-0.30
ms of CPU time, guard lookup took 2.25-2.28 ms, and commit took 0.06-0.07 ms.
Run-to-run device variation is large enough that these results should be
treated as a range. They validate the overlap mechanism, but the unsafe
indexing waiver is not part of the implementation.

Useful LLM decode normally mutates a KV cache. That requires a separate runtime
contract rather than treating the artifact as pure. For an append-only cache,
the artifact can write an uncommitted slot and advance a private or reversible
cursor. Commit publishes the new logical length; abort restores the cursor and
orders the correct graph after the discarded write. The compiler/runtime must
prove that no reader observes the tentative slot and that fallback overwrites
it before reading. Copying the full cache is not expected to be profitable.

## Performance and failure cost

If launch-safety validation costs `S`, remaining guards cost `G`, host launch
costs `H`, device work costs `D`, and commit costs `C`, the hit paths are
approximately:

```text
normal:      S + G + H + D
speculative: S + H + max(G, D) + C
```

The maximum hit saving is `min(G, D) - C`. A wrong prediction executes discarded
work before the correct path and can roughly double device latency. In the
Qwen decode experiment, six hits committed and two MRU transitions aborted;
results and the complete KV cache matched the reference bitwise. Hit median was
12.95 ms, while the two misses were 26.46 ms and 69.20 ms. The large miss was a
lazy CUDA Graph Trees path-transition cost and demonstrates why p95/p99 impact
and prediction stability matter more than hit latency alone.

## Alternatives

The same stable-specialization workloads are also good candidates for less
speculative approaches:

- versioned guard leases or aggregated dependency epochs;
- direct outer CUDA Graph replay;
- fixed-size decode chunks with an active mask;
- a device-resident decode loop; or
- AOTInductor with a controlled input contract.

These approaches reduce or amortize validation instead of paying for wrong-path
execution. Speculative evaluation is useful only when it hides a measured
device-idle bubble and its hit-rate and tail-latency tradeoff beat those options.

## Proposed rollout

1. Land the disabled backend protocol and runtime mechanics only after API and
   failure-semantics review.
2. Review the initial restricted Inductor descriptor and the cached-bytecode
   result handoff.
3. Define a way to certify data-dependent kernel preconditions, such as valid
   token-index ranges, without a device synchronization.
4. Measure synchronized latency, queued throughput, prediction accuracy,
   wasted GPU work, and p50/p95/p99 latency.
5. Design an explicit append-only KV-cache transaction with the serving/cache
   runtime rather than recognizing model-specific patterns in Dynamo.
6. Consider broader enablement only after stress testing concurrency,
   invalidation, device faults, RNG, and CUDA Graph Trees interactions.

## Open questions

- Should launch admission be split into `prepare()` and `launch()` so failures
  cannot occur after enqueue but before a ticket exists?
- Should the prediction policy remain MRU or use per-entry hit statistics?
- How should a deployment disable already-captured descriptors at runtime?
- Which Inductor/AOT metadata is sufficient to certify a pure artifact?
- How should a serving runtime certify value-domain invariants such as token
  bounds so an indexing kernel is safe before the full guard lookup?
- Where should the KV-cache transaction interface live across compiler,
  serving runtime, and cache implementation boundaries?
