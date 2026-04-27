# FSDP2 HSDP: Options to Let RS Proceed Without Waiting for AR

## Context

The memory-accumulation fix (commit `2b889c47474`) adds a back-edge from `all_reduce_stream` to `reduce_scatter_stream`:

```python
# _fsdp_collectives.py:787-788
if post_reduce_stream is not reduce_scatter_stream:
    reduce_scatter_stream.wait_event(post_reduce_event)
```

This makes **iter K−1's RS wait for iter K's AR post-reduce** on the AR stream. It's needed because `reduce_output` is allocated on RS stream but used in-place by AR stream (RS, in-place AR, cast, `+=`), and the caching allocator only tracks the alloc stream — without the sync, a later RS-stream alloc could reuse the block while AR stream's post-reduce ops are still draining.

Empirical perf data (Llama-3 8B, 2×4 HSDP, 8× H100):
- `grad_accum=1`: HEAD 460 ms vs HEAD~1 468 ms — **HEAD 1.8% faster** (added wait is overcompensated by reduced allocator / NCCL scheduling pressure).
- `grad_accum=2`: HEAD 882 ms vs HEAD~1 883 ms — within noise.

So the back-edge is not currently a throughput problem. The options below are for if a future workload shows it biting.

---

## Options

### 1. `torch.Tensor.record_stream(all_reduce_stream)`

Right before `all_reduce_input = reduce_output`, call `reduce_output.record_stream(all_reduce_stream)`. The caching allocator now tracks that AR stream used this block; on free, the allocator won't reuse the block until its internal event on AR stream fires.

**Pro:** Zero explicit RS-stream sync cost. RS_{K−1} can launch immediately.
**Con:** Well-known footgun — `record_stream` inflates block liveness until the allocator's internal checkpoint on AR stream passes. Under AR-stream stalls, memory can balloon instead of serializing. Also has historical issues with cudagraph paths. Explicitly rejected by FSDP2 today (see the `(NCCL's default-sync … calls neither recordStream nor stashing…)` comment in `_fsdp_param_group.py:660-662`).

### 2. Allocate `reduce_output` on AR stream, not RS stream

Move the `reduce_scatter_comm.allocate(...)` under `with device_handle.stream(all_reduce_stream)`. The block's alloc stream then matches the stream doing post-reduce work, eliminating the cross-stream free race.

**Pro:** Architecturally clean; no explicit sync needed for this buffer.
**Con:** RS kernel also writes to the buffer during reduce-scatter, so RS still uses an AR-stream-allocated block. A single RS→AR sync (which already exists via `all_reduce_stream.wait_stream(reduce_scatter_stream)`) is enough; the block's free side is clean. But RS-stream's own alloc pattern changes, which may affect pool fragmentation.

### 3. Dedicated `MemPool` for RS output tensors

Override `ReduceScatter.allocate` to draw from a `torch.cuda.MemPool` with built-in cross-stream sync. `SymmMemReduceScatter` already uses this pattern (`_fsdp_collectives.py:111-114`).

**Pro:** Pool owns the invariant; no per-iter explicit sync; scales to multiple RS backends uniformly.
**Con:** New infra, more code to maintain, pool-sizing policy questions.

### 4. Move post-reduce ops back to RS stream

Run the cast / `_div_if_needed` / `+=` on RS stream instead of AR stream. Alloc stream then equals usage stream; back-edge unnecessary.

**Pro:** Simplest from a stream-tracking standpoint.
**Con:** Serializes post-reduce behind RS stream, blocking the next layer's RS kernel. Defeats the whole reason post-reduce was split to AR stream (to pipeline with next RS). Likely a throughput regression.

### 5. CPU-side `post_reduce_event.synchronize()` before next RS

Host-blocking sync before iter K−1's allocation.

**Pro:** Trivially correct.
**Con:** Stalls the host thread. Throughput disaster.

### 6. Higher priority on AR stream

Give AR stream higher priority so post-reduce drains fast enough that the back-edge's wait is empirically a no-op.

**Pro:** No correctness change; one-line tweak.
**Con:** Doesn't address the fundamental ordering constraint; just makes the wait usually non-blocking.

---

## Option Ranking

| option | complexity | correctness risk | throughput win | memory risk |
|---|---|---|---|---|
| 1. `record_stream` | low | low | high | **high (balloon)** |
| 2. alloc on AR stream | low-medium | low | medium-high | low |
| 3. dedicated `MemPool` | high | low | high | low |
| 4. post-reduce on RS stream | low | low | **negative** | low |
| 5. CPU sync | trivial | low | **negative** | low |
| 6. stream priority | trivial | none | low-medium | none |

**Recommendation:** don't ship any of these preemptively. File as a follow-up if a future profile shows AR→RS serialization biting on a real workload. If it does, **Option 2** is the first prototype candidate.

---

## Alternative Design: RS of Layer n Waits for AR of Layer n+2

This section sketches a hand-rolled sync scheme that keeps the explicit-wait-event model (Option 0 baseline) but defers the wait by one backward iteration, letting iter n's RS start without waiting for iter n+1's AR.

### Ordering recap

Backward processes layers in reverse: layer N−1 first, then N−2, …, down to layer 0. For a "current" layer n in backward:
- Layer n+1 was processed one iteration ago.
- Layer n+2 was processed two iterations ago.

Current back-edge: `RS_n` waits for `AR_{n+1}` (most recent).
Proposed: `RS_n` waits for `AR_{n+2}` (second-most-recent).

### Why it could work

Correctness invariant to preserve: *before RS stream re-allocates over a block previously used by AR stream, RS stream must have synced with AR stream's work on that block.*

Block lifetime for `reduce_output_K` (Llama-3 8B, RS dp_shard=4, 218 MB fp32):
- **Iter K**: allocated on RS stream, used by AR stream for in-place AR + cast + `+=`, `post_reduce_event_K` recorded on AR stream.
- **End of iter K**: `comm_ctx.all_reduce_state = (reduce_output_K, all_reduce_event_K)`.
- **Iter K−1**: at end of `post_backward`, `del prev_all_reduce_state` drops the last ref to `reduce_output_K`. Block returns to the RS-stream pool.
- **Iter K−2 or later**: the block may be reused by a future RS-stream alloc.

So the *earliest reuse* of `reduce_output_K`'s block is in iter K−2. That means the sync `RS_stream.wait_event(post_reduce_event_K)` does not need to be queued before iter K−1's alloc — queuing it before iter K−2's alloc is sufficient.

### Proposed state machine

Extend `FSDPCommContext` with a second-most-recent slot:

```python
class FSDPCommContext:
    ...
    # Most-recent and second-most-recent AR states.
    # Invariant after iter K's post_backward:
    #   all_reduce_state      = (reduce_output_K,   all_reduce_event_K)
    #   prev_all_reduce_state = (reduce_output_{K+1}, all_reduce_event_{K+1})
    all_reduce_state: AllReduceState | None = None
    prev_all_reduce_state: AllReduceState | None = None

    # post_reduce_event from the layer before the prev one — queued on RS stream
    # to unblock reuse of the K+2 block (freed 2 iters ago).
    prev_prev_post_reduce_event: torch.Event | None = None
```

Per-iter state transition (at start of `post_backward` for layer K):

```python
# Rotate: prev -> drop (freed now), all_reduce -> prev, new -> all_reduce
to_drop = self.comm_ctx.prev_all_reduce_state               # = state_K+2
self.comm_ctx.prev_all_reduce_state = self.comm_ctx.all_reduce_state  # = state_K+1
self.comm_ctx.all_reduce_state = None                       # will be set to state_K below

# The sync we need before iter K's RS can safely reuse blocks freed at iter K+1:
# RS must have waited on AR_{K+2}'s post-reduce. We queue that now, before RS_K alloc.
if self.comm_ctx.prev_prev_post_reduce_event is not None:
    self.comm_ctx.reduce_scatter_stream.wait_event(
        self.comm_ctx.prev_prev_post_reduce_event
    )

# ... run foreach_reduce for iter K (no wait_event(post_reduce_event_K) at the tail now) ...

# After foreach_reduce:
self.comm_ctx.all_reduce_state = AllReduceState(all_reduce_input_K, all_reduce_event_K)
# Save post_reduce_event_K to be queued as RS-wait in iter K-2's post_backward
self.comm_ctx.prev_prev_post_reduce_event = post_reduce_event_K  # rotated via another slot

# Drop state_{K+2}, which is the block now going to RS pool.
del to_drop
```

Modifications to `foreach_reduce`:

```python
# REMOVE the existing tail wait:
# if post_reduce_stream is not reduce_scatter_stream:
#     reduce_scatter_stream.wait_event(post_reduce_event)
```

(The caller now owns this sync, and queues it with a one-iteration delay.)

Drain at end of backward (`flush_all_reduce_state`) must drain both slots:

```python
def flush_all_reduce_state(self):
    for st in (self.all_reduce_state, self.prev_all_reduce_state):
        if st is not None and st.event is not None:
            self.device_handle.current_stream().wait_event(st.event)
    self.all_reduce_state = None
    self.prev_all_reduce_state = None
    # No need to wait on prev_prev_post_reduce_event: by definition, its
    # corresponding reduce_output block was already freed 2 iters ago.
    self.prev_prev_post_reduce_event = None
```

### Correctness argument

1. `reduce_output_K` stays referenced via `all_reduce_state → prev_all_reduce_state → to_drop` for two iterations after its allocation. Earliest free: end of iter K−2 (when it rotates into `to_drop` and gets `del`'d).
2. Earliest RS-stream reuse of the block: iter K−3's alloc.
3. Sync: `post_reduce_event_K` is queued as `reduce_scatter_stream.wait_event(...)` at the start of iter K−2's `post_backward` (one iter before the earliest possible reuse). Queued in-order on RS stream — any subsequent RS-stream work waits.
4. Therefore iter K−3's RS alloc is safe.

### What overlap is gained

| dependency | current | proposed |
|---|---|---|
| RS_{K} blocks on | AR_{K+1} post-reduce | AR_{K+2} post-reduce |
| Can AR_{K+1} still be in-flight when RS_K starts? | no | **yes** |
| Block free latency | 1 iter (end of iter K−1) | **2 iters (end of iter K−2)** |

AR_{K+1} and RS_K can now overlap. If AR latency ≈ one backward-layer step, that's roughly one extra layer of RS/AR pipelining.

### Costs

1. **One extra keep-alive slot.** Peak liveness grows from 2 → **3** `reduce_output` tensors (current + prev + prev_prev). For 218 MB buffers, that's +218 MB peak. Still O(1); still a far cry from the O(n_layers) = ~7 GB bug.
2. **One extra event slot (`prev_prev_post_reduce_event`).** Trivial memory.
3. **State-machine complexity.** The rotate-by-two pattern is subtler than the current rotate-by-one. Higher chance of bugs during future maintenance.
4. **Drain logic** must cover two states plus one event — three things to remember at `flush_all_reduce_state` instead of one.
5. **Same-dtype fresh case** (param-grad views alive) was already not a problem; the extra slot just adds one more refcount-redundant reference. No observable cost.

### Should this be implemented?

Probably not now — the measured back-edge cost is already near zero (HEAD is 1.8% *faster* than HEAD~1 at grad_accum=1, and identical at grad_accum=2). Extending the keep-alive from 2 → 3 buffers and the state machine from "rotate-1" to "rotate-2" costs memory and complexity for a speedup we haven't shown a need for.

Conditions under which it *would* be worth revisiting:
- A workload surfaces where AR stream latency is a measurable fraction of the per-layer backward gap (so AR→RS serialization actually blocks).
- The +218 MB (or equivalent per-workload) peak-memory cost is acceptable.
- Option 2 (alloc on AR stream) was already tried and didn't work.

### Alternate framing

This design is essentially a "rotate-2" variant of the existing "rotate-1" keep-alive. The generalization is *"rotate-k"*: extend liveness by k iterations, gain k layers of AR-vs-RS overlap, cost k extra buffers. The fix at HEAD picks k=1 (smallest that fixes the memory bug). Raising k is a knob to trade memory for overlap. Most workloads probably don't need k>1.

---

## Parameterized Design: rotate-k as User Config

Rather than hard-coding k=1 (HEAD) or k=2 (section above), expose the depth as a user-tunable knob. The concrete design below lets a user set k=1, 2, 3, 4, … without code changes, with k=1 being the current HEAD behavior.

### What the knob controls

- **k = 1** (default): RS_n waits for AR_{n+1}. Current HEAD behavior. 2 simultaneous `reduce_output` buffers alive at peak.
- **k = 2**: RS_n waits for AR_{n+2}. AR_{n+1} can overlap with RS_n. 3 simultaneous buffers.
- **k = d**: RS_n waits for AR_{n+d}. AR_{n+1} … AR_{n+d-1} can overlap with RS_n. (d+1) simultaneous buffers.
- **k = n_layers**: degenerate — equivalent to the original O(n_layers) bug. *Must be rejected at config time.*

### Data structures

Replace the two current slot variables on `FSDPCommContext` with two bounded deques:

```python
from collections import deque

class FSDPCommContext:
    # User-configurable; 1 = current HEAD-equivalent behavior. Must be in [1, n_layers).
    ar_rs_lookahead_depth: int = 1

    # Keep-alive queue: holds up to `depth + 1` AllReduceState entries.
    # Oldest entry's reduce_output is dropped (returned to RS pool) when the
    # queue overflows. Newest entry is the most recently completed layer's state.
    _all_reduce_state_queue: deque[AllReduceState]

    # Deferred-wait queue: holds up to `depth` post_reduce_event entries.
    # On each new post_backward, pop the oldest event (which corresponds to the
    # AR post-reduce from `depth` iters ago) and queue it as
    # reduce_scatter_stream.wait_event(...) BEFORE the new RS alloc.
    _pending_rs_wait_queue: deque[torch.Event]
```

Both deques have `maxlen` set from `ar_rs_lookahead_depth`:
- `_all_reduce_state_queue`: `maxlen = depth + 1`
- `_pending_rs_wait_queue`: `maxlen = depth`

### Per-iter state transition (layer K's post_backward)

Two subtle points in the sketch below, both of which matter independently of `depth`:

- `prev_all_reduce_event` is captured as the **most-recent** AR event (`queue[-1].event`), not the event of the state being popped. This preserves HEAD's AR-stream serialization semantics (`foreach_reduce`'s `all_reduce_stream.wait_event(prev_all_reduce_event)` always waits for the immediately previous iter's AR kernel). Using the popped state's event would wait for an event from `k` iters ago, diverging from HEAD whenever `depth >= 2` and even at `depth = 1`.
- `to_drop` is explicitly `del`'d **before** `foreach_reduce`. Safety for the freed block is already provided by step 1's queued `wait_event` on RS stream, so holding the popped state alive longer only adds peak memory. With this ordering, peak `reduce_output` liveness is `depth + 1` (as in the memory table below); if `to_drop` were held until function end, peak would be `depth + 2`.

```python
comm_ctx = self.comm_ctx
depth = comm_ctx.ar_rs_lookahead_depth

# 1. Drain deferred wait: if we have an event from `depth` iters ago, queue it
#    on RS stream so iter K's alloc (and all subsequent RS work) waits for AR.
if len(comm_ctx._pending_rs_wait_queue) == depth:
    oldest_event = comm_ctx._pending_rs_wait_queue.popleft()
    comm_ctx.reduce_scatter_stream.wait_event(oldest_event)

# 2. Capture prev_all_reduce_event: the MOST RECENT AR event (one iter ago).
#    Independent of `depth` — foreach_reduce's AR-stream wait should always
#    serialize against the immediately previous iter's AR, never against an
#    older event.
prev_all_reduce_event = (
    comm_ctx._all_reduce_state_queue[-1].event
    if comm_ctx._all_reduce_state_queue
    else None
)

# 3. Drop oldest keep-alive if queue is full, and free the block *now* (before
#    foreach_reduce). Safety for any later RS-stream reuse of this block is
#    provided by step 1's queued wait_event, so holding to_drop longer only
#    inflates peak liveness.
if len(comm_ctx._all_reduce_state_queue) == depth + 1:
    to_drop = comm_ctx._all_reduce_state_queue.popleft()
    del to_drop

# 4. Run foreach_reduce (which no longer queues the tail wait_event itself;
#    the caller owns that sync now via the deferred-wait queue).
(_, _, post_reduce_event, all_reduce_input, all_reduce_event, _) = foreach_reduce(
    ..., prev_all_reduce_event=prev_all_reduce_event,
)

# 5. Append new entries.
if all_reduce_input is not None:
    comm_ctx._all_reduce_state_queue.append(
        AllReduceState(all_reduce_input, all_reduce_event)
    )
comm_ctx._pending_rs_wait_queue.append(post_reduce_event)
```

### Changes to `foreach_reduce`

Remove the tail back-edge; the caller owns it now via the deferred-wait queue:

```python
# REMOVE from _fsdp_collectives.py:787-788:
# if post_reduce_stream is not reduce_scatter_stream:
#     reduce_scatter_stream.wait_event(post_reduce_event)
```

The `prev_all_reduce_event` wait inside `foreach_reduce` stays unchanged.

### Drain at end of backward

```python
def flush_all_reduce_state(self):
    # Drain all remaining keep-alive states (up to depth+1 of them).
    for st in self._all_reduce_state_queue:
        if st.event is not None:
            self.device_handle.current_stream().wait_event(st.event)
    self._all_reduce_state_queue.clear()
    # Drain any still-pending RS-stream waits so the RS pool is safe for the
    # optimizer step / next forward.
    for ev in self._pending_rs_wait_queue:
        self.reduce_scatter_stream.wait_event(ev)
    self._pending_rs_wait_queue.clear()
```

### Config API

Exposed via the `fully_shard` API (or `share_comm_ctx`, since the depth is a `comm_ctx`-level property):

```python
from torch.distributed.fsdp import fully_shard

fully_shard(
    model,
    mesh=mesh,
    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
    ar_rs_lookahead_depth=2,  # default 1; must be in [1, n_layers).
)
```

Validation at construction time:
- `depth >= 1`: else raise `ValueError`.
- `depth < n_layers`: warn if `depth >= n_layers // 2`, reject if `depth >= n_layers`. A `depth` ≥ `n_layers` regresses to the O(n_layers) memory bug we just fixed.
- Optional: environment-variable override `FSDP2_AR_RS_LOOKAHEAD_DEPTH` for easy experimentation.

### Memory / overlap tradeoff

For Llama-3 8B HSDP with dp_shard=4, fp32 reduce (218 MB per `reduce_output`):

| depth `k` | live `reduce_output` tensors | extra peak memory | layers of AR/RS overlap unlocked |
|---:|---:|---:|---:|
| 1 (HEAD) | 2 | baseline | 0 |
| 2 | 3 | +218 MB | 1 |
| 3 | 4 | +436 MB | 2 |
| 4 | 5 | +654 MB | 3 |
| 8 | 9 | +1.5 GB | 7 |
| 32 (= n_layers) | 33 | +6.8 GB | **= original O(n_layers) bug. REJECTED.** |

(For same-dtype fresh case — where param-grad views already keep `reduce_output` alive — the extra keep-alive slots are refcount-redundant, so the "extra peak memory" column is a strict upper bound and often won't be observed.)

### Correctness argument (generalized)

For any valid `depth = k ∈ [1, n_layers)`:

1. `reduce_output_K` is held by `_all_reduce_state_queue` from iter `K` through iter `K − k − 1`'s step 3, where it rotates to the front of the queue and is explicitly `del`'d — returning the block to the RS pool *before* the same iter's `foreach_reduce` runs.
2. Earliest RS-stream reuse of the block: iter `K − k − 1`'s own `foreach_reduce` alloc (step 4 of the same `post_backward` as the free).
3. Before step 4 runs, the RS stream has already been issued `reduce_scatter_stream.wait_event(post_reduce_event_K)` — queued at iter `K − k`'s step 1, when `post_reduce_event_K` rotated out of `_pending_rs_wait_queue`. That's one wall-clock iteration earlier than the free/reuse.
4. RS-stream work is FIFO: step 3's free and step 4's alloc (and the alloc's downstream RS kernel) all run after the event queued by iter `K − k`'s step 1 completes → safe reuse.

Collapses to the `k=1` HEAD-equivalent case: `reduce_output_K` freed at iter `K − 2`'s step 3; reused in iter `K − 2`'s step 4; protected by the wait queued at iter `K − 1`'s step 1.

### Interactions with other features

- **Pipeline parallelism (PP):** each PP stage has its own backward. `depth` applies within each stage's `post_backward` sequence. No cross-stage coupling. Safe.
- **Gradient accumulation (grad_accum > 1):** the same-dtype-accumulate path (which needs the keep-alive even without a cast) still works — the queue still holds the keep-alive across microbatches.
- **`all_reduce_grads=False` / partial_reduce_output:** in this mode, `all_reduce_input` comes back as `None` and no entry is appended to the queue. Harmless — queue just stays shorter.
- **`share_comm_ctx` across modules:** the queue is on `comm_ctx`, so sharing a `comm_ctx` across multiple FSDP roots shares the queue too. Since the serial-backward invariant already requires non-concurrent backward on a shared `comm_ctx`, this is fine.
- **CPU offload:** the new RS-stream wait only fires when `post_reduce_stream is not reduce_scatter_stream`, which remains true on CPU-offload HSDP. No behavioral change.

### When to pick what

| scenario | recommended `depth` |
|---|---|
| default / correctness-first | **1** (HEAD) |
| AR stream is a measured bottleneck (network slow, inter-node AR) | try **2** first, benchmark |
| host has headroom, GPU is underutilized due to RS/AR serialization | **2–4**, pick lowest that saturates the GPU |
| wedge workloads (e.g., FP8 all-gather with slow FP32 AR) | up to `n_layers / 4` with explicit benchmark |
| never | `>= n_layers` — identical to the bug |

### Complexity cost

Compared to the current HEAD implementation (rotate-1 with single slots), the parameterized version adds:
- Two deque fields on `FSDPCommContext` replacing two named slots.
- A config field + validation.
- A one-liner to drain pending RS waits at `flush_all_reduce_state`.
- ~30 additional lines of code total; no new streams, no new events beyond what's already recorded.

If the default stays at `depth=1`, the new code path is identical to HEAD in observable behavior — the feature is effectively opt-in for workloads that profile shows need it.
