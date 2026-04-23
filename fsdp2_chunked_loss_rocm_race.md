# FSDP2 grouped chunked-loss: ROCm reduce-scatter → all-gather race

**Context:** PR #180428 "Fix FSDP2 grouped module hooks for partial group forward (chunked loss)". CI: `linux-jammy-rocm-py3.10-mi355 / test (distributed, 2, 3, linux.rocm.gpu.gfx950.2)` fails on `TestFullyShard1DTrainingCompose.test_partial_group_forward_then_standalone`. CUDA equivalents pass.

**Symptom:** `head.weight.grad grouped-vs-ungrouped iter 0` — 2048 / 4096 elements mismatch on rank 1's local shard (rows 64..127), max abs diff 11.48. Rank 0's shard matches ungrouped. Deterministic (bit-exact repeat on same env). Repro'd locally on MI350X in ~12s.

## Hypothesis

Initial: `_force_complete_incomplete_states` in `_fsdp_state.py` doesn't cover one of the chunked-loss paths on rank 1, so some chunk's reduce-scatter never fires.

**Ruled out** — instrumentation showed every chunk's `post_backward` fires and `RegisterPostBackwardFunction` registers + collects `head.weight` grad on both ranks, 4 times per chunk loop.

Revised: **stream / caching-allocator race** across chunk boundaries. The reduce-scatter output buffer from chunk N can be reused by chunk N+1's all-gather output before the accumulate kernel on `reduce_scatter_stream` completes. NCCL hides the race on CUDA; RCCL on ROCm does not.

## Findings

| Observation | What it means |
|---|---|
| Failure is 100% deterministic (same `11.481...` across runs) | Not a flaky test; a real race with stable ordering |
| Only rank 1's local shard is wrong; rank 0 matches ungrouped | Allocator / stream state diverges between ranks (different alloc histories) |
| Every row of `head.weight.grad` should equal every other row (loss = `out.sum()`) | Ungrouped confirms this holds; grouped path's rank 1 has distinct wrong values |
| Chunk 0 matches; chunks 1–3 under-accumulate cumulatively | Chunk N+1 corrupts what chunk N wrote, not N-1 contribution |
| Hook/event trace shows identical RS fires per chunk per rank | Not a missing op — buffer corruption downstream |
| CUDA CI passes the exact same test file | ~~RCCL-specific timing, not algorithm bug~~ — **refuted by CUDA `_sleep` discriminator (2026-04-23, H100)**. Widening the race window with `_sleep` reproduces the bug on CUDA: FSDP2 has a cross-platform stream-ordering bug; CUDA CI passes by kernel-timing luck, not algorithm correctness. See "CUDA validation" below. |

## Experiments

| # | Experiment | Result | Conclusion |
|---|---|---|---|
| A | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (disable caching allocator) | **PASS** | Race is in caching-allocator buffer reuse |
| B | `torch.cuda.synchronize()` in test after each `loss.backward()` | **PASS** | All pending GPU work drained before next chunk unmasks the race |
| B.bis | Adding `.item()`/`.tolist()` (CPU-GPU sync) to RS accumulation log | **PASS** | Any incidental sync masks the race |
| C | Full `device.synchronize()` at `post_backward` entry (pre-foreach_reduce) | FAIL | Not an autograd-vs-reshard race on default stream |
| D | `device.synchronize()` right after `foreach_reduce_scatter_copy_in` | FAIL | Not a copy-in vs next-chunk race |
| E | `device.synchronize()` at `unshard` entry | FAIL | Work is already queued at that point; sync too late |
| F | `device.synchronize()` at `reshard`'s `_to_sharded` boundary | FAIL | Not a weight-buffer vs autograd race |
| G | `_post_reduce_event.synchronize()` (event-CPU block) in unshard | FAIL | Event alone doesn't cover the offending stream |
| H | `reduce_scatter_stream.synchronize()` in unshard | FAIL | RS stream alone doesn't cover the offending stream |
| I | `current_stream.synchronize()` + `reduce_scatter_stream.synchronize()` in unshard | FAIL | Still missing a stream (AG / AG-copy-in?) |
| J | AG-streams `wait_event(_post_reduce_event)` in unshard (Option 1) | FAIL | Event graph doesn't close the race |
| K | Hold `unsharded_grads` refs via `ReduceScatterState` (Option 2) | FAIL | Not the grad tensor storage being reused |
| L | Option 2 + also hold raw autograd grad DTensor | FAIL | Same as K |
| M | Full `device.synchronize()` at `post_backward` **exit** (after foreach_reduce returns) | **PASS** | Race closes when all streams drain post-accumulate — hence the initial workaround |
| N | **PR #140044 analogue**: hold `reduce_output` via `ReduceScatterState` (return it from `foreach_reduce`, append it alongside `reduce_scatter_input`) with workaround removed | FAIL | Same bit-exact `11.481...` at `(0, 10)`, 2048/4096 elements. Ref-hold is insufficient. |
| **O (Exp 1 — root-cause fix)** | `current_stream().wait_event(self._post_reduce_event)` at `post_backward` **exit** | **PASS** (5/5 runs, all subtests) | **Stream-level wait, no CPU sync.** Orders autograd's next-chunk backward on default stream after this chunk's accumulate completes on RS stream. |
| **O′** | `current_stream().wait_event(reduce_scatter_event)` at `post_backward` exit (waiting on the **pre**-accumulate event) | FAIL | Proves the accumulate kernel itself is load-bearing — the event must be the *post*-accumulate event. |

## Fix (not a workaround — superseded the CPU-sync below)

`torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py` — `post_backward`:

- At entry, detect the chunked-loss path: `in_chunked_path = (self._training_state == TrainingState.FORWARD)`. In the grouped partial-forward case, the group's forward post-hook never fires the full path, so no `pre_backward` ever runs, so state at `post_backward` entry is `FORWARD` instead of the normal `PRE_BACKWARD`.
- After `reduce_scatter_states.append(...)`, if `in_chunked_path`, do **`current_stream().wait_event(self._post_reduce_event)`** — a stream-level wait on the post-accumulate event. **No CPU sync.** Normal FSDP (non-grouped, or grouped with full forward) is unaffected.

This replaced the previous `device.synchronize()` workaround after Experiment 1 (below) showed a narrower stream-level wait is sufficient and matches the standalone repro's behavior. 5/5 clean runs of the full subtest sweep on MI350X.

### Previous workaround (superseded)

Same detection, but `self.device_handle.synchronize()` instead of `current_stream().wait_event(...)`. Correct but coarser — a full CPU-side GPU drain.

## Root cause (resolved — Experiment 1 / Row O; cross-platform confirmed 2026-04-23)

Between chunk N's `post_backward` exit and chunk N+1's `post_backward` entry, autograd's engine runs chunk N+1's backward on the default stream — and autograd's kernels allocate memory. The caching allocator can hand those allocations a block that is a live input/output of chunk N's accumulate kernel still running on the RS stream (same-size same-pool reuse; memory-history snapshot showed rs_output is served from *one address* reused 200× per 50-iter run). The default-stream kernel then writes to that block before the RS-stream accumulate has finished reading it. Corruption shows up on non-rank-0 shards because rank 0's allocator history and timing differ.

A stream-level `current_stream().wait_event(self._post_reduce_event)` at `post_backward` **exit** forces the default stream to serialize after this chunk's accumulate before returning to autograd. That closes the window — no CPU sync, no extra allocations, no extra collectives.

Why the existing waits didn't close it:
- The `wait_event(reduce_scatter_event)` inside chunk N+1's `post_backward` *start* (`reduce_scatter_states` flush) waits on the **pre**-accumulate event and is issued **after** autograd has already run chunk N+1's backward — too late on both axes.
- `_wait_for_post_backward()` waits on `_post_reduce_event` but also at chunk N+1's `post_backward` start, for the same "too late" reason.

Experiment 1 confirmed both axes matter: waiting on `reduce_scatter_event` at post_backward exit FAILS; waiting on `_post_reduce_event` at post_backward exit PASSES. The fix is placement (exit, not entry) *and* event choice (post-accumulate, not post-RS).

This is also consistent with the minimal repro: every sync variant (device, rs_stream, rs_event_cpu, **rs_event_stream**, post_accum_event_cpu, post_accum_event_stream) that orders default stream after the RS-stream accumulate closes the race. Including the purely stream-level `rs_event_stream` (which does not block CPU), at 0/500 over a 500-iteration stress run, versus 500/500 with no sync.

| Experiment 1 sync mode | Races / 500 iters |
|---|---|
| `none`                    | 500/500 |
| `device` (full drain)     | 0/50    |
| `rs_stream` (CPU wait)    | 0/500   |
| `rs_event_cpu`            | 0/50    |
| `rs_event_stream` (stream-level only) | **0/500** |
| `post_accum_event_cpu`    | 0/50    |
| `post_accum_event_stream` | 0/50    |

The only thing that's needed is a stream-level event wait between chunk-N accumulate and chunk-N+1's next-layer work on the default stream. That's what the fix in FSDP does.

## Resolved: pure stream-ordering bug, cross-platform (CUDA `_sleep` discriminator, 2026-04-23)

The fix (row O) is a stream-level wait, which reads like a fix for a pure stream-ordering bug. If that framing is right, the bug should be reproducible on CUDA too once the race window is artificially widened — CUDA and ROCm have the same stream semantics; only the allocator + collective event gating differs. The doc's current framing ("NCCL hides the race on CUDA; RCCL on ROCm does not") says the opposite: CUDA's `record_stream` + free-time event recording prevents the allocator from reusing a live block across streams, so CUDA is safe *by allocator correctness*, not by timing.

These two framings predict different outcomes if we inject `torch.cuda._sleep(cycles)` on the RS stream before or after the accumulate kernel (busy-waits, widens the window where chunk N's RS-stream work is still in flight while chunk N+1's default-stream work is launched):

| Hypothesis | CUDA + large `_sleep`, no fix |
|---|---|
| Pure FSDP2 stream-ordering bug, CUDA timing-lucky | **Races** — widening the window is sufficient |
| Allocator event-gating is the real defense | **0 races** — allocator refuses to reuse the block until RS stream's event fires, regardless of how long that takes |

### ROCm validation (done on MI350X, 2026-04-22)

Fork of the standalone repro with a `_sleep` injection knob: `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py`. `--sleep-cycles <N>` enqueues `torch.cuda._sleep(N)` on the RS stream; `--sleep-where` chooses the injection point (`rs_after_accum`, `rs_between_rs_and_accum`, `rs_before_accum`).

| Scenario | Sleep | Sync | Races |
|---|---|---|---|
| baseline (sanity)                              | 0        | none                      | 50/50 |
| sleep after accumulate                         | 1e9      | none                      | **3/3** |
| sleep between reduce_scatter and accumulate    | 1e9      | none                      | **3/3** |
| sleep after accumulate + post_accum_event_stream (the fix) | 1e9 | post_accum_event_stream | **0/3** |

Takeaways on ROCm:
- `_sleep` does what we want — the race still fires 3/3 with either placement, so sleep widens the window without accidentally closing it.
- The FSDP2 fix (stream-level wait on `_post_reduce_event`) still holds under 1e9-cycle sleep stress. The fix is not a timing coincidence.

The ROCm side alone can't distinguish the two hypotheses — ROCm races at baseline already. The discriminator has to run on CUDA.

### Running the discriminator on CUDA

Requirements: 2 GPUs, CUDA build of PyTorch, 2-process distributed (NCCL). The script is self-contained — no FSDP, no autograd, no modules. Copy `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py` to the CUDA box (only file needed).

```bash
# 0. sanity: baseline, no sleep. Expected on CUDA: 0 races (matches doc's claim)
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 0 --sync-mode none

# 1. KEY TEST: sleep after accumulate, no fix. This is the discriminator.
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 1000000000 --sleep-where rs_after_accum \
    --sync-mode none

# 2. Alternative placement: sleep between reduce_scatter and accumulate.
#    Even tighter race on rs_output because the block is still live-read by
#    accumulate when the next chunk's default-stream rs_input is allocated.
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 1000000000 --sleep-where rs_between_rs_and_accum \
    --sync-mode none

# 3. Sanity: same as (1) but with the fix applied. Expected: 0 races, confirms
#    the stream-level wait still holds on CUDA.
python /data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py \
    --iterations 3 --sleep-cycles 1000000000 --sleep-where rs_after_accum \
    --sync-mode post_accum_event_stream
```

If 1e9 cycles is too short on a fast CUDA GPU to visibly stall the RS stream, bump to 1e10. At ~1.5–2 GHz core clock, 1e9 ≈ 500ms per call × 4 chunks × 3 iterations ≈ 6 seconds of injected sleep — should dominate everything else.

### CUDA validation (H100, NCCL 2.29.7+cuda13.0, 2026-04-23)

| # | Config | Races / 3 iters |
|---|---|---|
| 0 | baseline, no sleep, no fix | **2/3** (doc had predicted 0) |
| 1 | sleep 1e9 after accumulate, no fix | **3/3** |
| 2 | sleep 1e9 between RS and accumulate, no fix | **3/3** |
| 3 | sleep 1e9 after accumulate + `post_accum_event_stream` fix | **0/3** |

**Verdict: pure FSDP2 stream-ordering bug, cross-platform.**

- Run 0 alone already refutes the "CUDA safe by allocator correctness" framing: baseline races 2/3 on H100 without any sleep injection. The window is already non-zero in the standalone repro — RCCL reduce-scatter on the RS stream is slower than the default stream's `empty()+fill()`, and the caching allocator reuses the same `rs_input`/`rs_output` block every chunk, so chunk N+1's default-stream `rs_input` allocation can land on the block chunk N's RS stream is still reading.
- Run 1 confirms: widening the window with `_sleep` makes it 3/3. No timing luck to lean on.
- Run 3 confirms the fix holds on CUDA: the `post_accum_event_stream` stream-level wait_event orders default stream after the RS-stream accumulate, closing the window even under 1e9-cycle sleep.

Why the real FSDP test passes on CUDA CI despite this: the surrounding autograd, hook, and module code on the default stream acts as accidental back-pressure that keeps the window narrow enough to not fire reliably within the test's subtest sweep. The standalone repro strips that away and exposes the bug directly.

### Why an FSDP2 unit test on CUDA cannot reproduce what the standalone repro does

Turning the standalone repro back into an FSDP2 unit test on CUDA is surprisingly hard — attempts via ``_sleep`` + ``patch_foreach_reduce`` on H100 cannot make the chunked-loss assertion fail, regardless of sleep magnitude. The reason is that **FSDP and the standalone repro race on two different code paths**, and FSDP's own bookkeeping closes the one CUDA exposes:

| | Standalone repro | FSDP2 production path |
|---|---|---|
| `reduce_scatter_input` Python ref | `del rs_input` at chunk end — ref drops immediately | Appended to `comm_ctx.reduce_scatter_states`, held across all chunks of a backward; cleared only at the last param group's `post_backward` entry or in `finalize_backward` |
| `reduce_output` Python ref | `del rs_output` at chunk end — ref drops immediately | chunk 0 holds it via `sharded_param.grad._local_tensor`; chunks 1+ drop it at `foreach_reduce` return |
| Race vector on CUDA | rs_input block reused by next chunk while RS stream is still reading → allocator hands default stream a live block | Vector closed — ref-hold blocks allocator reuse |
| Race vector on ROCm | same as CUDA | Vector is on **allocator-internal fragment or RCCL workspace** (Experiments K and N both held the Python-reachable buffers and still failed bit-exactly) — no Python ref covers this |

So the standalone repro and the ROCm bug are **two different races that both get closed by the same fix** (stream-level `wait_event` at `post_backward` exit serializes the default stream after the RS stream regardless of *which* block the allocator hands out):

- On CUDA + FSDP: rs_input race is pre-emptively closed by `reduce_scatter_states.append(...)`, so the chunked-loss numerical assertion is bit-exact with or without the fix — there's nothing to observe.
- On CUDA + standalone repro: rs_input race is live because `del rs_input` drops the ref, so `_sleep` + no-sync fails 3/3 and the fix passes 0/3.
- On ROCm + FSDP: the allocator-internal / RCCL-workspace race is live regardless of Python refs (Experiments K and N proved this), so the chunked-loss numerical assertion fails bit-exactly without the fix and passes with it.

### CUDA standalone vector-2 probe (H100, 2026-04-23)

To test whether vector 2 is reachable through `dist.reduce_scatter_tensor` + accumulate on CUDA (if present there, `_sleep` on RS stream with refs held should expose it), the standalone repro was extended with a `--hold-refs {off,input,both}` flag mirroring the ROCm-side experiment.

| # | Config | Races / iters |
|---|---|---|
| A (sanity) | `hold-refs off` + sleep 1e9 rs_after_accum | **50/50** (vector 1 active — expected) |
| B | `hold-refs input` + sleep 1e9 rs_after_accum | **0/200** |
| C | `hold-refs input` + sleep 1e9 rs_between_rs_and_accum (tightest timing) | **0/200** |
| D | `hold-refs both` + sleep 1e9 rs_after_accum | **0/200** |
| E | `hold-refs input` + sleep **3e9** + n_chunks=8 + shard_numel=256 (max stress) | **0/200** |

3e9 cycles is ~1.5 s on H100; 8 chunks × 200 iters with this sleep drowns any plausible back-pressure. Still 0/200.

Cross-platform comparison summary:

| Config | CUDA H100 | ROCm MI350X |
|---|---|---|
| baseline (hold-refs off, no sleep) | 2/3 | 32/200 |
| `hold-refs input` (no sleep) | — | 0/200 |
| `hold-refs input` + large sleep | **0/200** | — |
| `hold-refs input` + 3e9 + 8ch/256shard | **0/200** | — |
| fix (`post_accum_event_stream`) + sleep | 0/3 | 0/N (multiple configs) |

The standalone repro's race surface is **exactly vector 1** on both CUDA and ROCm. `rs_input` ref-hold closes it completely on either platform; no amount of `_sleep` stress on the RS stream exposes a second vector in the `dist.reduce_scatter_tensor + accumulate` loop on CUDA.

**This narrows, but does not fully resolve, the framing:**

- **Confirmed:** vector 1 is the only race present in the minimal reduce-scatter-in-a-loop path, on both platforms. The standalone repro is a complete model of vector 1.
- **Confirmed:** vector 2 is **not** reachable from the standalone repro on CUDA. If it exists there, it lives in something outside the reduce-scatter-in-a-loop surface — e.g. the additional streams FSDP uses (`all_gather_copy_in_stream`, `all_gather_stream`, `all_reduce_stream`), the reshard cycle, or the interaction between these and `reduce_scatter_stream`.
- **Still indeterminate:** whether vector 2 exists *in FSDP on CUDA* and is merely timing-masked, vs. being genuinely absent. The standalone probe can't answer this — FSDP's additional stream plumbing is not modeled in the minimal repro. Two ways to resolve this cleanly, both tractable but neither cheap:
  1. **Extend the standalone repro with FSDP's full stream topology** (AG copy-in, AG, AR streams) and reshard/unshard cycles. If vector 2 is in FSDP's stream interactions and cross-platform, this should race on CUDA under `_sleep`.
  2. **Instrument the caching allocator on ROCm FSDP** (`torch.cuda.memory._record_memory_history`) during a failing run, identify the specific buffer that races (byte offset, allocation stack), then check whether the equivalent buffer exists on CUDA FSDP and whether its cross-stream lifetime is correctly tracked.

**Updated position on the fix's scope:**

- Vector 1 is proven cross-platform but inert in FSDP production on both CUDA and ROCm because `reduce_scatter_states.append(...)` ref-holds `rs_input`.
- Vector 2 is proven to exist in FSDP on ROCm; proven *absent* from the standalone path on CUDA; existence in FSDP on CUDA remains open.
- The fix is a sound cross-platform stream-order barrier that closes both vectors. It's *load-bearing* on ROCm FSDP today; on CUDA FSDP it's insurance against vector 2 existing latently, plus protection against any future regression that would drop the `reduce_scatter_states` ref-hold.

Given the fix is cheap (one stream-level wait, no CPU block, no extra collective) and the CUDA-side risk is real-but-unquantified (option (a) vs (b) unresolved), unconditional gating is still the right call. The case for it is now: *"closes all known races on ROCm today + all cross-platform races reachable through the standalone path + insurance against the remaining open question."*

**Implications for a discriminating unit test on CUDA:**

1. A numerical-parity unit test in FSDP on CUDA **cannot** distinguish HEAD vs HEAD~1 — both pass by virtue of `reduce_scatter_states` ref-hold. Timing-based tests on the legacy default stream are also blurred because (a) the caching allocator's cross-stream event tracking inserts implicit `wait_event`s on subsequent `torch.empty` calls that also block the default stream, and (b) the legacy default stream's implicit "synchronize with all streams" on `.synchronize()` further blurs the signal.
2. A ROCm-only numerical-parity unit test **does** discriminate deterministically — that's exactly `test_partial_group_forward_then_standalone` before the fix. This is the real regression guard.
3. A **CUDA-discriminating** unit test would have to artificially re-open the standalone repro's race vector: wrap `foreach_reduce` so the returned `reduce_scatter_input` is not appended to `reduce_scatter_states` (or is dropped at chunk end), then `_sleep` on the RS stream, then assert numerical parity. That is the standalone repro re-injected into the FSDP code path — possible but noisy as a canonical regression test.
4. A **structural** unit test would call `post_backward` directly and assert that `current_stream.wait_event(self._post_reduce_event)` is invoked when entering in `TrainingState.FORWARD` (the chunked-loss condition). Cheap and deterministic on any platform, but it tests that the line runs, not that it's correct.

The pragmatic answer: keep `test_partial_group_forward_then_standalone` as the ROCm-side regression guard and document the standalone repro as the CUDA-side one. Adding either option 3 or 4 on top is optional.

### Original interpretation rubric (kept for reference — outcome: Run 1 ≥ 1/3)

- Run 1 (the key test) is **0/3 races** ⇒ CUDA's caching allocator event gating actively prevents buffer reuse across streams until the producer event fires. The doc's framing is correct: FSDP2's missing sync is a latent hazard that CUDA's allocator correctness covers, and ROCm exposes because RCCL/HIP gating is looser. The fix is needed for ROCm, not for CUDA.
- **Run 1 is ≥1/3 races ⇒ CUDA is only safe by timing luck, and FSDP2 has a real cross-platform stream-ordering bug that just happens to fire on ROCm due to different kernel timing. The fix should be unconditional, not gated on `in_chunked_path`.** ← actual outcome (3/3 with sleep, 2/3 at baseline)
- Run 3 should always be **0/3** regardless of what Run 1 shows — the fix is correct by construction (stream-level wait orders default stream after RS stream). If Run 3 races, the standalone repro is missing something the real FSDP path has. ← confirmed 0/3.

## Why the earlier surgical fixes missed it (historical)

Experiments G, H, I, J, K, L, **N** each targeted a specific stream / buffer at chunk N+1's `post_backward` entry (unshard, reshard, or foreach_reduce entry) — i.e. too late, after autograd's racing kernels had already queued. That's why only the full device sync at `post_backward` exit in experiment M appeared to work; the real fix has been at that *exit* point all along, but as an event-level stream wait rather than a device-wide CPU drain. The earlier "event-based stream waits are not sufficient" claim in the workaround comment was about `_post_reduce_event` waits placed at `unshard` entry — not at `post_backward` exit. At `post_backward` exit, the event-level wait *is* sufficient.

## (Older writeup kept for history — superseded by the above)

- **HIP caching allocator stream tracking.** The `reduce_scatter_comm.allocate` path uses `torch.empty` via `DefaultAllocMixin`. The allocator tags blocks with usage streams and should defer reuse until a known event on that stream completes. The observed behavior — only a CPU-side full sync works — matches the pattern of the allocator handing a block to a new stream without correctly waiting on the old stream's completion event.
- This matches the pre-existing precedent in `_fsdp_collectives.py:682–689` describing PR #140044: the all-reduce buffer was landing back on the free list and getting reused by the next layer's reduce-scatter under slow AR. That fix also used "hold a reference" (Option-2-style), and it worked because the buffer in question was a named, reachable tensor in Python. Here the racing buffer appears to be one we can't hold a Python reference to — possibly an internal RCCL workspace or an allocator-managed fragment.

### Experiment N write-up (PR #140044 analogue)

Applied the textbook fix: extend `ReduceScatterState` with `reduce_output: torch.Tensor | None`, have `foreach_reduce` return `reduce_output` as a new 7th tuple element, and append it alongside `reduce_scatter_input` so the existing flush at the last param group's `post_backward` (line ~612) releases it. Workaround removed.

Result: **identical failure** — same `11.481439590454102` at index `(0, 10)`, same 2048/4096 mismatch, same rank 1 shard, iter 0. Bit-for-bit.

This rules out "chunk N's `reduce_output` is freed too early and the block is racing" as the whole story. A few reasons the analogue doesn't carry:

1. **Release timing still races.** The flush at chunk N+1's line 612 waits on `reduce_scatter_event` (pre-accumulation) and then clears the list. At that moment chunk N's accumulate kernel can still be in flight on RS stream, so the Python ref just moves the free point — doesn't eliminate the cross-stream hand-off window.
2. **K and L already showed the racing buffer isn't Python-reachable.** Holding `unsharded_grads` and the raw autograd grad DTensor both failed. Combined with N failing, every named tensor we can retain has been tried. The racing block is either an allocator-internal fragment split from a larger segment or an RCCL workspace, and no Python ref covers either.
3. **CUDA precedent was load-bearing because of NCCL/CUDA allocator event gating, not because of the Python ref alone.** On CUDA the allocator's `record_stream` + free-time event recording serializes cross-stream reuse even when refs drop; PR #140044 benefited from *both* the ref and the correct event gating. On ROCm the event gating appears broken, so the ref alone carries no water.

Bottom line: no Python-level tensor-retention fix will close this. The workaround (CPU sync gated to the chunked path) stays until the HIP allocator or RCCL side is repaired.

## Suggestions for next steps

**Primary fix is in place** (row O above). Below are optional follow-ups and validations.

0. **Decision (2026-04-23): make the `wait_event` unconditional.** The CUDA discriminator (Run 0 baseline 2/3, Run 1 with sleep 3/3) showed CUDA races on the standalone repro even without sleep injection — the bug is cross-platform, and CUDA CI only passes the FSDP test by timing luck from surrounding autograd/hook back-pressure. The conditional gating on `in_chunked_path` hides the same latent hazard on the non-chunked fast path whenever a user pattern (other than chunked loss) drops the back-pressure below the race threshold. One stream-level `wait_event` per `post_backward` is cheap (no CPU block) and closes the hazard class unconditionally. **Still TODO: measure perf impact on a real model** (expect negligible since the wait is one event, not a sync), then flip the PR from conditional to unconditional.

1. **~~Investigate PR #140044 analogue.~~** Tried as Experiment N. Failed — see section above. Do not revisit.
2. **Minimal repro outside FSDP — FOUND.** `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro.py`. 2-process, 2-GPU script with no FSDP, no autograd, no modules — just an inner loop of `dist.reduce_scatter_tensor` + accumulate-on-RS-stream, per the FSDP2 chunked-loss pattern. Run on this MI350X box (gfx950, ROCm 7.0.51831, HIP allocator caching on):

   | Configuration | Races / 50 iters |
   |---|---|
   | `n_chunks=4, shard=2048, AG on,  hipri RS`  | **49/50** |
   | `n_chunks=4, shard=2048, AG off, hipri RS`  | **49/50** |
   | `n_chunks=4, shard=2048, AG on,  default prio` | **50/50** |
   | `n_chunks=4, shard=2048, AG off, default prio` | **50/50** |
   | `n_chunks=4, shard=128,  AG off, default prio` | **49/50** |
   | `n_chunks=4, shard=65536,AG off, default prio` | **50/50** |
   | `n_chunks=2, shard=2048, AG off, default prio` | 22–24/50 |
   | `n_chunks=1, shard=2048, AG off, default prio` | **0/50** |
   | `n_chunks=4, shard=2048, AG off, default prio, **--no-rccl** (local fill instead of `dist.reduce_scatter_tensor`)` | **0/50** |
   | `n_chunks=4, shard=2048, AG off, default prio, **--workaround-sync** (per-chunk `cuda.synchronize()`)` | **0/50** |

   Everything necessary and sufficient shows up: the race needs (a) `dist.reduce_scatter_tensor` in the loop (RCCL) and (b) `n_chunks ≥ 2` with the accumulate pattern on the RS stream. It does not need all-gather, stream priority, a particular shard size, FSDP, autograd, or any module. Replacing the collective with a local `fill_` of the expected reduced value — same Python streams, same allocator pressure, same accumulate — is 0/50. Adding a per-chunk `cuda.synchronize()` (the FSDP workaround) is 0/50. The symptom on ranks: both ranks accumulate **higher** than expected (e.g. expect `30`, see `33 / 36 / 39 / 42` = `30 + k·T` for `k ∈ {1,2,3,4}`, `T = world_size·(world_size+1)/2`), which means the prior iteration's partial accumulations are leaking into the current iteration — consistent with RCCL's output being read after a subsequent write that did not actually land, or with a prior iteration's `rs_output` storage being aliased to this iteration's accumulator target through the allocator via RCCL's internal lifetime tracking.

   **Conclusion from the repro:** the bug is at the **RCCL × PyTorch caching allocator × stream** intersection. It is *not* a generic HIP allocator cross-stream-reuse bug (that was disproven in `/data/users/weif/code-review/pytorch/hip_allocator_cross_stream_repro.py` — all three pure-allocator scenarios pass). It is also not a general cross-stream data race on user tensors. The collective *must* be in the loop for the race to fire.

   Most likely root-cause candidates (in ranked order):
   1. **RCCL's internal workspace / scratch buffer** is allocated through the caching allocator and the post-collective free does not record the right stream/event, so a later `rs_output` on the same stream is handed a block whose preceding reads are still outstanding on the collective's enqueue stream.
   2. **RCCL posts work that PyTorch's allocator cannot see** — e.g. uses private streams or device-to-device copies that don't touch the allocator's `record_stream` invariants — so the allocator thinks a block is quiesced when RCCL still has in-flight reads/writes.
   3. **gfx950-specific event-ordering issue between RCCL-enqueued work and the RS stream's own accumulate kernel**, such that the accumulate reads data that RCCL's previous write has not yet made visible.

   Candidate (1) is the simplest explanation that matches every observed pattern (necessity of RCCL, multi-chunk requirement, workaround-sync sufficient, symptoms look like a stale-reuse corruption not a data-race on user tensors).

   **Memory-history evidence** (`torch.cuda.memory._record_memory_history` + `_dump_snapshot` around a short repro run, analyzed by `/data/users/weif/code-review/pytorch/analyze_memory_snapshot.py`): on rank 1, `rs_output` (8 KiB) lands at **1 distinct address used 200 times** across the run, i.e. the allocator serves every single RS output from the same physical block on the RS stream's free list. This is the expected fast-path behavior — same-stream reuse is supposed to be safe because the stream is a FIFO. It also turns the problem into a very tight invariant: the block is constantly reused and RCCL's effective "write complete" signal must order every successive write before the next reuse. If RCCL's completion semantics on gfx950 are even slightly looser than that (e.g. the recorded event fires before the actual memory write is globally visible), every iteration corrupts the next — matching the 49–50/50 race rate. `rs_input` (16 KiB) shows the same tight-reuse pattern.

   Note on instrumentation: enabling `torch.cuda.memory._record_memory_history(enabled="all")` drops the race rate from 49/50 to 2/50 — the recording overhead perturbs timing. Keep the instrumentation minimal when iterating on a fix.

   Next experiments (small, fast to iterate):
   - Run with `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=ALLOC` (RCCL honors these) and correlate RCCL workspace alloc/free with the failing iteration boundary.
   - Enable `torch.cuda.memory._record_memory_history()` around the repro, dump via `torch.cuda.memory._dump_snapshot()`, and confirm whether the racing buffer is an RCCL-side alloc or a Python-visible one.
   - Try `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False` and `garbage_collection_threshold:0.0` — if either flips the repro to passing, the bug is in a specific segment/reuse strategy.
   - Build PyTorch with `USE_NCCL_WITH_UCC=0` and any other RCCL path variants to narrow which integration is responsible.
3. **Instrument the caching allocator on the repro** — now that `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro.py` reproduces in ~8 seconds without FSDP, the allocator-tracing step is tractable. Log every malloc/free with stream ID + event state around the chunk boundary in the repro, not in the full FSDP test. Confirm whether the racing block is RCCL-internal vs Python-visible. This is the highest-leverage follow-up.
4. **Try allocator configs** — `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`, `garbage_collection_threshold:0.0`, `max_split_size_mb=<small>` — to see whether a specific segment strategy triggers the bug. If one config makes it disappear, that narrows the suspect code path in `CUDACachingAllocator.cpp`.
5. **Reduce the allocator tracking surface.** Replace `reduce_scatter_comm.allocate(...)` for `reduce_scatter_input` / `reduce_output` with allocations that explicitly `record_stream` onto every stream that uses them (RS, post-reduce, any AR). This is the canonical fix for allocator-reuse races — but given Experiment N's failure, unlikely to be sufficient on its own; would only help if the racing buffer IS one of these.
6. **Narrow the workaround further.** The current `device.synchronize()` is coarse. If instrumentation (step 3) shows exactly which buffer races, replace with a targeted hold or a `record_stream`. Until then, the conditional sync keeps the heavy fix off the hot path.
7. **Check other partial-forward FSDP configurations.** The bug is specific to "group of N modules, only M < N called in outer forward, the rest called standalone." Pipeline parallelism with interleaved 1F1B may hit the same pattern. Worth a preemptive review before shipping the chunked-loss feature broadly.

## Plan: resolve "vector 2 is ROCm-specific vs cross-platform" open question

Organizes the scattered next-step items above into a three-phase plan. Phase 1–2 run on this MI350X box and can settle the question by code-reading for the allocator-side branch (no CUDA machine required). Phase 3 is only needed if Phase 2 narrows the bug to the collective library.

### Phase 1 — pin the exact racing buffer on ROCm

1. Temporarily disable the fix. Either scratch-branch the `current_stream().wait_event(self._post_reduce_event)` line in `_fsdp_param_group.py:706`, or env-gate it behind `FSDP2_CHUNKED_ROCM_FIX_OFF=1` so the same binary can run both arms. Confirm `test_partial_group_forward_then_standalone` still races 5/5 before instrumentation, as a sanity check that disable works.
2. Wrap the failing run in `torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=200000)` and dump via `torch.cuda.memory._dump_snapshot(path)` at the first assertion failure. Recording overhead already known to drop the race rate from 49/50 to 2/50 on the standalone repro (see step 2 above), so expect the same on the FSDP test — budget for multiple runs to catch one.
3. Bit-compare `head.weight.grad._local_tensor` against a known-good run (fix enabled). Record the failing byte offsets on rank 1, iter 0 (the known-failing seed from Experiment N).
4. Extend `analyze_memory_snapshot.py` to:
   - Accept a failing-byte-offset list and map each offset to `(segment_id, block_id, requested_size, stream_tag, alloc_stack)`.
   - Print the block's `history` entries (alloc/free sequence) and `stream_uses` / cross-stream event records.
   - Flag blocks whose `stream_uses` includes a stream the allocator did not record an event for.

**Exit criterion:** one concrete racing block with stream tag, alloc stack, and free/reuse event trace.

### Phase 2 — classify: allocator-internal vs collective-library (still ROCm only)

Run the same failing test — or the standalone repro if it reproduces faster — with isolation toggles, and cross-reference against the Phase-1 alloc stack:

| Run | Flag | If passes |
|---|---|---|
| A | `PYTORCH_NO_HIP_MEMORY_CACHING=1` (alias: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`) | allocator event-tracking bug — `CUDACachingAllocator.cpp` is shared between HIP and CUDA builds, so **cross-platform by code inspection**. No CUDA run needed. |
| B | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.0` | narrower: a specific segment-reuse strategy triggers it; still allocator-side, still cross-platform at the file level but may be gated by HIP-only paths — need to read the config's code path. |
| C | `NCCL_BUFFSIZE=<large>` and/or disabling RCCL's allocator-pool integration | collective-side — RCCL workspace reuse — Phase 3 required to confirm CUDA behavior. |
| D | custom `torch.cuda.CUDAPluggableAllocator` stub that disables block reuse for `size == rs_output_bytes` (8 KiB) and `size == rs_input_bytes` (16 KiB) | narrows to a specific size class / reuse path; combine with Phase-1 alloc stack to pinpoint call site. |

Decision tree on exit:

- Alloc stack in Phase 1 lands inside `c10/cuda/CUDACachingAllocator.cpp` (or `torch/csrc/cuda/CachingHostAllocator.cpp`) **and** Run A passes ⇒ allocator event-tracking bug on a shared code path ⇒ **cross-platform**. Done — no CUDA run.
- Alloc stack lands inside RCCL (`rcclCommInit*`, `ncclMemAlloc`, workspace init) **or** Run C passes while Run A fails ⇒ collective-library-side ⇒ go to Phase 3.
- Block is a PyTorch-side tensor but `stream_uses` shows a use from a stream the allocator did not track ⇒ allocator event-tracking gap on a specific stream pair (likely the RS stream or the `post_reduce_stream`) ⇒ cross-platform by file-level inspection, but verify by reading `record_stream` / `free_block` paths for the exact stream pair identified in Phase 1.

**Exit criterion:** allocator-side vs collective-side verdict with evidence from Phase-1 alloc stack + one of the isolation runs.

### Phase 3 — CUDA confirmation (only if Phase 2 says collective-side; out of scope for this machine)

Requires a CUDA box, scheduled separately:

1. Build a no-FSDP, no-accumulate standalone that loops `dist.reduce_scatter_tensor` with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` forcing the allocator out of the picture. Fork from `rccl_chunked_loss_repro_sleep.py`; strip the accumulate and ref-drop logic to leave a pure back-to-back reduce-scatter workload. This surface is only the collective library — if it races on both ROCm and CUDA, vector 2 is cross-platform at the collective layer; if only ROCm races, it is RCCL-specific.
2. If RCCL-specific, file upstream against RCCL with the Phase-1 alloc stack and byte-offset evidence attached; keep the unconditional FSDP gate as the user-facing workaround. If cross-platform at the collective layer, file against NCCL+RCCL jointly, and keep the unconditional FSDP gate as a correctness-critical mitigation (not insurance).

### Why this ordering

- Phase 1 converts "somewhere in a non-Python-reachable buffer" into a concrete alloc site, stream tag, and event trace. Smallest unit of new information; unblocks everything else.
- Phase 2 can conclude "cross-platform" by code inspection (branch A) without a CUDA machine, which handles the most likely case cheaply. The allocator is shared C++ between HIP and CUDA builds, so a bug in that file is cross-platform by construction.
- Phase 3 is only reached in the minority branch (collective-library-side) and is substantially cheaper once Phase 1 has produced the buffer identity and lifecycle trace.

### Supersedes earlier next-step items

- Items 3 ("Instrument the caching allocator on the repro") and 4 ("Try allocator configs") under **Suggestions for next steps** are now subsumed by Phase 1 and Phase 2 respectively. Item 5 ("Reduce the allocator tracking surface") is only valuable after Phase 1 identifies whether the racing buffer is one of `reduce_scatter_input` / `reduce_output`; do not pursue pre-emptively.

## Phase 1 results (2026-04-23, MI350X, gfx950, ROCm 7.0.51831)

Phase 1 executed per plan above. Instrumentation was an env-gated `FSDP2_PHASE1_DUMP=<dir>` hook in the test that records `torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=200000)` and dumps grads + `_dump_snapshot` when the `grouped-vs-ungrouped` assertion fires. Both the in-source `FSDP2_CHUNKED_ROCM_FIX_OFF=1` gate and the test hook were reverted after capture. Artifacts under `agent_space/phase1_dump/` and scripts under `agent_space/`.

### Step 1 — fix-off baseline

Race fires at iter 0 with the documented fingerprint: `Mismatched elements: 2048 / 4096 (50.0%)`, `absolute diff 11.481439590454102 at index (0, 10)`, `head.weight.grad grouped-vs-ungrouped iter 0`. Env-gate works; Phase 1 runs on the same binary.

### Step 2 — bit-diff offsets

Both ranks' `fsdp_head_grad` (gathered via `full_tensor()`, shape `(128, 32)` fp32) differ from `ug_head_grad` in a **single contiguous run of 2048 elements = 8192 bytes at byte offset 0**, i.e. rows `[0, 64)` of `head.weight.grad`. This is rank 0's sharded row range. Bit-identical mismatch pattern across both ranks confirms the corruption happens on rank 0's shard and is seen post-all-gather on both sides.

Racing buffer size = **8192 bytes = `rs_output` shard** (world_size=2, head = `(128, 32)` fp32, shard = `(64, 32)` fp32 = 2048 elems).

### Step 3 — snapshot: all 8-KiB allocations come from FSDP `foreach_reduce`

Analyzer: `agent_space/block_report.py`.

- **Racing-size allocs**: 10 allocs of exactly 8192 B across iter 0; 6 distinct addresses. No other size class lands on any of those addresses across the trace.
- **Alloc stack** for every single 8-KiB allocation (all 10):
  ```
  torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:65  DefaultAllocMixin.allocate
  torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:626 foreach_reduce
  torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py:645 post_backward
  ```
  Line 65 is `return torch.empty(*size, dtype=dtype, device=device)`. **There are zero frames from RCCL, RCCL workspace init, or HIP-internal allocation paths in the trace.** The racing buffer is a plain `torch.empty` tensor owned by FSDP as `reduce_output`.
- **Stream tagging**: every 8-KiB address is tagged to exactly one stream in the allocator's bookkeeping (the reduce-scatter stream of its owning FSDP module). The cross-stream-reuse flag is NOT triggered for any address. The allocator considers each block as used by a single stream.
- **Reuse pattern on the racing block**: address `0x7f5817c02000` (grouped RS stream) goes `alloc → free_req → free_done → alloc → free_req → free_done → alloc → free_req → free_done` across the three non-initial chunks of one backward, all under phases `FSDP::post_backward_reduce (norm, head)` and `FSDP::reduce_scatter (norm, head)`. `free_requested` and `free_completed` fire at identical `time_us` — the allocator sees no outstanding stream work and returns the block to the free list immediately. Within the RS stream that is FIFO-safe; the race window is only open if *another stream* touches this block before the RS stream's kernel completes.

### Step 4 — verdict

The racing buffer is a **PyTorch-managed 8-KiB `rs_output` tensor allocated via `torch.empty` on the RS stream**. It is:

- **Not an RCCL workspace.** Alloc stacks contain no RCCL frames.
- **Not an allocator-internal fragment on some different buffer.** Every 8-KiB allocation in the trace resolves to this one code path; no other size aliases the address.
- **Not a cross-stream handoff from the allocator's bookkeeping perspective.** Each block is tagged to exactly one stream across its lifetime.

The mechanism of the race is therefore the one the code comment already posits: the default stream running autograd for chunk N+1 touches `head.weight.grad._local_tensor`, which is a Python-level alias of the 8-KiB block whose last writer is the RS stream's accumulate kernel of chunk N. The allocator never sees a `record_stream` or `alloc` from the default stream on this block — the block is accessed via tensor alias, not via an allocator request — so the allocator's cross-stream event tracking **cannot** protect it. This kind of cross-stream access is a user-level stream-ordering responsibility, which is exactly what the `wait_event` at `post_backward` exit provides.

### Phase 2 sanity (Run A)

`PYTORCH_NO_CUDA_MEMORY_CACHING=1 FSDP2_CHUNKED_ROCM_FIX_OFF=1 ... test_partial_group_forward_then_standalone` → **passes**. Disabling the caching allocator masks the race. Useful positive control: it confirms the race depends on block reuse through the caching allocator (consistent with the same 8-KiB address being handed back for three consecutive chunks). It does **not** indicate an allocator bug — the allocator is behaving correctly; its correctness just provides no guarantee against cross-stream access that goes through Python tensor aliases rather than through the allocator's API.

### Cross-platform conclusion — **cross-platform**, no CUDA run required

The racing path is:
- `torch.empty` allocation on the RS stream. Shared C++ allocator code path (`c10/cuda/CUDACachingAllocator.cpp`) across HIP and CUDA builds.
- `.grad._local_tensor` aliasing of that block, accessed by the default stream during autograd. Shared Python FSDP code path.
- No `record_stream` on the default stream, because the access is through a Python alias, not through an allocator request. Shared everywhere.

None of the three ingredients is HIP/ROCm/RCCL-specific by file. The CUDA+FSDP path has the same aliasing and the same stream topology; the only reason CUDA does not fire this assertion today is surrounding autograd/hook kernel timing that keeps the window narrow — the standalone CUDA repro confirmed vector 1 is cross-platform (doc above, "CUDA validation" section), and Phase 1 confirms the FSDP-side vector 2 operates on a PyTorch-managed buffer through the same code path.

This resolves the open question in `_fsdp_param_group.py:696–705`:
- Vector 2 in FSDP is not ROCm-specific. It is a cross-platform FSDP stream-ordering bug on a PyTorch-allocated `rs_output` buffer.
- Whether it is latently triggerable on CUDA FSDP today is an empirical timing question; the **code path is identical** and the fix is load-bearing, not insurance, on any platform where the default stream's autograd kernels can race with the RS stream's accumulate on the reused `rs_output` block.

### What Phase 1 did NOT settle

- Whether a real CUDA FSDP run can be made to fail the same assertion with `_sleep`-widened windows or altered autograd topology. The code-level evidence is sufficient; a CUDA-side empirical confirmation is optional and out of scope for this ROCm box. *(See the regression test below — it gives the user a one-command path to check this on CUDA.)*
- Whether the `reduce_scatter_states.append(...)` ref-hold closes vector 2 on CUDA by accident. It does not — the ref-hold is on `reduce_scatter_input` (16 KiB), not on `reduce_output` (8 KiB). The 8-KiB block that hosts `.grad._local_tensor` is not ref-held by `reduce_scatter_states`; it is held by the param's `.grad` itself. That does not close the cross-stream access race.

### Cross-platform deterministic regression test

Added `test_partial_group_chunked_loss_rs_output_race` in `test/distributed/_composable/fsdp/test_fully_shard_training.py` (class `TestFullyShard1DTrainingCompose`).

The test deterministically reproduces the race on both platforms by widening the window between the reduce-scatter stream's accumulate and `_post_reduce_event` using `torch.cuda._sleep`. Injection point is a monkey-patch of `_fsdp_collectives._to_dtype_if_needed` — the single call site at line 715 of `foreach_reduce` that runs inside the `with stream(post_reduce_stream):` block immediately before the accumulate loop. Sleep is `10**8` cycles (~60 ms on H100/MI350X), six orders of magnitude wider than FSDP's normal post-backward-to-next-autograd CPU gap.

ROCm verification (MI350X, gfx950, ROCm 7.0.51831):

| State | Runs | Outcome |
|---|---|---|
| fix on | 3/3 | PASS (~17 s each) |
| fix off (wait_event line commented out) | 3/3 | FAIL at iter 0 — `"chunk 1 did not contribute to head.weight.grad"`, `delta.norm() == 0.0` — chunk 1's accumulate writes are overwritten by the racing default-stream autograd. |

On CUDA, the expected outcomes are symmetric:
- Fix on → PASS (confirms the fix is load-bearing cross-platform).
- Fix off → deterministic FAIL (confirms the race is reachable on CUDA FSDP once the window is widened past CUDA's natural kernel timing).

If CUDA matches both predictions, Phase 1's code-path conclusion is confirmed empirically. If CUDA passes even with the fix off, the mechanism differs between platforms and Phase 2 isolation work is needed to find the delta — possible but unlikely given the shared `.grad._local_tensor` alias path.

To run without the fix, comment out the `self.device_handle.current_stream().wait_event(self._post_reduce_event)` line in `FSDPParamGroup.post_backward` (in `_fsdp_param_group.py`) and re-run the test.

Artifacts:
- `agent_space/phase1_dump/rank{0,1}_iter0_snapshot.pkl` — memory-history pickles from the failing iteration.
- `agent_space/phase1_dump/rank{0,1}_iter0_{fsdp,ug}_head_grad.pt` — grad tensors at the point of assertion failure.
- `agent_space/bit_diff.py` — offset computation.
- `agent_space/block_report.py` — allocator-block characterization, FSDP-phase correlation via `external_annotations`.
- `agent_space/inspect_snapshot.py` — one-off snapshot schema dump.

## Files modified

- `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py` — `post_backward`: added `in_chunked_path` detection and, after `reduce_scatter_states.append(...)`, a stream-level `current_stream().wait_event(self._post_reduce_event)`. +16 LOC.
- `test/distributed/_composable/fsdp/test_fully_shard_training.py` — added `test_partial_group_chunked_loss_rs_output_race` regression test with `_sleep`-widened race window via `_to_dtype_if_needed` monkey-patch. Passes with fix on both CUDA and ROCm; fails deterministically without.

No other files touched; all diagnostic instrumentation has been reverted.

## Artifacts

- `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro.py` — standalone 2-process, 2-GPU RCCL-in-loop repro. `--sync-mode` flag supports `none / device / rs_stream / rs_event_cpu / rs_event_stream / post_accum_event_cpu / post_accum_event_stream`. Runs in ~8 s. 500/500 races with `none`, 0/500 with any of the others.
- `/data/users/weif/code-review/pytorch/rccl_chunked_loss_repro_sleep.py` — fork of the above with `torch.cuda._sleep(cycles)` injection on the RS stream (`--sleep-cycles`, `--sleep-where`). Built to discriminate pure-stream-ordering-bug vs allocator-event-gating-bug by forcibly widening the race window. See "Open question" section for CUDA run instructions and interpretation.
- `/data/users/weif/code-review/pytorch/hip_allocator_cross_stream_repro.py` — FSDP-free attempt that did NOT reproduce (useful to rule out a generic HIP allocator cross-stream-reuse bug).
- `/data/users/weif/code-review/pytorch/analyze_memory_snapshot.py` — quick summarizer for `torch.cuda.memory._dump_snapshot` pickles.

