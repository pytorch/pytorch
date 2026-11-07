---
name: tune-pysymmem-collectives
description: Specialize PySymmem collectives for a user's workload by tuning backend orchestration or writing custom Triton collective kernels. Create workload-specific correctness checks and benchmarks, then compare with unchanged PySymmem and a reference such as NCCL while preserving numerical and execution requirements.
---

# Tune a PySymmem ProcessGroup

Make the supplied collective workload faster. Modify backend orchestration or
write custom Triton collective kernels as needed, keeping application calls through
`torch.distributed`. Deliver the best validated candidate and reproducible
measurements, including losses. No user-provided benchmark, profiling trace, or
existing experimental kernel is required. Create the benchmark needed to evaluate
the requested workload.

Establish a working benchmark and baseline, then implement one promising candidate,
check essential correctness, and compare immediately. Match further tuning and
validation to the requested outcome: a feasibility result, a workload-specific
implementation, or a change ready for broader backend use.

## Phase 1: get the first useful comparison

### Establish the workload cheaply

Read repository instructions and locate the current backend, registration, and
a relevant test. Inspect the requested operation's path without
surveying unrelated collectives. Check existing scratch kernels for a promising
starting point; remeasure old results. Historical paths may no longer exist.

Establish sizes, dtypes, layouts, reduction operator, group membership, precision,
and execution mode from the request. Ask only for missing information that changes
the experiment. For an exploratory POC, choose and disclose a small representative
set. Match the actual ownership model and rank count; do not rebalance uneven
shards or describe modeled buffers as training traces.

Define success before choosing the candidate: isolated latency, performance over
a representative mix of calls, sustained throughput, or application time with
compute overlap. Establish relevant workspace and resource limits. Derive case
weights from the workload when available; otherwise report cases separately.
If unspecified, start with public-call latency and state that assumption.

For reductions, distinguish input/storage, accumulation, and output dtypes;
permitted intermediate rounding; error tolerances; and required rank agreement
or determinism. Require a particular summation order only when the workload
requires it. Choose the correctness reference to match this contract.

Include any required scaling or division in the contract. A higher-precision
reference can assess error; an explicitly ordered reference can check a required
arithmetic order. Neither automatically defines every valid implementation of
wide accumulation. FP32 accumulation does not imply exact sums or identical
replicas with different summation orders. Retain the requested intermediate
precision until arithmetic is complete: widening an accumulator cannot recover
information lost by narrowing an unfinished sum. Storing a fully reduced result
in the output dtype is different.

Check GPU count, topology, software versions, and competing processes. Check peer
access when needed. Use the requested execution mode; default to eager if
unspecified. Add graph, async, and concurrent cases when relevant to the workload.

### Create the benchmark and measure the baseline

Build a small distributed launcher for the requested workload. An existing
harness is optional: reuse it only after checking its semantics and timing scope.
Do not ask the user to supply a benchmark or build a generalized framework as a
prerequisite. Make cases, backend choice, rank count, warmups, repetitions, and
output location explicit and reproducible. Use the same cases and measurement
procedure for all compared implementations.

Choose cases from the request and available application code. If only a range is
known, sample small, representative, and large sizes within that range and the
available memory budget. Include relevant tails and dispatch boundaries without
creating an exhaustive Cartesian product. Distinguish correctness stress cases
from performance cases and label synthetic workloads as such. Use deterministic,
rank- and position-dependent input generation to make failures reproducible.

Before implementing the candidate, validate the launcher and obtain a quick
baseline using the correctness and timing procedure below. Measure unchanged
PySymmem when evaluating improvement to the packaged backend, alongside a suitable
reference such as NCCL. Record revision, settings, launch command, and hashes of
measured sources. This establishes the measurement setup and performance target
before changing the implementation.

Match input/output semantics and numerical requirements. Include reference casts
needed to meet the contract. Native low-precision NCCL is a separate control unless
its actual precision meets that contract. An enabled NVLS setting or hardware
capability does not prove the algorithm or precision used by a particular call.
If a baseline cannot satisfy the contract, state that limitation explicitly.

### Implement one candidate

Prefer Triton for new collective kernels, using PySymmem's existing symmetric-memory
and synchronization primitives. Use another implementation path when the workload
or required hardware feature justifies it.

State a short specialization hypothesis: which workload property the candidate
exploits, which cost it removes, and which cost it introduces. Use the actual
data path and a rough accounting of launches, bytes moved, and synchronization
phases to choose the simplest plausible algorithm. State its supported domain
and dispatch conditions. Keep hardware thresholds and tuning values specific to
the measured configuration. A profiler is optional when the hypothesis is clear.

Make one candidate measurable before adding variants or sweeping launch settings.
Include preparation required by the candidate, such as supplying symmetric input
storage, in the application comparison. Do not silently change ownership,
precision, or application calls to make the candidate win.

Preserve affected starting sources before replacing them, including local edits.
A separate scratch candidate usually avoids needing an isolated checkout.
Do not edit imported sources during a run. Reconcile checkout changes before
continuing measurements.

Keep allocation, rendezvous, stream ordering, Work completion, and workspace reuse
correct. A prototype may reject unsupported inputs explicitly; changes to an
existing backend must preserve its fallback behavior and precision contract.
For each shared buffer or flag in a new protocol, briefly explain:

- The producer and readers, and the ordering making published data visible.
- The completion condition allowing overwrite across successive generations.
- The dependencies ensuring progress under the intended GPU launch schedule,
  including waiting blocks occupying resources needed by producers.

Check primitives and memory-ordering rules in the actual backend and authoritative
documentation. Address virtual aliases when used. Connect collective completion
to the consumer stream and Work object. Preserve cached peer views and captured
graph pointers across workspace growth. A host timeout bounds an experiment;
it does not prove protocol progress.

Store experiments in the repository's scratch area. Keep linting, builds, and
unrelated repository checks off this phase's critical path unless repository
instructions or the requested delivery require them.

### Check essential correctness, then measure

Before timing, check changing position-dependent inputs against a suitable
reference, expected output placement, relevant tails, and repeated workspace reuse.
For reductions, include cancellation or rounding-sensitive inputs. Explicitly
compare replicated outputs across ranks according to the agreement contract;
checking each rank against a tolerant reference alone can miss disagreement.
For sharded outputs, check each rank's expected portion. Check repeatability
separately when required. Benchmark immediately after these checks pass.

Time public calls including staging, synchronization, and output movement.
Warm up the exact path first. Exclude initialization, JIT, graph capture, and
input refresh unless the workload includes them. Do not hide unavoidable per-call
allocation or preparation by moving it into benchmark setup.

For GPU latency, place CUDA events around the call on the execution stream,
accounting for work on other streams through required completion dependencies.
An async launch returning to the host is not completion. Use wall-clock timing
with completion at the window boundary when measuring host-inclusive throughput
or application time, and state the scope. Keep benchmark coordination and result
collection outside measured intervals.

Run GPU experiments sequentially, alternate reference/candidate
order, use maximum-rank latency, and repeat enough to report a median and spread.
Refresh in-place inputs outside timing. Check changed-input replay when measuring
graphs, and keep eager and graph results separate. Bound distributed jobs with
a timeout and stop only their own failed launchers.
Release captured graphs before destroying their groups. Retain raw per-case
measurements and identify skipped or failed cases.

For back-to-back throughput, reproduce the application's buffer reuse and
dependencies. Label the whole-window result separately from isolated-call latency
and keep inputs numerically meaningful across repetitions.

Report the first comparison and its validation limits before expanding scope.
A feasibility request can finish here with reviewable code and measurements.

## Phase 2: investigate or strengthen the result

Enter this phase to explain a loss, resolve noisy or surprising measurements,
pursue another bounded optimization, or validate a candidate for broader use.

When a trace can distinguish hypotheses, add a diagnostic mode to the same
launcher with CPU and CUDA profiler activities. Warm up first, trace a few
representative calls, and export separate files per rank. Preserve intended
streams and dependencies; do not add barriers inside the collective to simplify
the trace. Profile graph replay separately when relevant. CPU range duration is
not GPU execution time. Account for overlap rather than adding parallel durations.

| Observation | Experiment to consider |
| --- | --- |
| Many small launches and host gaps | Fuse work or reduce launches; compare eager and replay to isolate launch overhead. |
| Large staging or output copies | Remove redundant copies or fuse useful work; include newly required application preparation. |
| Wait time dominates | Check arrival imbalance and publication ordering; reduce phases while preserving reuse protection. |
| One peer carries most traffic | Partition chunks or helpers while preserving requested ownership. |
| Kernel execution dominates | Try a bounded tile/warp/block sweep; inspect generated code or resource use when needed. |
| Casts or wide buffers dominate | Convert on load or reduce intermediate traffic while preserving the numerical contract. |
| A size boundary causes a slowdown | Verify dispatch and measure around the threshold. |

A trace suggests hypotheses. Do not infer saturated links from a long kernel or
poor occupancy from low throughput. Use available kernel profiling or counters
when they distinguish hypotheses about traffic, registers, stalls, or occupancy.
Check tool availability and supported flags. Return to unprofiled benchmark runs
for performance decisions.

Keep the best validated candidate before changing it. Use timings, code
inspection, or a targeted trace to choose the next experiment. Change a small
number of related levers. Expand numerical checks and affected backend tests in
proportion to the claimed scope.
Record the hypothesis, settings, correctness, measurement spread, and interpretation
for each experiment. Reprofile after major algorithm changes or unexplained
regressions rather than starting an unbounded sweep.

For an optimization request, continue beyond the first comparison until the
target is met, the agreed experiment budget is exhausted, or evidence supports
diminishing returns. Without a supplied budget, choose and report a bounded
round of experiments. Stop when remaining hypotheses lack a plausible benefit
for the stated objective; explain that limit rather than declaring optimality.
Recheck the selected candidate against the baseline with fresh measurements
after tuning. Select using the stated objective and retain per-case regressions
in the report.

## Validate for the claimed use

Run affected backend tests and extend workload-specific checks using repository
conventions. Cover empty inputs, guarded misaligned views, algorithm boundaries,
mixed-scale reductions, and large-buffer reuse where relevant. Exercise multiple
iterations of persistent kernels. Include exceptional values when the supported
numerical contract requires them.

Test changed-input graph replay when supported. Test async stream dependencies
without global synchronization masking completion failures. Validate concurrent
groups when claiming support, including shared scratch resources. State which
scenarios remain unvalidated.

When overlap is the objective, reproduce relevant compute, streams, dependencies,
arrival imbalance, and collective sequences. Measure application or representative
step time as well as collective latency. Avoid global synchronization that removes
the overlap being evaluated. Investigate resource contention when a faster isolated
collective slows the combined workload. Treat workspace and compute interference
as selection criteria when constrained. Label synthetic overlap experiments as
modeled workloads; reserve application-throughput claims for application measurements.

## Deliver

Provide the selected implementation and reproducible benchmark, supported
specialization and fallback, objective and stopping reason, exact commands,
source hashes, correctness coverage, per-case latency, speedup, variability,
and unsupported scenarios. Define
bandwidth bytes, units, and aggregation; traffic throughput ratios are not
latency speedups. Define speedup as reference latency divided by candidate latency.
Do not count endpoint TX plus RX as unique traffic. Hardware-counter results need
separate measurement windows and idle-traffic checks. Confirm complete reports,
clean exit, and unchanged measured sources. Report failure to improve as a valid
outcome, with evidence about the limiting cost. Leave results reviewable without
publishing or committing unless authorized.
