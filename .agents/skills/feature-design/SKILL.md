---
name: feature-design
description: Principles for designing PyTorch features and choosing between competing designs, especially designs that add invariants, user-facing contracts, deferred or asynchronous behavior, or special modes. Use when designing a feature, writing the design section of an RFC or PR description, comparing alternative designs, or judging whether a PR's design is sound. Includes situation-specific guidance for CUDA streams, the caching allocator, and CUDA graph capture.
---

# Feature Design

These principles are distilled from how maintainers worked through competing
designs for real features (see [case-studies/](case-studies/)). They are
principles, not rules: each one can be broken for good reason. A design that
breaks one should say so and say why. In review, a broken principle is a
discussion point, not a blocker by itself.

The guidance is split in two:

- **Generic principles** (this file) apply to most features.
- **Situation-specific guidance** (separate files, indexed below) applies only
  when the feature touches that area. Don't apply it outside its stated scope,
  and don't move situation-specific advice into the generic sections.

## Generic principles

### Weigh preconditions with the cost of their violation

It is often possible (and necessary) to simplify an API by requiring a user
fulfill some precondition.  But this means a user can, through inattention or
cognitive overload, violate this contract.  Reason through what the outcome of
violated preconditions is.

Here is a rough taxonomy of consequences, generated from the meta-rule that
you grade a consequence by how difficult it is for the user to identify that
it has happened:

- **Non-determinism.**  A non-deterministic error is difficult to detect (it
  doesn't happen that often) and difficult to reproduce (a small reproducer
  may not trigger the race).  A non-deterministic crash harms goodput (as you
  have to restart a training job), a non-deterministic numeric incorrectness
  can harm loss curve convergence.  Non-determinism like this is especially
  harmful in large scale training runs, and many people will enforce
  non-determinism even for otherwise benign nondeterminism (like slightly
  different numerics from CUDA atomics.)

- **Numeric incorrectness.**  Incorrect numerics are difficult to detect as
  there is often no golden numerics to compare against: you can only tell
  there is a problem because your loss curve doesn't look as good as it should
  be.  In a situation where a bitwise equivalence test is possible, numeric
  incorrectness can be detected fairly easily, but this is often not possible
  since there may be multiple acceptable numerics for a function
  implementation.

- **Extra wall time.**  If you degrade performance, this can be difficult
  to diagnose, especially if the performance degradation only shows in realistic
  cluster setups (where you have to run an expensive benchmark run to reproduce
  the problem).  However, wall time slowdown never blocks a run, and a slight
  regression here may not be a dealbreaker.  However, because it's not obvious
  that something is running slower than usual, silent performance slowdown may
  affect users without them noticing it longer than you would hope.  Performance
  slowdown that manifests as extra kernel launches is easier to diagnose as it
  more clearly shows in profiles.  APIs that can induce extremely bad performance
  should be treated as bad as errors (10% difference doesn't matter much, but
  an unnecessary device-to-host syncs or distributed collectives matter a lot.)

- **Extra memory usage.**  This is not too bad, especially if there is a way
  to reduce the memory usage.  In many situation, excess memory usage can be
  inexpensively reproduced on a single GPU using things like fake PG;
  additionally, if there is headroom extra memory usage can be harmless as
  long as the user job does not OOM.  However, if your feature is silently
  using extra memory (and especially if it is using a lot of extra memory),
  the user needs to know to use memory profiling to diagnose the problem,
  and it sometimes can be non-obvious what the retainer for some memory is.
  This is less of a problem in the AI agent era, but worth thinking about.
  Use of `recordStream` can cause non-deterministic memory usage because it
  ties the deallocation of memory to CUDA events becoming visible on CPU--this
  is why don't recommend usage of this feature!

- **Crash by raising an error.**  Crashes (especially non-deterministic ones)
  are serious in the sense that they prevent any progress at all (similar to
  an OOM), but they are also very obvious and preferable to difficult to
  detect situations.  Whether or not to crash or degrade performance is a
  difficult question, and in some cases we offer different modes for users to
  turn on errors to flush out problems.

### Build on contracts users already follow

Before asking users for a new obligation, check whether an existing one already
covers the case: the caching allocator's rule that uses are synced back to the
allocation stream before freeing, autograd's version counter, the rule that
in-place ops on inputs need `mark_dirty`, and so on. A design that is safe
under rules people have followed for years beats one with a new rule, however
well documented. A new obligation also becomes part of the contract of every
API built on top of yours.

### An operation's obligations should match what it looks like

If something looks like a read, it shouldn't need the synchronization of a
write. If it looks synchronous, it shouldn't be secretly asynchronous in some
cases. This is especially prevalent in C++, where it's expected that read-like
operations can be safely called in a multithreaded fashion without exposing races.

### One mechanism across modes

Prefer a primitive whose meaning is already correct in every mode (eager,
CUDA graph capture, tracing, compile, inference mode) over a branch that
special-cases a mode with "this doesn't matter in mode M". Mode-specific
branches are where the bugs concentrate, because each one is a separate
correctness argument that is rarely re-checked.

### Don't build on information you can't get right

If a signal is only approximately right (the current stream when a pointer is
taken, whether a pointer access is a read or a write, which tensor is
"original"), don't let correctness depend on it. Use it only in the direction
where being wrong is safe: to fall back to the slow path or invalidate an
optimization, never to skip a synchronization or a copy.

### Narrow the promise until you can keep it

Prefer a few small guarantees that can be checked over a sweeping semantic
claim like "behaves exactly as if X". Say explicitly what is not guaranteed. A
broad promise you can't keep in every case is worse than a narrow one users
can rely on.

### We don't believe in unrestricted undefined behavior

When a case is hard to support correctly, choose between two outcomes:

- **Fall back** to a slower path when it gives the same observable result
  (for example, copy eagerly instead of lazily).
- **Raise an error** when the fallback would change semantics. Make the error
  actionable: name the change at the user's call site that fixes it.

Silently doing something different is never one of the options.

More typical situations where we might rely on UB are for things like, the
fast implementation of an operation relies on hardware specific guarantees.
But just because you overflowed an integer doesn't give you license to have
arbitrary behavior.  We work with our hardware, and we override the hardware
behavior only when there is a good reason.

### Design for the common case, and drop special cases you can't justify

Don't let a rare, scary case shape the whole design. Make the common case
simple and the rare case explicit, even if that means more work for the rare
user. Drop an optimization or special case whose precondition can't be checked
at the point where it's needed, even if it motivated the work. Asking users to
do more is fine as long as what you ask for is clear.

### Don't mutate shared state that has escaped

Metadata shared by several references (a context shared by aliases, a cache
entry, a registry record) should be immutable once other references can see
it. Mutating it changes the contract under references that were created under
the old one. If it seems to need changing, create a new one instead.

### Keep the blast radius small

Limit behavior changes to users who opt in (an explicit API, one backend, one
device type). Leave other paths unchanged, even if their behavior now differs,
and say so in the design. "CPU is unchanged" is a feature.

## Process principles

- **Settle arguments with concrete scenarios.** A specific program, stream
  interleaving, or call sequence beats a claim that something is safe or
  unsafe. Keep each counterexample as a regression test.
- **Treat a stream of found holes as a design signal.** When adversarial review
  keeps finding new bugs of the same kind, stop patching and question the
  abstraction. A fix that also makes the code simpler is a sign the earlier
  design was wrong.
- **Revisit why the feature exists.** The motivation decides which constraints
  you can impose on users. If the feature is not meant for some pattern, it is
  fine to make that pattern expensive or an error.
- **Prototype competing designs and compare their failure modes.** Prototypes
  are cheap, especially with agents. Choose between designs by how they fail
  (principle 1), not by which one passes the tests.
- **Don't treat a clean review as proof.** Neither a human nor an agent finding
  no races shows there are none. For concurrency, prefer designs where a missed
  case costs performance, not correctness.
- **Know which parts you understand and which parts you delegated.** If fixes
  to a design are being written without anyone fully understanding why the
  design is correct, step back before continuing.
- **Settle the design before polishing the prose.** Wording of docs and
  comments can be fixed after the design is accepted; don't let it hold up
  agreement on the design.

## Situation-specific guidance

Read the file for a situation only when its scope applies.

| Situation | Applies when the feature... | File |
|---|---|---|
| CUDA streams, the caching allocator, and CUDA graph capture | defers, moves, shares, or reuses GPU memory; adds hooks on `data_ptr()` / storage access for CUDA; adds cross-stream synchronization; or must work during CUDA graph capture | [cuda-async.md](cuda-async.md) |

For deciding whether a specific `record_stream` call is avoidable, use the
`cuda-streams` skill.

When you find advice that only holds in one area, add a file here with a clear
"applies when" scope rather than adding it to the generic principles.

## Case studies

- [case-studies/cuda-lazy-clone.md](case-studies/cuda-lazy-clone.md): making
  CUDA copy-on-write (`_lazy_clone`) safe with streams and CUDA graphs, with
  five rejected designs and the one that was accepted.
