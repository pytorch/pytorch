# Distributed development

Apply this guidance to distributed Python APIs and their associated C++ backends
and tests. Read the repository-level instructions as well.

## Synchronization

- Do not use sleeps to establish ordering or make distributed/race tests pass
  (`time.sleep`, `asyncio.sleep`, `torch.cuda._sleep`, etc.). Use events, barriers,
  futures, or explicit handshakes that prove the required state was reached.
- Synchronization must involve the correct participants. Do not add a collective
  barrier on a path that another rank can skip or exit through an exception.
- Bound waits and propagate failures. A timeout is a failure bound, not proof of
  synchronization. For unavoidable production polling, use a bounded,
  cancellation-aware wait/backoff; never rely on its delay for correctness.

## Backend boundaries

- Keep top-level interfaces backend-generic. Do not hardcode backend names such
  as `"nccl"` or backend-specific branches in `distributed_c10d.py` to implement a
  feature or validate its support. Define a general interface and dispatch to
  the selected backend; keep backend-specific logic in backend implementations.
- Express support through backend interfaces/capabilities, not a Python allowlist
  of built-in backends. Unsupported operations must fail clearly rather than
  silently ignore arguments. Preserve support for third-party backends.
- Experimental backend extensions may use `dist.get_backend_impl(...).foo(...)`.
  This is an explicit escape hatch, not a reason to add vendor-specific public
  wrappers. It bypasses tracing and other hooks; do not claim compile/export
  support without implementing and testing it.

## Changes and validation

- Make minimal, reviewable changes. Preserve existing public behavior and
  compatibility unless a breaking change was explicitly requested. Check callers
  and downstream usage when changing interfaces.
- Prefer early returns/continues over deeply nested control flow.
- Put distributed-specific tests under `test/distributed/`, not shared unrelated
  test files. Use backend-generic suites unless testing backend-specific behavior.
- Use existing test harnesses, `TestCase.assertEqual`, and parametrization. Test
  supported CPU/device paths and unavailable-backend/insufficient-device paths.
- Restore process groups, environment variables, default devices, global flags,
  and other test state. Reused workers must not leak state between methods.
- Build affected components, run lint and targeted tests, and report exact
  commands/results and untested configurations. Enable only required backends.
  Fix regressions introduced by the change, not unrelated CI failures.
- Keep each stacked commit independently buildable/testable. PR descriptions
  should explain the issue, fix, and a concise `Test plan:`.

## Distributed review checklist

- **Rank agreement:** Check collective order, participants, group membership,
  rank mapping, roots, tensor shapes/dtypes/devices, split sizes, and tags. Follow
  divergent branches, empty inputs, nonmembers, and exceptional paths across ranks.
- **Deadlocks and progress:** Check lock ordering, callbacks under locks, GIL
  interactions, background progress, and whether a wait blocks the thread/stream
  required to complete it. Check cross-process-group operation ordering too.
- **Failure and teardown:** Check timeout budgets, cancellation, partial startup,
  rank failure, abort/reconfigure, and shutdown ordering. Ensure waiters/futures
  resolve with errors and cleanup neither hangs nor masks the original failure.
- **Async lifetime:** Keep tensors, buffers, registrations, communicators, stores,
  and callback state alive until native work really completes. Timeout/cancellation
  does not itself prove that DMA or remote access has stopped.
- **Accelerators:** Check device guards on every thread, producer/consumer stream
  dependencies, allocator lifetime, and the distinction between enqueue, stream
  completion, and host completion. Check graph capture/replay ownership when supported.
- **Compilation/autograd:** Preserve collective effects, ordering, configuration,
  and async semantics through functional operators, fake/meta kernels, tracing,
  export, compiler lowering, and backward. Test supported modes; reject unsupported
  modes explicitly rather than silently dropping behavior.
- **Coverage:** Require deterministic reproductions of races and negative paths.
  Check third-party/multiple-backend dispatch and fresh versus reused process state.
  Performance changes must preserve test identities, outcomes, and coverage; do
  not replace synchronization with timing assumptions or remove difficult cases.
