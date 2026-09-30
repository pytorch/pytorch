# Distributed development

- Never use sleeps to establish distributed/race ordering. Use events, barriers,
  futures, or handshakes; timeouts bound failure, not synchronization.
- Keep top-level APIs backend-generic: no hardcoded `"nccl"` checks or vendor
  branches in `distributed_c10d.py`. Dispatch through general backend interfaces;
  keep implementation and capability checks in backends, including custom ones.
  Extensions may use `dist.get_backend_impl(...).foo(...)`, which bypasses tracing
  and other hooks.
- Support CUDA graphs where possible. Test collective capture and repeated replay,
  not just eager execution; explicitly document unsupported paths.
- Keep multi-GPU tests fast. Prefer `MultiThreadedTestCase` or
  `MultiProcContinuousTest` over per-test process launches where semantics allow.
  Reset shared worker state; retain fresh processes for lifecycle/failure tests
  or state that cannot be isolated safely.
- Put distributed-specific tests in `test/distributed/`; prefer backend-generic
  suites unless the behavior is backend-specific.

## Distributed review checks

- Check collective order and participants across ranks and process groups,
  including divergent branches, empty inputs, nonmembers, and exceptions.
  Verify roots, rank mappings, shapes/dtypes, split sizes, and matching P2P tags.
- Check that waits cannot block collective progress; rank failure, timeout,
  cancellation, reconfiguration, and teardown must not strand peers or futures.
- Retain buffers, registrations, communicators, and callback state until native
  work completes. Timeout/cancellation does not prove DMA or remote access stopped.
- Preserve collective effects, ordering, configuration, and async semantics through
  functional operators, compilation/export, and backward; do not silently drop them.
