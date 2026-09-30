# Distributed development

- Avoid sleeps in distributed/race code; synchronize with events, barriers,
  futures, or handshakes instead of timing assumptions.
- Keep top-level interfaces backend-generic: no hardcoded `"nccl"` behavior in
  `distributed_c10d.py`. Put backend-specific logic in backend implementations;
  custom extensions may use `dist.get_backend_impl(...).foo(...)`.
- Support CUDA graphs where possible; test capture and repeated replay.
- Keep expensive multi-GPU tests fast: prefer `MultiThreadedTestCase` or
  `MultiProcContinuousTest` over per-test process launches where possible.
  Reset reused-worker state; retain fresh processes when isolation is required.
- Put distributed-specific tests in `test/distributed/`; use backend-generic
  suites unless testing backend-specific behavior.

## Review

- Check collective order/participants across ranks and groups, matching P2P tags,
  rank mappings, and tensor metadata, including divergent/error paths.
- Check progress and peer/future cleanup on failure, timeout, cancellation,
  reconfiguration, and teardown; avoid waits that block their own progress.
- Keep buffers/registrations/communicators alive until native work completes;
  timeout/cancellation does not imply DMA or remote access has stopped.
- Preserve collective effects, ordering, configuration, and async semantics
  through compilation/export and backward.
