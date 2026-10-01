# Distributed development

- Avoid sleeps in code and tests; synchronize with events, barriers,
  futures, or handshakes instead of timing assumptions.
- Keep top-level interfaces backend-generic: no hardcoded `"nccl"` behavior in
  `distributed_c10d.py`. Put backend-specific logic in backend implementations;
  custom extensions may use `dist.get_backend_impl(...).foo(...)`.
- Write accelerator-generic code where possible; avoid CUDA-only assumptions.
- Provide clear errors, graceful recovery, and bounded timeouts where possible.
- For security, avoid pickle where possible; prefer `torch.load(..., weights_only=True)`
  when deserializing checkpoints.
- Support CUDA graphs where possible.

## Testing

- Test CUDA graph capture and repeated replay where supported.
- Keep expensive multi-GPU tests fast: prefer `MultiThreadedTestCase` or
  `MultiProcContinuousTest` over per-test process launches where possible.
  Reset reused-worker state; retain fresh processes when isolation is required.
- Put distributed-specific tests in `test/distributed/`; use backend-generic
  suites unless testing backend-specific behavior.
