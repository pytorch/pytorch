## Summary

Add a `register_hook` API to `leaf_function` that allows registering backward hooks on leaf functions. This enables side-effect-only leaf functions (e.g., debug logging) to also execute during the backward pass without requiring the caller to capture the return value.

The hook function has the same signature as the leaf function, but each tensor argument receives the corresponding gradient instead of the original tensor. Non-tensor arguments are passed through unchanged. The hook is itself invoked as a leaf function, so it remains opaque to the compiler. Hooks are registered on the outer (pre-detach) tensors in `InvokeLeafFunctionAutogradOp.forward`, so they fire when autograd computes gradients in the enclosing graph. This works across eager, `torch.compile`, and `aot_function` paths.

As an example use case, `torch.utils.debug_log` provides `debug_log` and `debug_log_rank` utilities that print tensor norms during forward and backward, useful for debugging training numerics. `debug_log` uses `torch._higher_order_ops.print` for stdout output; `debug_log_rank` uses Python logging with optional rank filtering for distributed training.

Start by reading `torch/_dynamo/decorators.py` (the `register_hook` API and docstring), then `torch/_higher_order_ops/invoke_leaf_function.py` (the hook registration in autograd), then `torch/utils/debug_log.py` (the example usage).

## Test plan

- `python -m pytest test/higher_order_ops/test_debug_log.py -xvs` (22 passed, 1 xfailed)
- Tests cover eager, `torch.compile` (aot_eager), and `aot_function` paths for both `debug_log` and `debug_log_rank`
- Tests verify forward/backward logs appear, FX graphs contain `invoke_leaf_function` nodes, gradients are correct, and rank filtering works
- Expected failure test documents that DTensor cannot be passed directly (HOP has no DTensor dispatch rule)

Authored with Claude.
