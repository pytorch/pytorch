# Reporting Issues

If the provided workarounds were not enough to get `torch.compile` working,
then you should consider reporting the issue to PyTorch.
But there are a few things that you can do to make our lives significantly easier.

## Ablation

Check which component of the `torch.compile` stack is the one causing the issue using the `backend=` option for `torch.compile`.
In particular, try:

- `torch.compile(fn, backend="eager")`, which only runs TorchDynamo, the graph capture component of `torch.compile`.
- `torch.compile(fn, backend="aot_eager")`, which runs TorchDynamo and AOTAutograd, which additionally generates the backward graph during compilation.
- `torch.compile(fn, backend="aot_eager_decomp_partition")`, which runs TorchDynamo and AOTAutograd with operator decompositions/partitions.
- `torch.compile(fn, backend="inductor")`, which runs TorchDynamo, AOTAutograd, and TorchInductor, the backend ML compiler that generates compiled kernels.

If you only fail with the Inductor backend, you can additionally test various Inductor modes:

- `torch.compile(fn, backend="inductor", mode="default")`
- `torch.compile(fn, backend="inductor", mode="reduce-overhead")`
- `torch.compile(fn, backend="inductor", mode="max-autotune")`

You can also check if dynamic shapes is causing issues with any backend:

- `torch.compile(fn, dynamic=True)` (always use dynamic shapes)
- `torch.compile(fn, dynamic=False)` (never use dynamic shapes)
- `torch.compile(fn, dynamic=None)` (automatic dynamic shapes)

## Bisecting

Did you try on the latest nightly? Did something work in the past but now no longer works?
Can you bisect to determine the first nightly where your issue occurs?
Bisecting is especially helpful for performance, accuracy, or compile time regressions,
where it is not immediately obvious where the problem originates from.

## Creating a reproducer

Creating reproducers is a lot of work, and it is perfectly fine if you do not have the time to do it.
However, if you are a motivated user unfamiliar with the internals of `torch.compile`,
creating a standalone reproducer can have a huge impact on our ability to fix the bug.
Without a reproducer, your bug report must contain enough information for us to identify the root cause of the problem and write a reproducer from scratch.

Here's a list of useful reproducers, ranked from most to least preferred:

1. **Self-contained, small reproducer:** A script with no external dependencies, under 100 lines of code, that reproduces the problem when run.
2. **Self-contained, large reproducer:** Even if it's large, being self-contained is a huge advantage!
3. **Non-self-contained reproducer with manageable dependencies:**
   For example, if you can reproduce the problem by running a script after `pip install transformers`,
   that's manageable. We can likely run it and investigate.
4. **Non-self-contained reproducer requiring substantial setup:** This might involve downloading datasets,
   multiple environment setup steps, or specific system library versions requiring a Docker image.
   The more complex the setup, the harder it is for us to recreate the environment.

:::{note}
Docker simplifies setup but complicates changes to the environment, so it's not a perfect solution, though we'll use it if necessary.
:::

If possible, try to make your reproducer single-process, as those are easier to debug than a multi-process reproducer.

Additionally, below is a non-exhaustive list of aspects to check in your
issue that you can attempt to replicate in your reproducer:

- **Autograd**. Did you have tensor inputs with `requires_grad=True`? Did you call `backward()` on the output?
- **Dynamic shapes**. Did you set `dynamic=True`? Or did you run the test code multiple times with varying shapes?
- **Custom operators**. Is there a custom operator involved in the real workflow?
  Can you replicate some of its important characteristics using the Python custom operator API?
- **Configuration**. Did you set all the same configuration?
  This includes `torch._dynamo.config` and `torch._inductor.config` settings,
  as well as arguments to `torch.compile` like `backend` / `mode`.
- **Context managers**. Did you replicate any active context managers?
  This could be `torch.no_grad`, automatic mixed precision, `TorchFunctionMode` / `TorchDispatchMode`,
  activation checkpointing, compiled autograd etc.
- **Tensor subclasses**. Is there a tensor subclass involved?
