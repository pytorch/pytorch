# Optimizer review guidelines

Do not use `is_compiling` checks (e.g., `torch.compiler.is_compiling()` or
`torch._dynamo.is_compiling()`) in `torch.optim` library code.
