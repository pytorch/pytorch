(miscellaneous_environment_variables)=

# Miscellaneous Environment Variables

| Variable                              | Description |
|---------------------------------------|-------------|
| `TORCH_FORCE_WEIGHTS_ONLY_LOAD`       | If set to [`1`, `y`, `yes`, `true`], the `torch.load` will use `weights_only=True`. This will happen even if `weights_only=False` was passed at the callsite. For more documentation on this, see [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html). |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`    | If set to [`1`, `y`, `yes`, `true`], the `torch.load` will use `weights_only=False` if the `weights_only` variable was not passed at the callsite. For more documentation on this, see [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html). |
| `TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT`  | Under some conditions, autograd threads can hang on shutdown, therefore we do not wait for them to shutdown indefinitely but rely on a timeout that is by default set to `10` seconds. This environment variable can be used to set the timeout in seconds. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD`       | If set to `1`, out-of-tree backend extensions will be automatically imported when running `import torch`. |
