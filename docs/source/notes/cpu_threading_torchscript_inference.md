(cpu-threading-torchscript-inference)=

# CPU threading and TorchScript inference

```{versionchanged} 2.16
The deprecated public TorchScript compilation, tracing, save/load, and fusion
entrypoints have been removed. Shared runtime support remains available for
existing consumers. Trusted legacy TorchScript archives can still be loaded
through `torch.load(..., weights_only=False)`; use `torch.export` for new models.
```

```{warning}
TorchScript is deprecated, please use
[torch.export](https://docs.pytorch.org/docs/stable/export.html) instead.
```
