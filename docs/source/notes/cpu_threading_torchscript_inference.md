(cpu-threading-torchscript-inference)=

# CPU threading and TorchScript inference

```{versionchanged} 2.16
The deprecated public TorchScript compilation, tracing, save/load, and fusion
entrypoints have been removed. Shared native runtime support remains available
for existing consumers, but `torch.load` no longer accepts TorchScript archives.
Migrate the original Python model to `torch.export`; use an older PyTorch version
if you need to load an existing legacy archive.
```

```{warning}
TorchScript is deprecated, please use
[torch.export](https://docs.pytorch.org/docs/stable/export.html) instead.
```
