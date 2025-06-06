
# AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models

```{warning}
AOTInductor and its related features are in prototype status and are subject to backwards compatibility breaking changes.
```

AOTInductor is a specialized version of {doc}`TorchInductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`, designed to process exported PyTorch models, optimize them, and produce shared libraries as well as other relevant artifacts. These compiled artifacts
