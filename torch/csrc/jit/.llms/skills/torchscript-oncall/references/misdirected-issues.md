# Misdirected Issues

~35% of issues filed against the TorchScript oncall are not TorchScript bugs. Identify and redirect these quickly.

## Identification Guide

| Category | Count | Signals | Redirect To |
|---|---|---|---|
| FX tracing issues | 8+ | `torch.fx.wrap` not respected, proxy objects, "symbolically traced variables" errors | FX / torch.compile oncall |
| Custom op / infrastructure | 5+ | Thrift/RPC failures in custom ops, missing CUDA backends, model preloading timeouts | Custom op owner or infra oncall |
| torch.compile / TorchDynamo | 2+ | User applied `@torch.jit.ignore` to a `torch.compile` error (often from ChatGPT advice) | torch.compile oncall |
| ONNX export | 1+ | TS-to-ONNX path broke | Recommend bypassing TorchScript entirely; export from `nn.Module` directly |
| Cinder / lazy imports | 2+ | `lazy_import3` breaking torch module initialization, `_LazyImportWrap` errors | Cinder team |
| Build system | 2+ | Missing Buck targets, AMD vs NVIDIA build gaps | Build system / platform oncall |

## Quick Checks

1. **Does the error mention `torch.fx`, `Proxy`, or "symbolically traced"?** → FX issue, not TorchScript
2. **Is the user using `torch.compile` or `torchdynamo`?** → torch.compile oncall
3. **Does the stack trace point to a custom op (thrift, FBGEMM, etc.)?** → Custom op owner
4. **Is this an ONNX export failure?** → Recommend exporting from eager `nn.Module` directly
5. **Does the error involve `lazy_import` or `_LazyImportWrap`?** → Cinder team
6. **Is this a missing Buck target or build configuration issue?** → Build/platform team
