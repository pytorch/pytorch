# FlashAttention-4 vendoring

PyTorch vendors FlashAttention-4 together with the small QuACK utility subset it
uses. Both revisions are pinned in `vendor.sh` so FA4 does not share the separate
PyTorch QuACK pin used by FlexGEMM.

Refresh the vendored dependency universe with:

```bash
tools/vendoring/flash_attn4/vendor.sh
```

Verify reproducibility with local upstream checkouts:

```bash
tools/vendoring/flash_attn4/vendor.sh --check \
  --src-fa /path/to/flash-attention \
  --src-quack /path/to/quack
```

The script copies the FA4 CuTe package and only the QuACK utility modules reached
by FA4. It rewrites all imports into `torch._vendor.flash_attn`, adds the required
modification notice to changed Apache-licensed sources, rejects remaining external
`flash_attn` or `quack` imports, and verifies the relative-import closure.
