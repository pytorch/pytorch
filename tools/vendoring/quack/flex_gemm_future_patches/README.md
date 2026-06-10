# Deferred FlexGEMM QuACK patches

`tools/vendoring/quack/vendor.sh` does not apply patches from this directory.
They are kept here as the rebased backlog for later FlexGEMM PRs so the active
vendored patchset can stay scoped to the current PyTorch adapter.

Source of truth for when to move pieces into the active patchset: the FlexGEMM upstream landing plan referenced by the workspace prime file.

Base active endpoint: `2e952bf23ff0cbe04fb389bce13d59ed92e68603`
Deferred endpoint: `1328c04ada1a306a74774d0237d6b3a5d1282f57`

## Deferred commits

```text
bb25750 Simplify QUACK epilogue worker plumbing
3d79c76 Accept blocked MXFP8 epilogue scales
f262fe4 Add SM100 dynamic-N grouped GEMM epilogues
c0fd44a Tune dynamic-N grouped epilogue defaults
85b9dbb Tighten varlen-N epilogue invariants
fe68398 Use deterministic varlen-N configs
25d6272 Support captured auxes in blockscaled epilogues
bf1fdba Support NVFP4 scaled epilogues
33299f4 Test NVFP4 scaled epilogues with real scales
64b9206 Apply global scales in blockscaled epilogues
a30f336 Use tuple aux lists for blockscaled epilogues
72cf62e Stress blockscaled epilogue aux lists
bb4064f Add heavier blockscaled aux stress tests
3c05c0d Support MXFP8 varlen scaled epilogues
6f47885 Use make_tensor for varlen scaled inputs
192c8c9 Cover MXFP8 epilogues with nonunit scales
1328c04 Adapt MXFP8 varlen scales for public grouped layout
```
