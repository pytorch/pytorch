# Vendored libraries

## `packaging`

Source: https://github.com/pypa/packaging/

PyPI: https://pypi.org/project/packaging/

Vendored version: `23.2.0`

Instructions to update:

- Copy the file `packaging/version.py` and all files that it is depending on
- Check if the licensing has changed from the BSD / Apache dual licensing and update the license files accordingly

## `quack`

This is a subset of the full quack library, currently vendoring RMSNorm, the
EpiMod GEMM runtime used by `torch._inductor.kernel.flex_gemm`, and the
symmetric GEMM, together with their transitive dependencies.

`tools/vendoring/quack/flex_gemm_patches/series` is applied to the pristine
upstream checkout before copying. The patches are git-format against the
upstream repository layout (`quack/`, `tests/`) so they remain directly
upstreamable. After copying, the vendoring script mechanically rewrites every
`quack` package reference: imports become absolute `torch._vendor.quack`
imports, and `torch.library` op namespaces, the autotuner package name, and the
on-disk cache name get a `torch_vendor_quack` prefix so the copy cannot collide
with a pip-installed `quack`.

Source: https://github.com/Dao-AILab/quack

The vendored subset's pinned upstream commit is the `PINNED_SHA` constant in
`tools/vendoring/quack/vendor.sh` (`__version__` in the generated vendored
package records the upstream version). That constant is the single source of
truth; do not duplicate the pin here. The vendoring script verifies that the
pinned commit is reachable from Dao-AILab/quack main before applying local
patches.

Instructions to update:

Edit `PINNED_SHA` in `tools/vendoring/quack/vendor.sh` to the new commit, then
re-render (no SHA is passed):

```
tools/vendoring/quack/vendor.sh

# Or, to reuse an existing local clone instead of fetching:

tools/vendoring/quack/vendor.sh --src /path/to/local/quack
```

Instructions to update the subset of quack being vendored:

- In the `vendor.sh script`:
  - Update the files to be copied (`FILES`)
  - Update `rewrite_package_references` if the mechanical rewrite misses a
    new `quack` reference form
- Add FlexGEMM deltas as git-format patches to
  `tools/vendoring/quack/flex_gemm_patches` and list them in `series`
