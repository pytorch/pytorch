# Vendored libraries

## `packaging`

Source: https://github.com/pypa/packaging/

PyPI: https://pypi.org/project/packaging/

Vendored version: `23.2.0`

Instructions to update:

- Copy the file `packaging/version.py` and all files that it is depending on
- Check if the licensing has changed from the BSD / Apache dual licensing and update the license files accordingly

## `quack`

This is a subset of the full quack library, currently vendoring the following implementation paths:

- RMSNorm
- Lower-level GEMM epilogue implementation dependencies used by PyTorch-owned adapters

Note: two patch phases are applied after copying the upstream subset:
- `tools/vendoring/quack/flex_gemm_patches`: FlexGEMM QuACK feature deltas that are not yet merged into Dao-AILab/quack main
- `tools/vendoring/quack/patches`: PyTorch-only vendoring/runtime changes, such as relative imports, cache/worker namespace renames, and removal of RMSNorm custom-op registration

Source: https://github.com/Dao-AILab/quack

The pinned upstream commit is the `PINNED_SHA` constant in
`tools/vendoring/quack/vendor.sh` (`__version__` in the generated vendored
package records the upstream version). That constant is the single source of
truth; do not duplicate the pin here. The vendoring script verifies that the
pinned commit is reachable from Dao-AILab/quack main before applying local
FlexGEMM patches.

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
  - Update the `rewrite_imports` methods is there are more patterns required
- Add QuACK feature deltas needed for FlexGEMM to `tools/vendoring/quack/flex_gemm_patches`
- Add PyTorch-only vendoring/runtime deltas to `tools/vendoring/quack/patches`
