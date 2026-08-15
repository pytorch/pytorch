# Vendored libraries

## `packaging`

Source: https://github.com/pypa/packaging/

PyPI: https://pypi.org/project/packaging/

Vendored version: `23.2.0`

Instructions to update:

- Copy the file `packaging/version.py` and all files that it is depending on
- Check if the licensing has changed from the BSD / Apache dual licensing and update the license files accordingly

## `quack`

This is a subset of the full quack library containing the RMSNorm implementation
and its transitive dependencies.

After copying the upstream subset, `tools/vendoring/quack/patches` applies
PyTorch-only vendoring/runtime changes such as cache/worker namespace renames
and removal of RMSNorm custom-op registration.

Source: https://github.com/Dao-AILab/quack

The pinned upstream commit is the `PINNED_SHA` constant in
`tools/vendoring/quack/vendor.sh` (`__version__` in the generated vendored
package records the upstream version). That constant is the single source of
truth; do not duplicate the pin here. The vendoring script verifies that the
pinned commit is reachable from Dao-AILab/quack main before applying local
PyTorch vendoring patches.

FlexGEMM separately uses the full external QuACK package. Its public base is
pinned in `.github/ci_commit_pins/quack.txt`, and
`tools/vendoring/quack/prepare_flex_gemm.sh` applies the ordered
`tools/vendoring/quack/flex_gemm_patches/series` before CI installs the package.
Those feature patches do not participate in rendering the RMSNorm-only vendored
subset.

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
- Add PyTorch-only vendoring/runtime deltas to `tools/vendoring/quack/patches`
