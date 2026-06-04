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

Note: There are a couple of patchsets applied to make the library vendorable - at a high level:
- Change exports from absolute `quack.module` to relative `.module`
- Rename cache directories / worker module paths so this copy is independent of any external `quack` package
- Remove custom-op registration from RMSNorm; PyTorch owns the public operator adapters

Source: https://github.com/Dao-AILab/quack

The vendored version and upstream SHA are recorded in
`torch/_vendor/quack/__init__.py` (the `Upstream SHA` line and `__version__`),
which `vendor.sh` writes on each run. That file is the single source of truth;
do not duplicate the pin here.

Instructions to update:

Run the following script:

```
tools/vendoring/quack/vendor.sh <NEW_SHA>

# Or if you have an existing local clone:

tools/vendoring/quack/vendor.sh <NEW_SHA> /path/to/local/quack
```

Instructions to update the subset of quack being vendored:

- In the `vendor.sh script`:
  - Update the files to be copied (`FILES`)
  - Update the `rewrite_imports` methods is there are more patterns required
- Add any extra patchsets needed in `tools/vendoring/quack/patches`
