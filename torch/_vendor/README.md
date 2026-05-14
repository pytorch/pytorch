# Vendored libraries

## `packaging`

Source: https://github.com/pypa/packaging/

PyPI: https://pypi.org/project/packaging/

Vendored version: `23.2.0`

Instructions to update:

- Copy the file `packaging/version.py` and all files that it is depending on
- Check if the licensing has changed from the BSD / Apache dual licensing and update the license files accordingly

## `quack`

This is a subset of the full quack library, currently vendoring the following operators:

- RMSNorm

Note: There are a couple of patchsets applied to make the library vendorable - at a high level:
- Change exports from absolute `quack.module` to relative `.module`
- Remove `@custom_op` and `@register_fake` decorators as we don't want to register custom ops here.

Source: https://github.com/Dao-AILab/quack

Vendored version: `0.4.0`
Vendored SHA: `6bceaad2dba3b979b898824b146b1bb2816fc483`

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
