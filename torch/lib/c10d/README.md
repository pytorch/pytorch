# THD refactor

This is a work in progress. It is separate from the main THD directory
to avoid disrupting THD users or have to deal with backwards compat
early on. Once this gets to a usable state, we'll add Python bindings
and a compat layer.

See https://github.com/pytorch/pytorch/issues/7434 for the main issue.

This tree is intentionally not part of the main build and will be
buildable/testable in isolation, as long as ATen is available in
`<repository root>/torch/lib/tmp_install`.

To build and install ATen here, navigate to the root of this
repository and run:

``` shell
tools/build_pytorch_libs.sh --with-cuda ATen
```
