What is torchgen/packaged?
--------------------------

This directory is a collection of files that have been mirrored from their
original locations. setup.py is responsible for performing the mirroring.

These files are necessary config files (e.g. `native_functions.yaml`) for
torchgen to do its job; we mirror them over so that they can be packaged
and distributed with torchgen.

Ideally the source of truth of these files exists just in torchgen (and not
elsewhere), but getting to that point is a bit difficult due to needing to
deal with merge conflicts, multiple build systems, etc. We aspire towards
this for the future, though.

Note well that although we bundle torchgen with PyTorch, there are NO
BC guarantees: use it at your own risk. If you're a user and want to use it,
please reach out to us on GitHub.
