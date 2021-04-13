# Lazy Tensors Core

1. Clone under your copy of the PyTorch repo (assuming you've already built it from source).
1. Install glob2 and the Lark parser, used for automatic code generation:

```bash
pip install glob2 lark-parser
```

1. Run `python setup.py develop`.
1. Run `example.py`. It'll report that no lazy tensor backend is registered.

Suggested build environment:

```bash
export CC=clang-8
export CXX=clang++-8
export BUILD_CPP_TESTS=0
```

If you want to debug it as well:

```bash
export DEBUG=1
```
