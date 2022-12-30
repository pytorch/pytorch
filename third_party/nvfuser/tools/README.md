# cpp-repro-gen.py

## Usage

Prepare a python file containing only the fusion definition, for example `examples/repro.py`. Then run

```
python cpp-repro-gen.py < examples/repro.py > examples/repro.cpp
```

to get the generated C++ test.

Note that `cpp-repro-gen.py` has no knowledge about the actual sizes of the input tensors if they are symbolic sizes, so you will get code like:

```C++
auto t0 = at::randn({-1}, options);
```

You can either manually modify the test, or use `--symbolic_sizes` to specify the symbolic sizes.

Example:

```
python cpp-repro-gen.py --symbolic_sizes 768 768 1024 768 < examples/repro.py > examples/repro.cpp
```
