From the root of pytorch repo, run:
```
python -m benchmarks.tensorexpr --help
```
to show documentation.

An example of an actual command line:
```
python -m benchmarks.tensorexpr broadcast --device gpu --mode fwd --jit_mode trace
```
