From the root of pytorch repo, run:
```
python -m benchmarks.tensorexpr --help
```
to show documentation.

An example of an eager CPU benchmark:
```
python -m benchmarks.tensorexpr --device cpu --mode fwd batch_matmul_fwd_cpu_1_4_4_4
```
