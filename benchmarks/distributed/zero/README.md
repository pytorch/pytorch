# ZeRO Partition Algorithm Experiment
This code examines the optimizer state partitioning in ZeRO --- in particular,
the number of parameters per shard and the latency of the `optimizer.step()`.
It is meant for comparing alternative partioning algorithms (i.e.
implementations of `ZeroRedundancyOptimizer.partition_parameters()`).

## Requirements
This requires `torch`, `torchvision`, and `transformers`.

## Instructions
The experiment script `zero.py` should be run on a machine with sufficient GPUs
to accommodate the desired world size (e.g. 4 GPUs for a world size of 4) since
the script spawns one process per GPU. The distributed backend can be specified
with the `-b` flag. In addition, the output directory and a list of world sizes
should be specified.

As a running example, suppose we want to specify a NCCL backend, an output
directory of "greedy/", and world sizes of 2 and 4. We may run:
```
python zero.py -b nccl greedy 2 4
```
The corresponding output looks like:
```
-----------------------------------
PyTorch ZeRO Experiment
-----------------------------------

* PyTorch version: 1.9.0a0+git920619d
* CUDA version: 11.1
* Distributed backend: nccl

world_size=2 model=ResNet50
rank=0: optimizer.step() took 0.039 s
rank=0: 13249600 parameters
rank=1: optimizer.step() took 0.048 s
rank=1: 12307432 parameters

world_size=2 model=ResNet152
rank=1: optimizer.step() took 0.052 s
rank=1: 29625320 parameters
rank=0: optimizer.step() took 0.055 s
rank=0: 30567488 parameters

world_size=2 model=BERT
rank=0: optimizer.step() took 0.016 s
rank=0: 54749184 parameters
rank=1: optimizer.step() took 0.019 s
rank=1: 54733056 parameters

world_size=4 model=ResNet50
rank=2: optimizer.step() took 0.012 s
rank=2: 6847360 parameters
rank=1: optimizer.step() took 0.011 s
rank=1: 7524864 parameters
rank=3: optimizer.step() took 0.014 s
rank=3: 5519080 parameters
rank=0: optimizer.step() took 0.018 s
rank=0: 5665728 parameters

world_size=4 model=ResNet152
rank=2: optimizer.step() took 0.041 s
rank=2: 14222696 parameters
rank=0: optimizer.step() took 0.041 s
rank=0: 14230720 parameters
rank=3: optimizer.step() took 0.042 s
rank=3: 15507200 parameters
rank=1: optimizer.step() took 0.043 s
rank=1: 16232192 parameters

world_size=4 model=BERT
rank=0: optimizer.step() took 0.010 s
rank=0: 26985216 parameters
rank=1: optimizer.step() took 0.012 s
rank=1: 28151040 parameters
rank=2: optimizer.step() took 0.017 s
rank=2: 27766272 parameters
rank=3: optimizer.step() took 0.017 s
rank=3: 26579712 parameters

```
Suppose we modify the `partition_parameters()` implementation to first sort the
parameters by size in descending order and then run `python zero.py -b nccl
greedy_sorted 2 4`. Now, to analyze the results, we use the script `analyze.py`
and specify the output directory names. In our case, we can run:
```
python analyze.py greedy greedy_sorted
```
The corresponding output looks like:
```
-----------------------------------
greedy
-----------------------------------
world_size=2 model_name=ResNet50
max params     = 13249600
mean params    = 12778516.000
diff           = 471084.000
max time (std) = 0.048 (0.00033)

world_size=2 model_name=ResNet152
max params     = 30567488
mean params    = 30096404.000
diff           = 471084.000
max time (std) = 0.055 (0.00044)

world_size=2 model_name=BERT
max params     = 54749184
mean params    = 54741120.000
diff           = 8064.000
max time (std) = 0.019 (0.00008)

world_size=4 model_name=ResNet50
max params     = 7524864
mean params    = 6389258.000
diff           = 1135606.000
max time (std) = 0.018 (0.00222)

world_size=4 model_name=ResNet152
max params     = 16232192
mean params    = 15048202.000
diff           = 1183990.000
max time (std) = 0.043 (0.00027)

world_size=4 model_name=BERT
max params     = 28151040
mean params    = 27370560.000
diff           = 780480.000
max time (std) = 0.017 (0.00020)

-----------------------------------
greedy_sorted
-----------------------------------
world_size=2 model_name=ResNet50
max params     = 12794816
mean params    = 12778516.000
diff           = 16300.000
max time (std) = 0.044 (0.00028)

world_size=2 model_name=ResNet152
max params     = 30111424
mean params    = 30096404.000
diff           = 15020.000
max time (std) = 0.054 (0.00037)

world_size=2 model_name=BERT
max params     = 55327488
mean params    = 54741120.000
diff           = 586368.000
max time (std) = 0.022 (0.00010)

world_size=4 model_name=ResNet50
max params     = 6436864
mean params    = 6389258.000
diff           = 47606.000
max time (std) = 0.015 (0.00045)

world_size=4 model_name=ResNet152
max params     = 15090152
mean params    = 15048202.000
diff           = 41950.000
max time (std) = 0.043 (0.00020)

world_size=4 model_name=BERT
max params     = 28352256
mean params    = 27370560.000
diff           = 981696.000
max time (std) = 0.015 (0.00031)
```
