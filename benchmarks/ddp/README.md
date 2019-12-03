# Distributed Data Parallel Benchmark

This tool is used to measure distributed training iteration time. This
is helpful for evaluating the performance impact of code changes to
`torch.nn.parallel.DistributedDataParallel`, `torch.distributed`, or
anything in between.

It optionally produces a JSON file with all measurements, allowing for
an easy A/B comparison of code, configuration, or environment. This
comparison can be produced by `diff.py`.

## Requirements

This benchmark depends on PyTorch and torchvision.

## How to run

Run as many copies of this script as you have model replicas.

If you launch a single task per machine with multiple GPUs, consider
using [`torch.distributed.launch`][launch] to spawn multiple processes
per machine.

[launch]: https://pytorch.org/docs/stable/distributed.html#launch-utility

Example output (only on rank 0):

```
-----------------------------------
PyTorch distributed benchmark suite
-----------------------------------

* PyTorch version: 1.4.0a0+fcb7371
* CUDA version: 10.0

--- nvidia-smi topo -m ---

        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_2  mlx5_0  mlx5_3  mlx5_1  CPU Affinity
GPU0     X      NV1     NV1     NV2     NV2     SYS     SYS     SYS     SYS     PIX     SYS     PHB     0-19,40-59
GPU1    NV1      X      NV2     NV1     SYS     NV2     SYS     SYS     SYS     PIX     SYS     PHB     0-19,40-59
GPU2    NV1     NV2      X      NV2     SYS     SYS     NV1     SYS     SYS     PHB     SYS     PIX     0-19,40-59
GPU3    NV2     NV1     NV2      X      SYS     SYS     SYS     NV1     SYS     PHB     SYS     PIX     0-19,40-59
GPU4    NV2     SYS     SYS     SYS      X      NV1     NV1     NV2     PIX     SYS     PHB     SYS     0-19,40-59
GPU5    SYS     NV2     SYS     SYS     NV1      X      NV2     NV1     PIX     SYS     PHB     SYS     0-19,40-59
GPU6    SYS     SYS     NV1     SYS     NV1     NV2      X      NV2     PHB     SYS     PIX     SYS     0-19,40-59
GPU7    SYS     SYS     SYS     NV1     NV2     NV1     NV2      X      PHB     SYS     PIX     SYS     0-19,40-59
mlx5_2  SYS     SYS     SYS     SYS     PIX     PIX     PHB     PHB      X      SYS     PHB     SYS
mlx5_0  PIX     PIX     PHB     PHB     SYS     SYS     SYS     SYS     SYS      X      SYS     PHB
mlx5_3  SYS     SYS     SYS     SYS     PHB     PHB     PIX     PIX     PHB     SYS      X      SYS
mlx5_1  PHB     PHB     PIX     PIX     SYS     SYS     SYS     SYS     SYS     PHB     SYS      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe switches (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing a single PCIe switch
  NV#  = Connection traversing a bonded set of # NVLinks

--------------------------


Benchmark: resnet50 with batch size 32
   1 GPUs --   warmup: p50:  0.098   327/s  p75:  0.098   326/s  p90:  0.098   326/s  p95:  0.098   326/s
   1 GPUs --    1M/1G: p50:  0.097   329/s  p75:  0.097   328/s  p90:  0.097   328/s  p95:  0.097   328/s
   2 GPUs --    1M/2G: p50:  0.104   307/s  p75:  0.104   307/s  p90:  0.104   307/s  p95:  0.104   306/s
   4 GPUs --    1M/4G: p50:  0.105   305/s  p75:  0.105   305/s  p90:  0.105   305/s  p95:  0.105   305/s
   8 GPUs --    1M/8G: p50:  0.108   296/s  p75:  0.108   296/s  p90:  0.108   296/s  p95:  0.108   295/s
  16 GPUs --    2M/8G: p50:  0.109   294/s  p75:  0.111   288/s  p90:  0.112   284/s  p95:  0.113   282/s
  24 GPUs --    3M/8G: p50:  0.111   289/s  p75:  0.111   287/s  p90:  0.112   286/s  p95:  0.112   286/s
  32 GPUs --    4M/8G: p50:  0.110   289/s  p75:  0.112   286/s  p90:  0.114   281/s  p95:  0.114   280/s
  40 GPUs --    5M/8G: p50:  0.113   282/s  p75:  0.114   281/s  p90:  0.114   279/s  p95:  0.115   278/s
  48 GPUs --    6M/8G: p50:  0.112   285/s  p75:  0.114   281/s  p90:  0.116   276/s  p95:  0.118   271/s
  56 GPUs --    7M/8G: p50:  0.115   279/s  p75:  0.117   273/s  p90:  0.117   273/s  p95:  0.118   271/s
  64 GPUs --    8M/8G: p50:  0.113   284/s  p75:  0.113   282/s  p90:  0.114   281/s  p95:  0.114   281/s
```

## How to diff

Run the benchmark with the `--json PATH_TO_REPORT_FILE` argument to
produce the JSON file that the diff script can consume.

Then, run the diff script as follows:

```
$ python3 diff.py PATH_TO_BASELINE_FILE PATH_TO_TEST_FILE
                                 baseline                      test
                     --------------------      --------------------
cuda_version:                        10.0  vs                  10.0
pytorch_version:          1.4.0a0+fcb7371  vs       1.4.0a0+fcb7371

Benchmark: resnet50 with batch size 32
   1 GPUs: p75:  0.097 ( -0.0%)  p95:  0.097 ( +0.1%)
   2 GPUs: p75:  0.104 ( -0.8%)  p95:  0.104 ( -0.8%)
   4 GPUs: p75:  0.105 ( -1.3%)  p95:  0.105 ( -1.4%)
   8 GPUs: p75:  0.108 ( -3.5%)  p95:  0.108 ( -3.5%)
  16 GPUs: p75:  0.111 ( -5.4%)  p95:  0.113 ( -7.6%)
  24 GPUs: p75:  0.111 ( -5.4%)  p95:  0.112 ( -5.9%)
  32 GPUs: p75:  0.112 ( -5.5%)  p95:  0.114 ( -7.8%)
  40 GPUs: p75:  0.114 ( -7.3%)  p95:  0.115 ( -7.9%)
  48 GPUs: p75:  0.114 ( -7.1%)  p95:  0.118 (-10.0%)
  56 GPUs: p75:  0.117 ( -9.4%)  p95:  0.118 ( -9.9%)
  64 GPUs: p75:  0.113 ( -6.4%)  p95:  0.114 ( -5.2%)
```

This compares throughput between `bucket_cap_mb=25` (the default) and
`bucket_cap_mb=1` on 8 DGX machines with V100 GPUs. It confims that
even for a relatively small model on machines with a very fast
interconnect (4x 100Gb InfiniBand per machine), it still pays off to
batch allreduce calls.
