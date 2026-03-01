# FX Partitioner Micro-benchmark

Quick timing harness for `CapabilityBasedPartitioner.propose_partitions` across graph sizes and unsupported-op ratios.

## Usage

```bash
python benchmarks/fx/partitioner_benchmark.py \
  --sizes 4000:40 10000:100 40000:400 100000:1000 \
  --iters 20 \
  --json /tmp/partitioner_bench.json
```

- `--sizes`: list of `NUM_NODES:NUM_UNSUPPORTED` cases (default: `4000:40 10000:100 40000:400 100000:1000`).
