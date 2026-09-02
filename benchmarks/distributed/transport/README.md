# Transport benchmark

Install the optional backend package matching the operation:

```bash
uv pip install "ibverbs[gpunetio-triton]"
uv pip install "ucxx-cu12==0.51.1"  # use ucxx-cu13 with CUDA 13
```

GPUNetIO needs `doca-sdk-gpunetio`, `doca-sdk-gpunetio-devel`, and ibverbs
bitcode ABI 2 or newer. The measurements below used rdma4py commit `22cf83d`;
until that change is released, install its `ibverbs/` package from a checkout.
Build and cache its device bitcode once:

```bash
python -c "from ibverbs.gpunetio import build_bitcode; print(build_bitcode(arch='sm_90'))"
```

Run one rank per host with `torchrun`. `--options` accepts one backend option
object per rank. `--interfaces` records physical byte counters and rejects
throughput below 80% of link rate by default.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  benchmarks/distributed/transport/benchmark.py \
  --backend ibverbs --interfaces "$INTERFACE" \
  --options="{\"hca\":\"$HCA\",\"num_qps\":4}"
```

Use `--device cpu --backend tcp` for frontend-NIC testing. Set each rank's
`host` in `--options` to the address of the interface under test.

Add `--cuda-graph` to capture one write and one read and benchmark graph
replay. The `ibverbs` backend also needs `"cuda_graph":true` in each rank's
options to select GPUNetIO.

## UCXX transports

Force TCP so same-host tests do not silently use shared memory:

```bash
UCX_TLS=tcp,cuda_copy UCX_NET_DEVICES=eth0 \
UCX_SOCKADDR_TLS_PRIORITY=tcp UCX_PROTO_INFO=y torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  benchmarks/distributed/transport/benchmark.py \
  --backend ucxx --interfaces "$INTERFACE" \
  --options="{\"host\":\"$LOCAL_IPV4\"}"
```

The `ucxx-cu12` and `ucxx-cu13` wheels bundle UCX without verbs. To use RC,
provide UCX 1.19 or newer built with CUDA, verbs, and RDMA-CM, then run:

```bash
RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true \
UCX_TLS=rc,cuda_copy UCX_NET_DEVICES=mlx5_0:1 \
UCX_SOCKADDR_TLS_PRIORITY=tcp UCX_PROTO_INFO=y torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  benchmarks/distributed/transport/benchmark.py \
  --backend ucxx --interfaces "$INTERFACE" \
  --options="{\"host\":\"$LOCAL_IPV4\"}"
```

Omit `cuda_copy` for CPU-only runs. `UCX_PROTO_INFO=y` reports the selected
data path.

## Benchmark report

Measured 2026-09-01 on one host with two H100 GPUs and two directly attached
400 Gb/s ConnectX-7 ports (`GPU0/mlx5_0/beth3` and
`GPU1/mlx5_3/beth4`). The environment used DOCA 3.4.0112, Triton 3.8.0,
ibverbs 0.1.0 at `22cf83d`, UCXX-cu12 0.51.1, and CUDA 12.8. Results are
medians for 64 MiB transfers.

| Backend | Mode | Write | Read | Result |
|---|---|---:|---:|---|
| rdma4py | host-posted, 4 QPs | 381.4 Gb/s | 381.0 Gb/s | 95% application rate |
| rdma4py GPUNetIO | eager, 4 QPs | 368.7 Gb/s | 370.2 Gb/s | 92% application rate |
| rdma4py GPUNetIO | CUDA graph, 4 QPs | 381.5 Gb/s | 382.7 Gb/s | 95% application rate |
| TCP | CPU, 16 flows, loopback | 224.2 Gb/s | 249.6 Gb/s | host-only |
| TCP | CUDA staging, 16 flows, loopback | 13.3 Gb/s | 13.2 Gb/s | host-only |
| UCXX TCP | CPU, forced TCP loopback | 45.4 Gb/s | 41.3 Gb/s | host-only |
| UCXX TCP | CUDA, forced TCP loopback | 3.6 Gb/s | 3.6 Gb/s | host-only |

The graph run's directional NIC counters measured 384.0/387.8 Gb/s. A
604-completion-per-QP graph test crossed the 256-entry CQ twice without a hang;
at 1 MiB it reached 174.9/179.0 Gb/s and wire throughput matched payload
throughput.

Physical TCP and UCXX-RC bandwidth were not measurable on this single host:
its data interfaces expose only IPv6 while UCXX 0.51 creates an IPv4 listener,
and local addresses route through the host instead of the NIC. System UCX with
the verbs plugins exposed `rc_verbs`; GPU RC additionally needs a combined
CUDA+verbs UCX build. The torchcomms RDMA extension was unavailable.

The exact same-host report commands were:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 67108864 --warmup 5 --iterations 20 --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4},{"hca":"mlx5_3","num_qps":4}]'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 67108864 --warmup 5 --iterations 20 --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 67108864 --warmup 5 --iterations 20 --cuda-graph --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 1048576 --warmup 2 --iterations 300 --cuda-graph --minimum-line-rate 0 --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'
.venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend tcp --device cpu --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='{"host":"127.0.0.1","num_flows":16}'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend tcp --device cuda --sizes 67108864 --warmup 1 --iterations 3 --minimum-line-rate 0 --options='{"host":"127.0.0.1","num_flows":16}'
UCX_TLS=tcp UCX_SOCKADDR_TLS_PRIORITY=tcp .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ucxx --device cpu --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='{"host":"127.0.0.1"}'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UCX_TLS=tcp,cuda_copy UCX_SOCKADDR_TLS_PRIORITY=tcp .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ucxx --device cuda --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='{"host":"127.0.0.1"}'
```
