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
`host` in `--options` to the address of the interface under test. Use
`--init-method=file:///shared/path` when the ranks cannot share a TCPStore
address, such as isolated network namespaces.

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
RAPIDS_LIBUCXX_PREFER_SYSTEM_LIBRARY=true \
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
medians for 64 MiB transfers. Torchcomms 0.3.0 at `6288fc4d` was built from
source against this PyTorch checkout.

| Backend | Mode | Write | Read | Result |
|---|---|---:|---:|---|
| rdma4py | host-posted, 4 QPs | 381.4 Gb/s | 381.0 Gb/s | 95% application rate |
| rdma4py GPUNetIO | eager, 4 QPs | 368.7 Gb/s | 370.2 Gb/s | 92% application rate |
| rdma4py GPUNetIO | CUDA graph, 4 QPs | 381.5 Gb/s | 382.7 Gb/s | 95% application rate |
| torchcomms RDMA | eager | 380.2 Gb/s | 381.8 Gb/s | 95% application rate |
| TCP | CPU, 16 flows, physical IPv6 | 119.6 Gb/s | 157.0 Gb/s | 30%/39% application rate |
| TCP | CPU, 16 flows, loopback | 224.2 Gb/s | 249.6 Gb/s | host-only |
| TCP | CUDA staging, 16 flows, loopback | 13.3 Gb/s | 13.2 Gb/s | host-only |
| UCXX TCP | CPU, forced TCP loopback | 45.4 Gb/s | 41.3 Gb/s | host-only |
| UCXX TCP | CUDA, forced TCP loopback | 3.6 Gb/s | 3.6 Gb/s | host-only |

The graph run's directional NIC counters measured 384.0/387.8 Gb/s. A
604-completion-per-QP graph test crossed the 256-entry CQ twice without a hang;
at 1 MiB it reached 174.9/179.0 Gb/s and wire throughput matched payload
throughput.

The physical TCP run put IPvlan interfaces over `beth3` and `beth4` in separate
network namespaces on distinct routed IPv6 subnets. This prevented local
routing from bypassing the NICs. Directional interface counters measured
112.1/113.2 Gb/s for writes and 153.4/153.4 Gb/s for reads.

UCXX-RC bandwidth was not measurable on this single host: UCXX 0.51 creates an
IPv4 listener, while these data interfaces expose only IPv6. Direct local UCX
worker-address connections between the two HCAs also timed out. System UCX
exposed `rc_verbs`; GPU RC additionally needs a combined CUDA+verbs UCX build.
The published torchcomms nightly did not support CUDA IPC on this RHEL 9 host,
so its transport was built from source. Its directional NIC counters measured
383.2/383.4 Gb/s for writes and 386.1/381.0 Gb/s for reads.

The exact same-host report commands were:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 67108864 --warmup 5 --iterations 20 --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4},{"hca":"mlx5_3","num_qps":4}]'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 67108864 --warmup 5 --iterations 20 --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 67108864 --warmup 5 --iterations 20 --cuda-graph --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ibverbs --device cuda --sizes 1048576 --warmup 2 --iterations 300 --cuda-graph --minimum-line-rate 0 --interfaces beth3,beth4 --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'
PYTHONPATH="$TORCHCOMMS_SOURCE/comms" LD_LIBRARY_PATH="$TORCHCOMMS_PREFIX/lib:$LD_LIBRARY_PATH" .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend torchcomms --device cuda --sizes 67108864 --warmup 5 --iterations 20 --interfaces beth3,beth4
RANK="$RANK" WORLD_SIZE=2 LOCAL_RANK="$RANK" GLOO_SOCKET_IFNAME="$IPVLAN_INTERFACE" .venv/bin/python benchmarks/distributed/transport/benchmark.py --backend tcp --device cpu --init-method="file://$SHARED_STORE" --interfaces trp0,trp1 --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='[{"host":"2401:db00:145a:4888:bace:0:3a2:1","num_flows":16,"chunk_size":4194304},{"host":"2401:db00:145a:488b:bace:0:3c7:1","num_flows":16,"chunk_size":4194304}]'
.venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend tcp --device cpu --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='{"host":"127.0.0.1","num_flows":16}'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend tcp --device cuda --sizes 67108864 --warmup 1 --iterations 3 --minimum-line-rate 0 --options='{"host":"127.0.0.1","num_flows":16}'
UCX_TLS=tcp UCX_SOCKADDR_TLS_PRIORITY=tcp .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ucxx --device cpu --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='{"host":"127.0.0.1"}'
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UCX_TLS=tcp,cuda_copy UCX_SOCKADDR_TLS_PRIORITY=tcp .venv/bin/torchrun --standalone --nproc-per-node=2 benchmarks/distributed/transport/benchmark.py --backend ucxx --device cuda --sizes 67108864 --warmup 2 --iterations 10 --minimum-line-rate 0 --options='{"host":"127.0.0.1"}'
```
