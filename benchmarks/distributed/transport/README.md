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

The PyPI wheels support TCP. On two standard hosts, force TCP so UCX does not
select shared memory:

```bash
UCX_TLS=tcp,cuda_copy UCX_NET_DEVICES="$INTERFACE" \
torchrun --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  benchmarks/distributed/transport/benchmark.py \
  --backend ucxx --device "$DEVICE" --interfaces "$INTERFACE" \
  --minimum-line-rate 0 \
  --options="{\"host\":\"$LOCAL_IP\"}"
```

The `ucxx-cu12` and `ucxx-cu13` wheels bundle UCX without verbs. To use RC,
build UCX 1.19 or newer with CUDA, verbs, and RDMA-CM, then rebuild UCXX against
it. These are the relevant commands used for this report:

```bash
UCX_PREFIX="$PWD/ucx-install"
UCXX_PREFIX="$PWD/ucxx-install"
UCXX_VENV="$PWD/ucxx-venv"
git clone --branch v1.19.1 https://github.com/openucx/ucx.git
git clone https://github.com/rapidsai/ucxx.git
git -C ucxx checkout f38aa25666abe4fd758929d04684b5cc064f3b60
git -C ucxx apply --unidiff-zero - <<'PATCH'
diff --git a/cpp/src/listener.cpp b/cpp/src/listener.cpp
--- a/cpp/src/listener.cpp
+++ b/cpp/src/listener.cpp
@@ -4,0 +5 @@
+#include <cstdlib>
@@ -31 +32 @@ Listener::Listener(std::shared_ptr<Worker> worker,
-  auto info               = ucxx::utils::get_addrinfo(NULL, port);
+  auto info = ucxx::utils::get_addrinfo(std::getenv("UCXX_LISTENER_ADDRESS"), port);
diff --git a/cpp/src/endpoint.cpp b/cpp/src/endpoint.cpp
--- a/cpp/src/endpoint.cpp
+++ b/cpp/src/endpoint.cpp
@@ -213,3 +213,2 @@ std::shared_ptr<Endpoint> createEndpointFromConnRequest(
-    .field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_CONN_REQUEST |
-                  UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER,
-    .flags        = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK,
+    .field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
+                  UCP_EP_PARAM_FIELD_ERR_HANDLER,
PATCH
mkdir ucx-build
cd ucx-build
../ucx/contrib/configure-release --prefix="$UCX_PREFIX" --enable-mt \
  --with-cuda=/usr/local/cuda-12.8 --with-verbs=/usr --with-rdmacm=/usr \
  --with-mlx5 --without-gdrcopy
make -j && make install
cd ..
cmake -S ucxx/cpp -B ucxx-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$UCXX_PREFIX" -DCMAKE_PREFIX_PATH="$UCX_PREFIX" \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DUCXX_ENABLE_CCCL=ON
cmake --build ucxx-build -j && cmake --install ucxx-build
uv venv --python 3.13 --managed-python "$UCXX_VENV"
uv pip install --python "$UCXX_VENV/bin/python" \
  rapids-build-backend scikit-build-core cython cmake ninja cuda-bindings numpy
CMAKE_PREFIX_PATH="$UCXX_PREFIX:$UCX_PREFIX" \
SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$UCXX_PREFIX;$UCX_PREFIX;-DFIND_UCXX_CPP=ON" \
  uv pip install --python "$UCXX_VENV/bin/python" --no-build-isolation \
  --no-deps --config-settings rapidsai.disable-cuda=true \
  --config-settings skbuild.install.components=Unspecified \
  --config-settings skbuild.install.components=ucxx ucxx/python/ucxx
```

The listener patch is required for IPv6-only interfaces. The endpoint patch is
required only for this report's same-host topology. Keep this source UCXX
package separate from the wheel and add its site-packages directory to
`PYTHONPATH` at runtime.

Run RC with multiple paths and rendezvous lanes:

```bash
RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true \
RAPIDS_LIBUCXX_PREFER_SYSTEM_LIBRARY=true \
PYTHONPATH="$UCXX_VENV/lib/python3.13/site-packages:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$UCXX_PREFIX/lib64:$UCX_PREFIX/lib:$UCX_PREFIX/lib/ucx:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
UCX_MODULE_DIR="$UCX_PREFIX/lib/ucx" \
UCX_TLS=rc_verbs,ud_verbs,cuda_copy UCX_NET_DEVICES="$HCA:1" \
UCX_IB_GID_INDEX="$GID_INDEX" UCX_RDMA_CM_SOURCE_ADDRESS="$LOCAL_IPV6" \
UCX_SOCKADDR_TLS_PRIORITY=rdmacm UCX_IB_NUM_PATHS=4 \
UCX_IB_ROCE_REACHABILITY_MODE=all UCX_RC_VERBS_IS_GLOBAL=y \
UCX_UD_VERBS_IS_GLOBAL=y \
UCX_MAX_RNDV_LANES=4 UCX_MAX_RNDV_RAILS=4 UCX_PROTO_INFO=y \
UCXX_LISTENER_ADDRESS="$LOCAL_IPV6" torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  benchmarks/distributed/transport/benchmark.py \
  --backend ucxx --device "$DEVICE" --interfaces "$INTERFACE" \
  --minimum-line-rate 0 --rdma-counters \
  --options="{\"host\":\"$LOCAL_IPV6\"}"
```

Omit `cuda_copy` for CPU-only runs. If `nvidia_peermem` is unavailable, add
`UCX_ZCOPY_THRESH=inf`, `UCX_RNDV_FRAG_MEM_TYPES=host`, and
`UCX_RNDV_SCHEME=get_ppln`; CUDA then stages through host memory.
`UCX_PROTO_INFO=y` reports the selected lanes.

## Benchmark report

Measured 2026-09-01 on one host with two H100 GPUs and two directly attached
400 Gb/s ConnectX-7 ports (`GPU0/mlx5_0/beth3` and
`GPU1/mlx5_3/beth4`). The environment used CUDA 12.8, DOCA 3.4.0112, Triton
3.8.0, ibverbs 0.1.0 at `22cf83d`, torchcomms 0.3.0 at `6288fc4d`, UCX
1.19.1, and a source UCXX 0.52.0 build. Results are median application rates
for 64 MiB transfers through the physical NICs.

| Backend | Tensor/mode | Write | Read | Result |
|---|---|---:|---:|---|
| rdma4py | CPU, 4 QPs | 384.0 Gb/s | 385.2 Gb/s | 96% |
| rdma4py | CUDA host-posted, 4 QPs | 381.4 Gb/s | 381.0 Gb/s | 95% |
| rdma4py GPUNetIO | CUDA eager, 4 QPs | 368.7 Gb/s | 370.2 Gb/s | 92% |
| rdma4py GPUNetIO | CUDA graph, 4 QPs | 381.5 Gb/s | 382.7 Gb/s | 95% |
| torchcomms RDMA | CPU | 384.4 Gb/s | 385.4 Gb/s | 96% |
| torchcomms RDMA | CUDA eager | 380.2 Gb/s | 381.8 Gb/s | 95% |
| TCP | CPU, 16 flows, IPv6 | 150.2 Gb/s | 143.4 Gb/s | 38%/36% |
| TCP | CUDA staging, 16 flows, IPv6 | 44.6 Gb/s | 40.8 Gb/s | 11%/10% |
| UCXX TCP | CPU, IPv6 | 26.9 Gb/s | 20.8 Gb/s | 7%/5% |
| UCXX TCP | CUDA staging, IPv6 | 3.28 Gb/s | 3.79 Gb/s | <1% |
| UCXX RC | CPU, 4 paths | 272.6 Gb/s | 271.5 Gb/s | 68% |
| UCXX RC | CUDA host staging, 4 paths | 4.75 Gb/s | 5.61 Gb/s | 1% |

The RDMA counters confirmed 388.0-392.7 Gb/s for rdma4py CPU,
384.0-387.8 Gb/s for GPUNetIO graph, and 381.0-386.1 Gb/s for torchcomms.
The graph run crossed the 256-entry CQ twice with 604 completions per QP. At
1 MiB it reached 174.9/179.0 Gb/s and wire rate matched payload rate.

TCP and UCXX TCP used IPvlan interfaces over `beth3` and `beth4` in separate
network namespaces on distinct routed IPv6 subnets. This prevents local routing
from bypassing the NICs. TCP tuning covered 8, 16, 32, and 64 flows with 1, 4,
and 8 MiB chunks; 16 flows gave the best balanced rate. The 4 and 8 MiB chunk
settings both produce one segment per flow for a 64 MiB transfer. UCXX RC
selected four repeated RC lanes. Direct `ucx_perftest` reported 25% per path
and reached 256.9 Gb/s with one outstanding operation and 370.7 Gb/s with four.
CUDA RC staged through host because `nvidia_peermem` was not loaded and direct
CUDA memory registration failed.

Latency sweeps used 20 warmups and 1,000 iterations. Each cell is median
write/read latency in microseconds.

| Backend/mode | 8 B | 64 B | 256 B | 1 KiB | 4 KiB | 16 KiB | 64 KiB | 256 KiB | 1 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rdma4py CPU | 8.6/9.1 | 8.6/9.0 | 8.6/9.1 | 8.9/9.4 | 9.1/9.8 | 9.9/10.5 | 10.8/12.1 | 16.1/18.2 | 33.3/36.4 |
| rdma4py CUDA | 21.1/25.3 | 21.3/25.7 | 21.2/25.6 | 21.4/26.0 | 21.5/26.1 | 22.0/26.6 | 23.0/27.8 | 27.9/33.6 | 44.5/50.4 |
| GPUNetIO eager | 62.7/62.0 | 62.8/62.5 | 62.2/62.1 | 62.5/62.3 | 62.3/62.1 | 61.6/62.0 | 65.6/65.3 | 68.2/68.5 | 87.4/86.9 |
| GPUNetIO graph | 22.5/22.6 | 22.6/22.4 | 22.5/22.5 | 22.5/22.4 | 22.4/22.4 | 22.4/22.4 | 25.7/25.9 | 29.1/29.4 | 47.6/46.7 |
| torchcomms CPU | 19.0/18.1 | 19.1/10.0 | 9.9/10.1 | 18.7/10.7 | 19.3/18.8 | 19.1/18.9 | 19.0/19.4 | 25.9/26.0 | 42.0/45.9 |
| torchcomms CUDA | 24.5/23.3 | 23.9/23.0 | 23.8/23.5 | 22.2/23.5 | 23.4/23.5 | 23.4/23.5 | 24.0/23.7 | 27.5/29.4 | 44.4/47.8 |
| TCP IPv6 CPU | 456.0/445.3 | 865.8/786.2 | 833.9/783.4 | 875.6/793.1 | 843.8/767.5 | 815.4/813.4 | 851.4/811.2 | 864.9/818.6 | 1008.9/967.8 |
| TCP IPv6 CUDA | 929.8/857.5 | 1853.6/1820.2 | 1833.6/1777.6 | 1861.8/1743.7 | 1820.4/1762.4 | 1843.5/1863.8 | 1922.2/1755.4 | 1909.4/1896.3 | 2097.2/2003.9 |
| UCXX TCP IPv6 CPU | 228.6/210.1 | 228.2/211.5 | 227.9/203.3 | 228.8/210.9 | 234.6/211.9 | 314.7/243.9 | 388.7/334.8 | 415.0/393.0 | 710.1/648.7 |
| UCXX TCP IPv6 CUDA | 264.0/236.3 | 273.4/228.5 | 276.4/222.8 | 280.3/223.5 | 288.9/239.0 | 405.4/385.5 | 562.0/543.6 | 979.8/975.7 | 2765.0/2808.7 |
| UCXX RC CPU | 199.2/156.2 | 204.2/155.6 | 201.9/155.4 | 196.8/155.2 | 202.9/155.2 | 225.8/174.0 | 204.8/170.6 | 226.6/173.0 | 237.6/189.2 |
| UCXX RC CUDA staged | 195.2/148.8 | 201.1/154.8 | 202.9/156.8 | 211.5/161.1 | 212.2/163.0 | 220.9/172.8 | 295.1/242.8 | 565.7/526.3 | 1915.1/1616.2 |

Torchcomms records CUDA device 0 for CPU memory. Its CPU row exposes one
topology-local physical GPU per rank with `CUDA_VISIBLE_DEVICES`, making each
one logical device 0, and selects the adjacent HCA with `NCCL_IB_HCA`. The
source UCXX validation build included the IPv6 listener patch above and removed
`UCP_EP_PARAM_FIELD_FLAGS` and `UCP_EP_PARAMS_FLAGS_NO_LOOPBACK` from
`cpp/src/endpoint.cpp` for same-host endpoints. Only the endpoint change is
unnecessary on two hosts.

The size sweep replaced `--sizes` below with
`8,64,256,1024,4096,16384,65536,262144,1048576`, used `--warmup 20`, and
used `--iterations 1000`. Representative 64 MiB commands were:

```bash
.venv/bin/torchrun --standalone --nproc-per-node=2 \
  benchmarks/distributed/transport/benchmark.py --backend ibverbs \
  --device cuda --sizes 67108864 --warmup 5 --iterations 20 \
  --interfaces beth3,beth4 \
  --options='[{"hca":"mlx5_0","num_qps":4},{"hca":"mlx5_3","num_qps":4}]'

.venv/bin/torchrun --standalone --nproc-per-node=2 \
  benchmarks/distributed/transport/benchmark.py --backend ibverbs \
  --device cuda --sizes 67108864 --warmup 5 --iterations 20 --cuda-graph \
  --interfaces beth3,beth4 \
  --options='[{"hca":"mlx5_0","num_qps":4,"cuda_graph":true},{"hca":"mlx5_3","num_qps":4,"cuda_graph":true}]'

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" NCCL_IB_HCA="=$HCA:1" \
NCCL_IB_GID_INDEX="$GID_INDEX" NCCL_CTRAN_IB_DEVICES_PER_RANK=1 \
RANK="$RANK" WORLD_SIZE=2 LOCAL_RANK=0 \
PYTHONPATH="$TORCHCOMMS_SOURCE/comms" \
LD_LIBRARY_PATH="$TORCHCOMMS_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
  .venv/bin/python benchmarks/distributed/transport/benchmark.py \
  --backend torchcomms --device cuda --tensor-device cpu \
  --init-method="file://$SHARED_STORE" --interfaces beth3,beth4 \
  --rdma-counters --sizes 67108864 --warmup 5 --iterations 20 \
  --minimum-line-rate 0

RANK="$RANK" WORLD_SIZE=2 LOCAL_RANK="$RANK" \
GLOO_SOCKET_IFNAME="$IPVLAN_INTERFACE" .venv/bin/python \
  benchmarks/distributed/transport/benchmark.py --backend tcp \
  --device "$DEVICE" \
  --init-method="file://$SHARED_STORE" --interfaces trp0,trp1 \
  --sizes 67108864 --warmup 10 --iterations 30 --minimum-line-rate 0 \
  --options='[{"host":"2401:db00:145a:4888:bace:0:3a2:1","num_flows":16},{"host":"2401:db00:145a:488b:bace:0:3c7:1","num_flows":16}]'

RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true \
RAPIDS_LIBUCXX_PREFER_SYSTEM_LIBRARY=true \
PYTHONPATH="$UCXX_VENV/lib/python3.13/site-packages:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$UCXX_PREFIX/lib64:$UCX_PREFIX/lib:$UCX_PREFIX/lib/ucx:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
UCX_MODULE_DIR="$UCX_PREFIX/lib/ucx" UCX_TLS=tcp,cuda_copy \
UCX_NET_DEVICES="$IPVLAN_INTERFACE" UCXX_LISTENER_ADDRESS="$LOCAL_IPV6" \
RANK="$RANK" WORLD_SIZE=2 LOCAL_RANK="$RANK" \
GLOO_SOCKET_IFNAME="$IPVLAN_INTERFACE" .venv/bin/python \
  benchmarks/distributed/transport/benchmark.py --backend ucxx \
  --device "$DEVICE" --init-method="file://$SHARED_STORE" \
  --interfaces trp0,trp1 --sizes 67108864 --warmup 10 --iterations 30 \
  --minimum-line-rate 0 \
  --options='[{"host":"2401:db00:145a:4888:bace:0:3a2:1"},{"host":"2401:db00:145a:488b:bace:0:3c7:1"}]'

RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true \
RAPIDS_LIBUCXX_PREFER_SYSTEM_LIBRARY=true \
PYTHONPATH="$UCXX_VENV/lib/python3.13/site-packages:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$UCXX_PREFIX/lib64:$UCX_PREFIX/lib:$UCX_PREFIX/lib/ucx:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
UCX_MODULE_DIR="$UCX_PREFIX/lib/ucx" \
UCX_TLS=rc_verbs,ud_verbs,cuda_copy UCX_SOCKADDR_TLS_PRIORITY=rdmacm \
UCX_IB_ROCE_REACHABILITY_MODE=all UCX_RC_VERBS_IS_GLOBAL=y \
UCX_UD_VERBS_IS_GLOBAL=y UCX_IB_NUM_PATHS=4 \
UCX_MAX_RNDV_LANES=4 UCX_MAX_RNDV_RAILS=4 \
UCX_NET_DEVICES="$HCA:1" UCX_IB_GID_INDEX="$GID_INDEX" \
UCX_RDMA_CM_SOURCE_ADDRESS="$LOCAL_IPV6" \
UCXX_LISTENER_ADDRESS="$LOCAL_IPV6" RANK="$RANK" WORLD_SIZE=2 \
LOCAL_RANK="$RANK" .venv/bin/python \
  benchmarks/distributed/transport/benchmark.py --backend ucxx --device cpu \
  --init-method="file://$SHARED_STORE" --interfaces beth3,beth4 \
  --sizes 67108864 --warmup 10 --iterations 100 --minimum-line-rate 0 \
  --one-way-connect --rdma-counters \
  --options='[{"host":"2401:db00:145a:4888:bace:0:3a2:0"},{"host":"2401:db00:145a:488b:bace:0:3c7:0"}]'
```

The CUDA RC row used `--device cuda --iterations 50` plus
`UCX_ZCOPY_THRESH=inf`, `UCX_RNDV_FRAG_MEM_TYPES=host`, and
`UCX_RNDV_SCHEME=get_ppln`.
