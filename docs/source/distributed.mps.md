# MPS distributed backend (Apple Thunderbolt RDMA)

PyTorch's MPS distributed backend runs collectives over Apple Thunderbolt
RDMA using the system `librdma.dylib` (the JACCL path). It is meant for
multi-Mac training/inference on Apple Silicon, connected point-to-point with a
Thunderbolt cable.

If your setup is not Thunderbolt RDMA, use the `gloo` backend instead -
`ProcessGroupMPS` construction intentionally fails on setups that cannot
allocate an RDMA protection domain on every rank.

## Quickstart (two Macs, single TB5 cable, macOS 26.2+)

### 0. Hardware + OS requirements

- Two (or more) Apple Silicon Macs.
- macOS 26.2 or later on every Mac (this is when Apple shipped
  `librdma.dylib` + `rdma_en*` devices).
- A Thunderbolt 5 cable directly between them. No hub, no dock - point-to-point.
- Same LAN (e.g., same Wi-Fi) for the one-time SSH bootstrap below.

### 1. Enable the RDMA driver, once per Mac, in macOS Recovery

Apple gates this behind Recovery - it cannot be done from a running
system even with sudo.

```
1. Reboot holding the power button (Apple Silicon) to enter Recovery.
2. Utilities → Terminal.
3. rdma_ctl enable
4. Reboot normally.
```

### 2. Enable SSH + distribute your key, once per peer Mac

On every Mac *except* the one you'll run the cluster configurator from:

```
System Settings → General → Sharing → Remote Login : on
```

Then from the controller Mac, for each peer:

```
ssh-copy-id <peer-user>@<peer-host>.local
```

Accept the host-key prompt with `yes` on first connect. `<peer-host>` is
what `hostname` prints on the peer; `.local` is Bonjour/mDNS resolution
over the shared LAN.

### 3. Configure every node's TB networking, once per boot

Run from the controller Mac:

```
scripts/distributed_mps/configure_tb5_cluster.sh \
    <peer-user>@<peer-host>.local [more peers…]
```

The script destroys `bridge0` on every node (locally with sudo, on
peers via ssh+sudo), switches into a dedicated network location named
`tb5`, and creates a DHCP/APIPA network service per TB port (`TB5-1`,
`TB5-2`, …). At the end it prints every node's per-port IPs - that's
what you pass as `MASTER_ADDR`.

### 4. Launch the distributed job

Pick one node to be rank 0 and use **one of its TB-port IPs** (the
`169.254.x.y` ones the configurator printed) as `MASTER_ADDR` on every
node. Example:

```python
import os, torch, torch.distributed as dist

os.environ["MASTER_ADDR"] = "169.254.29.27"   # one of rank 0's TB IPs
os.environ["MASTER_PORT"] = "29501"
dist.init_process_group(backend="mps", rank=RANK, world_size=WORLD)

t = torch.ones(1024, device="mps")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
```

On successful init every rank logs `ProcessGroupMPS: using JACCL RDMA
transport on rdma_enX`. A construction error pointing at `gloo` means
something in steps 1-3 went wrong (`rdma_ctl enable` skipped, `tb5`
location not active, or `MASTER_ADDR` unreachable over TB).

## Supported operations

See [`torch/csrc/distributed/c10d/ProcessGroupMPS.md`](../../torch/csrc/distributed/c10d/ProcessGroupMPS.md)
for the current op support table and notes on what is implemented vs
pending.
