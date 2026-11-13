# Distributed Triage Rubric

Detailed guidance for sub-triaging issues in the `oncall: distributed` queue. Use this rubric alongside [distributed-labels.json](distributed-labels.json) when classifying issues.

---

## 1. Sub-Oncall Routing Rules

Route each issue to exactly ONE sub-oncall. When in doubt, default to `oncall: distributed parallelisms`.

### `oncall: distributed parallelisms` (default)

Covers the parallel training APIs that users interact with directly:
- **FSDP** (FSDP1 and FSDP2): `FullyShardedDataParallel`, `fully_shard`, sharding strategies, mixed precision
- **DDP**: `DistributedDataParallel`, gradient synchronization, parameter broadcasting
- **DTensor**: `distribute_tensor`, placements (`Shard`, `Replicate`), tensor redistribution
- **Tensor Parallel**: column/row parallel, sequence parallel
- **Context Parallel**: context parallelism for long sequences
- **Pipeline Parallel**: `PipelineSchedule`, `PipelineStage`, pipeline schedules
- **Activation Checkpointing** (when used with distributed training)

### `oncall: distributed infra`

Covers the communication infrastructure layer:
- **c10d / Process Groups**: `init_process_group`, `ProcessGroup`, `new_group`, stores (`TCPStore`, `FileStore`)
- **Collectives**: `all_reduce`, `all_gather`, `broadcast`, `reduce_scatter`, `all_to_all`, `barrier`
- **Backends**: NCCL, Gloo, MPI, UCC — backend-specific errors, configuration, initialization
- **Elastic / torchrun**: `torch.distributed.elastic`, `torchrun`, rendezvous, agent, worker management
- **RPC**: `torch.distributed.rpc`, RRef, distributed autograd
- **Distributed tools**: debugging utilities, flight recorder, distributed logging
- **DeviceMesh**: `init_device_mesh`, mesh dimensions, multi-dimensional parallelism
- **Symmetric Memory**: `SymmetricMemory`, `symm_mem`

### `oncall: distributed checkpointing`

Covers saving/loading distributed model state:
- **DCP** (Distributed Checkpoint): `torch.distributed.checkpoint`, `save`, `load`, `state_dict`
- **State dict utilities**: `get_model_state_dict`, `get_optimizer_state_dict`, `set_model_state_dict`
- **Checkpoint format**: file system planner, HDF5, resharding across different world sizes
- **Async checkpointing**: non-blocking checkpoint operations

### Routing Precedence

Match the issue against the three sections above. When an issue could fit more than one bucket, apply in this order:

1. `oncall: distributed checkpointing` — if the issue is about saving/loading distributed state.
2. `oncall: distributed infra` — otherwise, if the issue is about the communication/infra layer.
3. `oncall: distributed parallelisms` — otherwise (also the default when unsure).

### Edge Cases

- **NCCL timeout during FSDP training**: Route to `oncall: distributed infra`. The NCCL timeout is the bug, FSDP is just the user context.
- **FSDP state_dict saving**: Route to `oncall: distributed checkpointing` if about checkpoint save/load. Route to `oncall: distributed parallelisms` if about FSDP's internal state dict handling (e.g., `ShardedStateDictConfig`).
- **torchrun + DDP**: Route to `oncall: distributed infra` if the issue is about launching/rendezvous. Route to `oncall: distributed parallelisms` if the launch works but DDP training itself fails.
- **Backend selection errors at init_process_group**: Route to `oncall: distributed infra`.

---

## 2. Module Classification Signals

For each module label, here are the signals to look for:

### `module: fsdp`
- **Keywords**: FSDP, FullyShardedDataParallel, fully_shard, ShardingStrategy, MixedPrecision, fsdp_auto_wrap, FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, BackwardPrefetch, CPUOffload
- **Import paths**: `torch.distributed.fsdp`, `torch.distributed._composable.fsdp`
- **Stack trace patterns**: frames in `torch/distributed/fsdp/`, `torch/distributed/_composable/fsdp/`
- **Confusions**: Issues mentioning "sharding" could be DTensor sharding, not FSDP. Check for FSDP-specific APIs.

### `module: ddp`
- **Keywords**: DDP, DistributedDataParallel, find_unused_parameters, gradient_as_bucket_view, static_graph
- **Import paths**: `torch.nn.parallel.DistributedDataParallel`
- **Stack trace patterns**: frames in `torch/nn/parallel/distributed.py`
- **Confusions**: "data parallel" could mean legacy `DataParallel` (single-machine). Check for `DistributedDataParallel` specifically.

### `module: dtensor`
- **Keywords**: DTensor, distribute_tensor, Shard, Replicate, Placement, redistribute, DeviceMesh (when used with DTensor)
- **Import paths**: `torch.distributed.tensor`, `torch.distributed._tensor`
- **Stack trace patterns**: frames in `torch/distributed/tensor/`, `torch/distributed/_tensor/`
- **Confusions**: DTensor is the substrate for FSDP2 and TP. If the user is using FSDP2 and hits a DTensor error, label BOTH `module: fsdp` and `module: dtensor`.

### `module: c10d`
- **Keywords**: ProcessGroup, init_process_group, destroy_process_group, new_group, all_reduce, all_gather, broadcast, barrier, reduce_scatter, all_to_all, send, recv, isend, irecv, TCPStore, FileStore, PrefixStore, c10d, Work (as in collective Work handle)
- **Import paths**: `torch.distributed` (the base module, not subpackages)
- **Stack trace patterns**: frames in `torch/csrc/distributed/c10d/`, `torch/distributed/distributed_c10d.py`
- **Confusions**: Almost all distributed issues touch c10d indirectly. Only apply this label when the issue is specifically about the collective/PG/store layer, NOT when c10d just appears on the call path.

### `module: nccl`
- **Keywords**: NCCL error, ncclSystemError, ncclInternalError, NCCL timeout, NCCL watchdog, ProcessGroupNCCL, NCCL_SOCKET_IFNAME, NCCL_DEBUG, ncclAllReduce
- **Stack trace patterns**: frames mentioning `ProcessGroupNCCL`, `ncclCommInitRank`
- **Confusions**: "NCCL error at import" with no distributed code is a packaging issue (`module: binaries`), NOT `module: nccl`.

### `module: DeviceMesh`
- **Keywords**: DeviceMesh, init_device_mesh, mesh_dim, mesh["dp"], mesh["tp"]
- **Import paths**: `torch.distributed.device_mesh`
- **Confusions**: DeviceMesh is used by FSDP2, DTensor, and TP. Label `module: DeviceMesh` only if the bug is in DeviceMesh itself, not in code that uses it.

### `module: pipelining`
- **Keywords**: PipelineSchedule, PipelineStage, ScheduleGPipe, Schedule1F1B, ScheduleInterleaved1F1B, pipeline parallelism, pipeline_stage
- **Import paths**: `torch.distributed.pipelining`
- **Stack trace patterns**: frames in `torch/distributed/pipelining/`

### `module: rpc`
- **Keywords**: rpc, RRef, rpc_sync, rpc_async, remote, distributed autograd, dist_autograd, distributed optimizer
- **Import paths**: `torch.distributed.rpc`
- **Stack trace patterns**: frames in `torch/distributed/rpc/`
- **Note**: RPC is legacy and less actively maintained. Many RPC issues may be won't-fix.

### `module: elastic`
- **Keywords**: torchrun, elastic, elastic_launch, RendezvousHandler, elastic agent, worker group, torch.distributed.run
- **Import paths**: `torch.distributed.elastic`, `torch.distributed.run`
- **Stack trace patterns**: frames in `torch/distributed/elastic/`
- **Confusions**: "torchrun crashes" could be an elastic issue OR the user's script crashing via torchrun. Check the actual error.

### `module: data parallel`
- **Keywords**: DataParallel (NOT DistributedDataParallel), torch.nn.DataParallel, dp, replicas, scatter, gather
- **Import paths**: `torch.nn.DataParallel`
- **Note**: Legacy single-machine data parallel. Usually distinguish from DDP by checking: does the user call `init_process_group`? If no, it's legacy DataParallel.

### `module: symm_mem`
- **Keywords**: symmetric_memory, symm_mem, SymmetricMemory
- **Import paths**: `torch.distributed.tensor._symmetric_memory`, `torch._C._distributed_c10d`

### `module: context parallel`
- **Keywords**: context parallel, context_parallel, CP, sequence parallel (when referring to context parallelism)
- **Import paths**: `torch.distributed.tensor.parallel`

### `module: activation checkpointing`
- **Keywords**: checkpoint, activation checkpointing, gradient checkpointing, recomputation
- **Import paths**: `torch.utils.checkpoint`, `torch.distributed.algorithms._checkpoint`
- **Confusions**: Only label this if the issue is about activation checkpointing specifically, not about distributed checkpoints (saving model state). If it's "checkpoint saving", that's `oncall: distributed checkpointing`.

---

## 3. Confidence Calibration

### HIGH Confidence Examples

- Issue title: "FSDP2 crashes with `fully_shard` when using `MixedPrecision`" → `module: fsdp` (explicit API mention)
- Code snippet: `model = DDP(model, device_ids=[rank])` and it hangs → `module: ddp`
- Error: `ncclSystemError: Connection refused` during `all_reduce` → `module: nccl` + `module: c10d`

### MEDIUM Confidence Examples

- "My distributed training hangs after 100 steps" — no code, but mentions using multiple GPUs → could be DDP, FSDP, or a c10d issue. Best guess based on any other clues.
- "Error when loading checkpoint across different number of GPUs" — likely distributed checkpointing but could be FSDP state dict.
- Issue mentions both DTensor and FSDP2 — probably `module: fsdp` + `module: dtensor` but uncertain which is the root cause.

### LOW Confidence Examples

- "My model doesn't converge on multiple GPUs" — no code, no error, no mention of which distributed API.
- "torch.distributed doesn't work" — too vague.
- Issue in a language you can't parse or with no technical details.

---

## 4. Common Mislabel Traps

### NOT distributed issues (flag as mislabeled)

- **NCCL import error on single GPU**: `undefined symbol: ncclAlltoAll` at `import torch` is a packaging/build issue (`module: binaries`), not distributed.
- **CUDA errors on a single GPU**: `CUDA out of memory` or `CUDA error: device-side assert` without any distributed code.
- **torch.compile errors**: If the issue is purely about compilation failing (even on distributed code), it may belong to `oncall: pt2`. Check if the root cause is in the compiler or the distributed runtime.
- **Single-machine DataParallel**: `torch.nn.DataParallel` bugs are technically distributed but very low priority. Still label as `module: data parallel`.

### Label based on root cause, not keywords

- A stack trace through `c10d` doesn't mean `module: c10d` if the actual bug is in FSDP's use of collectives.
- An error mentioning "NCCL" might be a c10d watchdog timeout, not an NCCL bug.
- "distributed" in the title doesn't make it a distributed issue — read the actual content.

---

## 5. PT2 + Distributed Overlap

When an issue involves BOTH `torch.compile` and distributed:

- If the error is in the **compiler** (dynamo trace, inductor codegen, AOTAutograd) when compiling distributed ops → primarily `oncall: pt2`, but also add the relevant distributed `module:` label for visibility.
- If the error is in the **distributed runtime** but triggered after compilation → primarily distributed, still keep `oncall: pt2` if present.
- **Never remove `oncall: pt2`** if it was already applied by the PT-level bot. Add distributed labels alongside it.
- Common pattern: "torch.compile breaks FSDP" → likely needs both `oncall: pt2` + `module: fsdp`.
