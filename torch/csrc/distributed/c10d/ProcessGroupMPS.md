# ProcessGroupMPS: Supported Operations

RDMA-only c10d backend for Apple Silicon Macs connected via Thunderbolt 5.
Collectives are layered on top of [JACCL](../../../../third_party/jaccl/),
MLX's standalone RDMA-over-Thunderbolt library (vendored in
`third_party/jaccl/`). Construction requires `ibv_alloc_pd` to succeed on every
rank; otherwise the constructor throws and users should select the `gloo`
backend.

## Implemented

| Op          | JACCL primitive                      | Notes                                                                              |
|-------------|--------------------------------------|------------------------------------------------------------------------------------|
| `allreduce` | `all_sum` / `all_min` / `all_max`    | `SUM` / `MIN` / `MAX` only. Dtypes: any JACCL dtype (fp16/bf16/fp32/fp64/int8-64/uint8-64/bool/complex64) |
| `broadcast` | `send` + `recv`                      | Root sends to each peer; all ranks, any root                                       |
| `send`      | `send`                               | Single tensor, any dst                                                             |
| `recv`      | `recv`                               | Single tensor, any src                                                             |
| `barrier`   | `barrier`                            | Direct call to `jaccl::Group::barrier()`                                           |

## Not implemented

- `allreduce` with `PRODUCT`
- `allgather` / `_allgather_base`
- `allgather_into_tensor_coalesced`
- `reduce_scatter` / `_reduce_scatter_base`
- `reduce` (to single rank)
- `gather` (to single rank)
- `scatter` (from single rank)
- `alltoall` / `alltoall_base`
- `*_coalesced` variants
- `recvAnysource`
- `allreduce_sparse`
- `startCoalescing` / `endCoalescing`
- `monitored_barrier`
