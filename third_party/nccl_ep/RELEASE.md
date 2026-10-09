# NCCL EP (Expert Parallelism) - Release Notes
NCCL EP is a high-performance NCCL API extension for efficient Mixture-of-Experts (MoE) communication.
It provides optimized dispatch and combine primitives for Expert Parallelism (EP) across distributed GPU systems
implemented on top of NCCL Device API: Load-Store Accessible (LSA) and GPU-Initiated Networking (GIN) operations.


## Known Limitations

This release has the following limitations:

| Limitation                       | Description |
|----------------------------------|-------------|
| **Limited QA coverage**          | This release is considered experimental and had limited Quality Assurance (QA) testing |
| **No FP8 support**               | FP8 data types are not currently supported |
| **Fabric selection**             | Current version of API does not provide a way to force RDMA-only communication |
| **Up to 8 nodes**                | Maximum of 64 GPUs (8 nodes × 8 GPUs) is supported |
| **Up to 8 ranks per node**       | Only up to 8 GPUs per node is supported |
| **max_tokens_per_rank required** | `NCCL_EP_AUTO` is not yet supported; `max_tokens_per_rank` must be set to the per-rank batch size (max tokens any single rank will dispatch) |
| **Max Num Tokens Support**       | Currently implementation of HT kernels is using a define variable configured in `common.h` as `MAX_SUPPORTED_TOKENS_PER_RANK` |
| **HT mode performance**          |  HT mode is currently being tuned for performance and may demonstrate sub-optimal results in this release |


## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [GPU-Initiated Networking Paper](https://arxiv.org/abs/2511.15076)
- [NCCL EP Readme](README.md)

## License

See LICENSE.txt for license information.
