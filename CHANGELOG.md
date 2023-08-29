# Change Log for PyTorch

## PyTorch for ROCm5.7 (upstream commit: 08c4a44)

### Fixed
- [SWDEV-396381] Fixed FSDP UTs by limiting to 8 GPUs
- Fixed Circular issue in hipify using current_state and iterative DFS.
- Added hipblaslt support. Requires setting the environment variable USE_HIPBLASLT=1 and must be on a supported architecture, for example gfx908.
