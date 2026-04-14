# RHEL 9.6 Docker Image for PyTorch CI

Docker image for building and testing PyTorch on RHEL 9.6 with CUDA 13.0 and NVIDIA H200 GPUs.

## What's Included

- RHEL 9.6 (UBI9 base)
- Python 3.12, GCC 11, Ninja, LLD
- CUDA 13.0 (via upstream `.ci/docker/common/install_cuda.sh`)
- cuDNN, NCCL, cuSparseLt, NVSHMEM
- Conda environment (`pytorch_build`)
- OpenBLAS, Triton

## Building

```bash
cd .ci/docker
podman build -f rhel9/Dockerfile -t ci-image:pytorch-rhel-9.6-py3.12-gcc11 .
```

The Dockerfile uses upstream common scripts (`install_cuda.sh`, `install_nccl.sh`, `install_cusparselt.sh`) to install CUDA and related libraries, keeping version pins consistent with the rest of CI.

## RHEL-Specific Notes

- Uses Podman (rootless) instead of Docker
- No jenkins user — runs as root
- `/etc/profile.d/which2.sh` is removed to prevent sccache spawn failures
- `LIBRARY_PATH` includes CUDA stubs for static lib linking
- `libgomp.so` symlink created (UBI9 only ships runtime `libgomp.so.1`)
