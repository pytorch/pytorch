# PyTorch TheRock Nightly CI Workflow

## Summary

This PR adds a new GitHub Actions workflow to build PyTorch nightly using TheRock nightly wheels for ROCm. The workflow runs at midnight UTC daily and consists of two jobs:

1. **Build TheRock-based Docker image** - Creates a CI Docker image with TheRock nightly ROCm wheels
2. **Build PyTorch nightly** - Builds PyTorch from the `nightly` branch using the TheRock Docker image

## Changes Made

### 1. New Workflow: `.github/workflows/therock-nightly.yml`

**Features:**
- Scheduled to run at midnight UTC daily via cron trigger
- Manual workflow dispatch capability
- Two-job pipeline:
  - `build-therock-docker`: Builds Docker image with TheRock nightly ROCm
  - `build-pytorch-nightly`: Builds PyTorch using the TheRock image
- Artifact management: Docker image passed between jobs, PyTorch wheels uploaded
- Configurable via environment variables (ROCm version, Python version, GPU arch)

**Key Configuration:**
```yaml
env:
  PYTORCH_NIGHTLY_REF: nightly
  THEROCK_NIGHTLY_INDEX_URL: https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/
  USE_THEROCK_NIGHTLY: "1"
  UBUNTU_VERSION: "22.04"
  ROCM_VERSION: "6.4"
  ANACONDA_PYTHON_VERSION: "3.11"
  PYTORCH_ROCM_ARCH: "gfx942"
```

### 2. Dockerfile Updates: `.ci/docker/ubuntu-rocm/Dockerfile`

**Added build arguments:**
- `USE_THEROCK_NIGHTLY` - Flag to enable TheRock nightly installation
- `THEROCK_NIGHTLY_INDEX_URL` - URL for TheRock nightly wheel index

**Changes:**
```dockerfile
ARG USE_THEROCK_NIGHTLY
ARG THEROCK_NIGHTLY_INDEX_URL
ENV USE_THEROCK_NIGHTLY=${USE_THEROCK_NIGHTLY}
ENV THEROCK_NIGHTLY_INDEX_URL=${THEROCK_NIGHTLY_INDEX_URL}
```

These environment variables are passed to `install_rocm.sh` during the Docker build.

### 3. ROCm Installation Script: `.ci/docker/common/install_rocm.sh`

**Added TheRock nightly support:**
- Checks for `USE_THEROCK_NIGHTLY=1` environment variable
- When enabled:
  - Removes existing `/opt/rocm` directory
  - Installs `rocm-sdk[libraries,devel]` from TheRock nightly index
  - Sets up ROCm environment using `rocm-sdk` CLI helper
  - Creates `/etc/profile.d/rocm-sdk.sh` for persistent environment configuration
  - Early exits to skip traditional ROCm installation

**Logic flow:**
```bash
if [[ "${USE_THEROCK_NIGHTLY:-0}" == "1" ]]; then
  # Install TheRock nightly wheels
  python3 -m pip install --index-url "${THEROCK_NIGHTLY_INDEX_URL}" "rocm-sdk[libraries,devel]"
  # Configure environment
  export ROCM_HOME="$(rocm-sdk path --root)"
  # Exit early
  exit 0
fi
# Traditional ROCm installation continues...
```

## Workflow Details

### Job 1: build-therock-docker

1. Checks out PyTorch repository
2. Sets up Docker Buildx
3. Builds Docker image using `.ci/docker/ubuntu-rocm/Dockerfile` with TheRock build args
4. Tags image with date (e.g., `nightly-20251121`) and `latest`
5. Saves and uploads Docker image as artifact

### Job 2: build-pytorch-nightly

1. Checks out PyTorch nightly branch with submodules
2. Downloads Docker image artifact from previous job
3. Runs PyTorch build inside the TheRock container:
   - Sources ROCm SDK environment from `/etc/profile.d/rocm-sdk.sh`
   - Activates conda environment
   - Installs PyTorch dependencies
   - Builds PyTorch wheel with `USE_ROCM=1`
4. Uploads built wheel artifacts (retained for 7 days)
5. Generates build summary in GitHub Actions UI

## Benefits

1. **Automated nightly testing** of PyTorch with TheRock nightly ROCm wheels
2. **Early detection** of compatibility issues between PyTorch and TheRock
3. **Reproducible builds** via Docker containerization
4. **Artifact preservation** for debugging and testing
5. **No impact** on existing CI workflows (opt-in via build args)

## Testing

To test this workflow:

1. **Manual trigger**: Go to Actions → therock-nightly → Run workflow
2. **Local Docker build**:
   ```bash
   docker build \
     -f .ci/docker/ubuntu-rocm/Dockerfile \
     --build-arg UBUNTU_VERSION=22.04 \
     --build-arg ROCM_VERSION=6.4 \
     --build-arg ANACONDA_PYTHON_VERSION=3.11 \
     --build-arg PYTORCH_ROCM_ARCH=gfx942 \
     --build-arg USE_THEROCK_NIGHTLY=1 \
     --build-arg THEROCK_NIGHTLY_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ \
     --build-arg BUILD_ENVIRONMENT=therock-nightly-rocm6.4 \
     -t pytorch-therock-rocm-ci:test \
     .
   ```

## Future Enhancements

- Add test execution step after build
- Push Docker images to a registry for reuse
- Support multiple GPU architectures (gfx908, gfx90a, etc.)
- Add Slack/email notifications on build failures
- Create separate workflow for TheRock stable releases

## Files Changed

- `.github/workflows/therock-nightly.yml` (new)
- `.ci/docker/ubuntu-rocm/Dockerfile` (modified)
- `.ci/docker/common/install_rocm.sh` (modified)

---

## PR Checklist

- [ ] Workflow tested with manual trigger
- [ ] Docker image builds successfully with TheRock nightly wheels
- [ ] PyTorch builds successfully inside TheRock Docker image
- [ ] Artifacts are properly uploaded
- [ ] No breaking changes to existing CI workflows
- [ ] Documentation updated (if needed)
