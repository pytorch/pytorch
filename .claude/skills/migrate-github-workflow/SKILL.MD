---
name: migrate-github-workflow
description: Use when migrating an existing GitHub workflow from AWS EC2 runners (Docker-in-Docker pattern) to ARC (Actions Runner Controller) runners on Kubernetes.
---

# Migrate GitHub workflow to run on ARC runners

## Overview

The old CI pattern runs jobs on EC2 instances, pulls a custom Docker image
from ECR, and shells into it via `docker run`/`docker exec`. The new
pattern runs jobs directly inside a container using GitHub's native
`container:` directive on ARC (Kubernetes) runners — no Docker-in-Docker,
no ECR, no SSH setup.

The core transformation: everything baked into the custom Docker image
(Python, system packages, ecosystem libraries) becomes explicit workflow
steps, and the custom image is replaced by a standard base image
(manylinux or nvidia/cuda).

## When NOT to Use

- **Windows or macOS workflows** — leave intact
- **Jobs on GitHub-hosted runners** (`ubuntu-latest`, `linux.24_04.*`) — leave intact
- **Anything outside the `jobs:` block** (name, triggers, permissions) — do not modify
- **Partner self-hosted runners using Docker-in-Docker** (e.g., IBM
  `linux.s390x`) — these are not migrating to ARC. Extract their code
  paths into separate dedicated workflows (e.g., `_s390x-build.yml`,
  `_s390x-test.yml`) so the main `_linux-build.yml` / `_linux-test.yml`
  can be cleanly migrated without s390x conditionals. The extracted
  workflows keep the original Docker-in-Docker pattern, runner labels,
  and artifact upload method (`actions/upload-artifact` instead of S3).
  This applies to any partner-provided runner that still relies on
  Docker-in-Docker and is not part of the ARC migration.

## Prerequisites

- The file must be a YAML GitHub workflow under `.github/workflows/`
- The Dockerfile for existing CI images is at
  `https://github.com/pytorch/pytorch/blob/main/.ci/docker/ubuntu/Dockerfile`
  (used for both CPU and CUDA jobs)
- The list of available ARC runners is at
  `https://github.com/pytorch/ciforge/tree/main/arc/runners/arc/hooks`
- The list of AWS EC2 runners being migrated away from is at
  `https://github.com/pytorch/test-infra/blob/main/.github/scale-config.yml`

## Step 1: Categorize the workflow

Determine which category the workflow falls into:

### Category A: Top-level workflow calling reusable workflows

Example: `pull.yml`. These pass parameters like `runner`,
`docker-image-name`, and `test-matrix` to reusable workflows.

**What to migrate:**
- Update `runner` and `docker-image-name` parameters
- Update runner labels inside `test-matrix` JSON
- Keep `runner_prefix` as-is (it will be empty for ARC migrations)

### Category B: Reusable workflow with EC2 runners and Docker logic

Example: `_linux-build.yml`. These have `runs-on` with EC2 runner labels,
`setup-ssh`, ECR login, `calculate-docker-image`, `docker run`/`docker exec`.

**What to migrate:** Full rewrite of job structure (see steps 3-8).

For workflows or jobs that don't fall into Category A or B, see
"When NOT to Use" above.

## Step 2: Understand the Docker image

Look at `https://github.com/pytorch/pytorch/blob/main/.ci/docker/ubuntu/Dockerfile`
to understand what the existing Docker image installs. After migration,
these setup steps move into the workflow itself:

- Python dependencies → `setup-uv` + `uv pip install`
- System dependencies via `apt-get` → use `dnf` instead (manylinux is
  CentOS-based)
- CUDA dependencies → handled by the `nvidia/cuda` base image
- Ecosystem libraries pinned in `.ci/docker/ci_commit_pins/` → build
  them in CI steps

## Step 3: Map runners to ARC labels

Map EC2 runner labels to ARC runner labels using this table. The ARC
label always starts with `a.` prefix.

**Size matching rule:** When the table says "Use c7i/r7i/m8g instead",
pick the c7i/r7i/m8g size that matches the old runner's size suffix.
If the exact size doesn't exist as an ARC runner, use the next larger
size. Examples:
- `linux.large` → `a.linux.c7i.2xlarge` (no c7i.large exists, round up)
- `linux.2xlarge` → `a.linux.c7i.2xlarge`
- `linux.4xlarge` → `a.linux.c7i.4xlarge`
- `linux.12xlarge` → `a.linux.c7i.12xlarge`
- `linux.9xlarge.ephemeral` → `a.linux.c7i.12xlarge` (round up)

### Runner mapping table

| EC2 runner label | ARC runner label |
|---|---|
| linux.large | a.linux.c7i.2xlarge |
| linux.2xlarge | a.linux.c7i.2xlarge |
| linux.4xlarge | a.linux.c7i.4xlarge |
| linux.9xlarge.ephemeral | a.linux.c7i.12xlarge |
| linux.12xlarge | a.linux.c7i.12xlarge |
| linux.10xlarge.avx2 | a.linux.10xlarge.avx2 |
| linux.2xlarge.amx | a.linux.2xlarge.amx |
| linux.2xlarge.avx2 | a.linux.2xlarge.avx2 |
| linux.8xlarge.amx | a.linux.8xlarge.amx |
| linux.24xlarge.amd | a.linux.24xlarge.amd |
| linux.c7i.2xlarge | a.linux.c7i.2xlarge |
| linux.c7i.4xlarge | a.linux.c7i.4xlarge |
| linux.c7i.12xlarge | a.linux.c7i.12xlarge |
| linux.4xlarge.memory | a.linux.r7i.4xlarge |
| linux.8xlarge.memory | a.linux.r7i.12xlarge |
| linux.12xlarge.memory | a.linux.r7i.12xlarge |
| linux.24xlarge.memory | a.linux.r7i.12xlarge |
| linux.r7i.2xlarge | a.linux.r7i.2xlarge |
| linux.r7i.4xlarge | a.linux.r7i.4xlarge |
| linux.r7i.12xlarge | a.linux.r7i.12xlarge |
| linux.arm64.2xlarge.ephemeral | a.linux.arm64.2xlarge |
| linux.arm64.m7g.4xlarge | a.linux.arm64.m8g.4xlarge |
| linux.arm64.m7g.metal | a.linux.arm64.m7g.metal |
| linux.arm64.m8g.4xlarge | a.linux.arm64.m8g.4xlarge |
| linux.arm64.r7g.12xlarge.memory | a.linux.arm64.m8g.4xlarge |
| linux.arm64.r8g.12xlarge.memory | a.linux.arm64.r8g.12xlarge |
| linux.aws.a100 | a.linux.aws.a100 |
| linux.aws.h100 | a.linux.aws.h100 |
| linux.aws.h100.4 | a.linux.aws.h100.4 |
| linux.aws.h100.8 | a.linux.aws.h100.8 |
| linux.dgx.b200 | a.linux.dgx.b200 |
| linux.dgx.b200.8 | a.linux.dgx.b200.8 |
| linux.g4dn.4xlarge.nvidia.gpu | a.linux.g6.4xlarge |
| linux.g4dn.12xlarge.nvidia.gpu | a.linux.g6.12xlarge |
| linux.g4dn.metal.nvidia.gpu | TBD |
| linux.g5.4xlarge.nvidia.gpu | a.linux.g6.4xlarge |
| linux.g5.12xlarge.nvidia.gpu | a.linux.g6.12xlarge |
| linux.g6.4xlarge.experimental.nvidia.gpu | a.linux.g6.4xlarge |
| linux.g6.12xlarge.nvidia.gpu | a.linux.g6.12xlarge |
| linux.24xl.spr-metal | TBD |

**Note on pytorch-canary:** When the target repo is `pytorch-canary`
(the test bed), ARC runner labels get a `c.` prefix prepended, e.g.,
`a.linux.c7i.2xlarge` becomes `c.a.linux.c7i.2xlarge`. For `pytorch`
proper, no extra prefix is needed.

## Step 4: Choose the base Docker image

Replace custom PyTorch CI Docker images with standard base images, used
via the GitHub `container:` directive:

| Job type | Base image |
|---|---|
| CPU x86_64 | `quay.io/pypa/manylinux_2_34_x86_64` |
| CPU aarch64 | `quay.io/pypa/manylinux_2_34_aarch64` |
| CUDA | `nvidia/cuda:<version>-cudnn-devel-ubuntu24.04` |

For CUDA images: match the CUDA version to the job. Use only `devel`
images with cudnn, prefer Ubuntu 24.04. Example:
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` for CUDA 12.8 jobs.

Refer to `https://github.com/pytorch/pytorch/blob/main/.ci/docker/build.sh`
to find out how existing Docker images map to CUDA versions.

If unsure, use `<PLACEHOLDER>` for the image name.

## Step 5: Unpack Docker image setup into workflow steps

All setup previously baked into the Docker image must now be explicit
workflow steps:

- **Python:** Use `setup-uv` action + `uv pip install` for dependencies
- **CUDA:** Handled by the `nvidia/cuda` base image (no manual install)
- **System packages:** Use the correct package manager for the base image:

  | Base image | Package manager | Reason |
  |---|---|---|
  | `manylinux_2_34_*` | `dnf` | CentOS/AlmaLinux-based |
  | `nvidia/cuda:*-ubuntu*` | `apt-get` | Ubuntu-based |
- **Ecosystem libraries** (pinned in `.ci/docker/ci_commit_pins/`):
  Build them as CI steps
- **Other dependencies:** Keep installation the same, add a comment to
  migrate to S3 if not already there

## Step 6: Composite GitHub actions

These composite actions continue to be used. All S3-uploading actions
need AWS credentials via OIDC. Use `aws-actions/configure-aws-credentials`
with role `arn:aws:iam::308535385114:role/gha_workflow_upload-benchmark-results`:

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@e2dba801e5e982e0da498d67a8769e14a0f83498
  with:
    role-to-assume: arn:aws:iam::308535385114:role/gha_workflow_upload-benchmark-results
    aws-region: us-east-1
```

Actions that need this credential step before them:
- `upload-artifact-s3`
- `upload-sccache-stats`
- `upload-utilization-stats`
- `pytest-cache-upload`
- `upload-benchmark-results`

## Step 7: Reusable workflow parameters

Reusable workflows continue to be used. Update their `docker-image-name`
parameter to use the base Docker image from step 4.

## Step 8: Rewrite setup, teardown, and Docker logic

Remove or replace these patterns:

| Remove | Replacement |
|---|---|
| `setup-ssh` | Remove entirely |
| `setup-python` | `setup-uv` with same Python version |
| ECR login (`ecr-login`) | Remove entirely |
| `calculate-docker-image` | Remove (image is now in `container:`) |
| `pull-docker-image` | Remove (GitHub handles container pull) |
| Container-detection checks | Remove (always running in container) |
| `setup-nvidia` / GPU_FLAG | Remove (handled by nvidia/cuda image) |
| `docker run` / `docker exec` | Run commands directly |
| Docker cleanup / `teardown-linux` | Remove entirely |

## Examples

Linter migration (already completed):
- Before: https://github.com/pytorch/pytorch/blob/main/.github/workflows/lint.yml
- After: https://github.com/pytorch/pytorch-canary/blob/main/.github/workflows/lint.yml

Key patterns from the lint migration:
- Reusable `_lint.yml` uses `container: image: ${{ inputs.docker-image }}`
- Caller passes `docker-image: quay.io/pypa/manylinux_2_34_x86_64`
- System deps via `dnf install -y clang doxygen git graphviz nodejs npm`
- Python deps via `setup-uv` + `uv pip install -r .ci/docker/requirements-ci.txt`
- Jobs on GitHub runners (e.g., `pr-sanity-checks` on `linux.24_04.4x`)
  are left unchanged

## Tips for Success

- **Change summary comment:** Always add a comment block at the top of
  the migrated workflow file listing all changes made. Example:
  ```yaml
  # Migration to ARC runners:
  # - Runner: linux.4xlarge → c.a.linux.c7i.4xlarge
  # - Container: quay.io/pypa/manylinux_2_34_x86_64
  # - Replaced setup-linux with direct setup-uv (upstream action has old EC2 logic)
  # - Replaced pip install with uv pip install
  # - Added OIDC AWS credentials (no EC2 instance profile on ARC)
  # - Installed awscli early (was pre-installed on EC2)
  # - Added dnf install unzip (not on manylinux by default)
  # - Removed teardown-linux
  ```
- **Commit strategy:** When migrating a top-level workflow and its
  reusable workflows together, prefer separate commits for each file
- **TBD runners:** If a runner maps to TBD, leave a `<TBD>` placeholder
  and note it for follow-up
- **Missing ARC runner labels:** If the ARC runner label from the
  mapping table is not listed at
  `https://github.com/pytorch/ciforge/tree/main/arc/runners/arc/hooks`,
  use the label anyway and add a comment (e.g.,
  `# TODO: verify ARC runner label exists`) — new runners are added
  regularly and may not be in the hooks directory yet
