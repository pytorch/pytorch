import subprocess

import modal

from tools.sandbox.common import repository_root


CI_IMAGE_NAME = "pytorch-linux-jammy-cuda13.2-cudnn9-py3-gcc11"

# sshd gives sessions a clean environment, so the image's ENV never reaches an ssh session.
# Login shells read /etc/profile.d and interactive shells read /etc/bash.bashrc; the toolchain
# setup goes in the first and is sourced from the second, so both kinds of shell see it.
# PATH is the CI image's own, minus /opt/cache/bin, whose sccache compiler wrappers expect CI's
# S3 bucket; ccache takes that role.
SANDBOX_PROFILE = (
    "export PATH=/opt/rust/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/python-3.11-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "export USE_SYSTEM_NCCL=1 NCCL_INCLUDE_DIR=/usr/local/cuda/include/ NCCL_LIB_DIR=/usr/local/cuda/lib64/ MAGMA_HOME=/usr/local/cuda/magma",
    # Cover the NVIDIA GPU architectures offered by the flavors in flavor.py.
    'export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0"',
    "export CCACHE_DIR=/root/.ccache CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache",
)


def ci_image_reference() -> str:
    """The PyTorch CI image matching this checkout: tags are keyed by the .ci/docker tree hash."""
    tree_hash = subprocess.run(
        ["git", "-C", repository_root(), "rev-parse", "HEAD:.ci/docker"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"ghcr.io/pytorch/ci-image:{CI_IMAGE_NAME}-{tree_hash}"


def base_image() -> modal.Image:
    """The CI toolchain plus what a dev box needs: sshd, ccache, tmux, the GitHub CLI and Claude Code."""
    return (
        modal.Image.from_registry(ci_image_reference())
        .apt_install("openssh-server", "ccache", "tmux")
        .run_commands(
            # GitHub CLI from its own apt repository; Ubuntu jammy's package is years old.
            "mkdir -p -m 755 /etc/apt/keyrings"
            " && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg"
            " -o /etc/apt/keyrings/githubcli-archive-keyring.gpg"
            " && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg",
            'echo "deb [arch=$(dpkg --print-architecture)'
            " signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg]"
            ' https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list',
            "apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*",
        )
        .run_commands(
            "curl -fsSL https://claude.ai/install.sh | bash",
            "ln -s /root/.local/bin/claude /usr/local/bin/claude",
        )
        .run_commands(
            "printf '%s\\n' "
            + " ".join(f"'{line}'" for line in SANDBOX_PROFILE)
            + " > /etc/profile.d/sandbox.sh",
            "echo '. /etc/profile.d/sandbox.sh' >> /etc/bash.bashrc",
        )
    )
