#!/bin/bash
# Updates Triton to the pinned version for this copy of PyTorch
pip install --index-url https://download.pytorch.org/whl/nightly/ "pytorch-triton==$(cat .ci/docker/triton_version.txt)+$(head -c 10 .ci/docker/ci_commit_pins/triton.txt)"
