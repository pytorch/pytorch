__version__ = '{{VERSION}}'
debug = False
cuda = '{{CUDA_VERSION}}'
# TODO: use workspace status to stamp the correct version
git_version = ""
hip = None

# This is a gross monkey-patch hack that depends on the order of imports
# in torch/__init__.py
# TODO: find a more elegant solution to set `USE_GLOBAL_DEPS` for the bazel build
import torch
torch.USE_GLOBAL_DEPS = False
