# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments

import os
import subprocess
import sys

import torch

from torch.testing._internal.common_cuda import IS_JETSON, IS_WINDOWS

if torch.cuda.is_available() and not IS_JETSON and not IS_WINDOWS:
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    test_cuda_filepath = os.path.join(current_dir, "test_cuda.py")
    test_cudagraphs_filepath = os.path.join(current_dir, "dynamo/test_cudagraphs.py")
    test_cudagraph_trees_filepath = os.path.join(
        current_dir, "inductor/test_cudagraph_trees.py"
    )

    subprocess.run([sys.executable, test_cuda_filepath], env=env, check=True)
    subprocess.run([sys.executable, test_cudagraphs_filepath], env=env, check=True)
    subprocess.run([sys.executable, test_cudagraph_trees_filepath], env=env, check=True)
