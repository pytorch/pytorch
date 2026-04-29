"""
Re-exports checkpoint_self / restore_self from the standalone
checkpoint/cuda_checkpoint.py module, which wraps the CUDA driver's
cuCheckpointProcess{Lock,Checkpoint,Restore,Unlock} APIs.
"""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "cuda_checkpoint",
    os.path.join(os.path.dirname(__file__), "../../checkpoint/cuda_checkpoint.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

checkpoint_self = _mod.checkpoint_self
restore_self = _mod.restore_self
CudaCheckpointError = _mod.CudaCheckpointError
