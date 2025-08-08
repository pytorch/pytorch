"""
PyTorch Inductor module.

PyTorch Inductor is a compilation framework designed to optimize PyTorch models
for better performance on various hardware targets including CPUs, GPUs, and
specialized accelerators.
"""

# Import standard torch._inductor components
from torch._inductor import config

# Import our XPU backend components and try to initialize
try:
    from .xpu_backends import integration
    
    # Initialize XPU backend if auto-init is enabled
    # This will check for Intel GPUs and enable optimizations if available
    from .xpu_backends.integration import initialize_xpu_backend
except ImportError:
    # XPU backend initialization failed - likely missing dependencies
    # or the feature is not available in this build
    pass
