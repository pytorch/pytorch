import os
from .env import check_env_flag, check_negative_env_flag

# Get ROCm Home Path
ROCM_HOME = os.getenv("ROCM_HOME", "/opt/rocm")
ROCM_VERSION = ""
USE_ROCM = False

# Check if ROCm disabled.
if check_negative_env_flag("USE_ROCM"):
    USE_ROCM = False
else:
    # If ROCM home exists or we explicitly enable ROCm
    if os.path.exists(ROCM_HOME) or check_env_flag('USE_ROCM'):
        USE_ROCM = True
