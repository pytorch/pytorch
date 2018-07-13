from .env import check_env_flag
# Check if ROCM is enabled
USE_ROCM = check_env_flag('USE_ROCM')
ROCM_HOME = "/opt/rocm"
ROCM_VERSION = ""
