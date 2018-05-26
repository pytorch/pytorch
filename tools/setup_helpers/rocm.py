from .env import check_env_flag
# Check if ROCM is enabled
WITH_ROCM = check_env_flag('WITH_ROCM')
ROCM_HOME = "/opt/rocm"
ROCM_VERSION = ""
