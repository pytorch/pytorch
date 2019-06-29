from .env import check_negative_env_flag

USE_NNPACK = not check_negative_env_flag('USE_NNPACK')
