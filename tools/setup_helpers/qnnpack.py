from .env import check_negative_env_flag

USE_QNNPACK = not check_negative_env_flag('USE_QNNPACK')
