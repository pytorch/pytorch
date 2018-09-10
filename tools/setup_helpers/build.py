from .env import check_env_flag, check_negative_env_flag

BUILD_BINARY = check_env_flag('BUILD_BINARY')
BUILD_TEST = not check_negative_env_flag('BUILD_TEST')
USE_OPENCV = check_env_flag('USE_OPENCV')
