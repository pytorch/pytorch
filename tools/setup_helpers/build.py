from .env import check_negative_env_flag

BUILD_BINARY = not check_negative_env_flag('BUILD_BINARY')
BUILD_TEST = not check_negative_env_flag('BUILD_TEST')
