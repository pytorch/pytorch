from .env import check_env_flag, check_negative_env_flag

BUILD_BINARY = check_env_flag('BUILD_BINARY')
BUILD_TEST = not check_negative_env_flag('BUILD_TEST')
BUILD_CAFFE2_OPS = not check_negative_env_flag('BUILD_CAFFE2_OPS')
USE_LEVELDB = check_env_flag('USE_LEVELDB')
USE_LMDB = check_env_flag('USE_LMDB')
USE_OPENCV = check_env_flag('USE_OPENCV')
USE_TENSORRT = check_env_flag('USE_TENSORRT')
USE_FFMPEG = check_env_flag('USE_FFMPEG')
USE_FBGEMM = not (check_env_flag('NO_FBGEMM') or check_negative_env_flag('USE_FBGEMM'))
