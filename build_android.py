"""Configuration for the Caffe2 installation targeted for Android.

To run the build, first create a standalone toolchain from the NDK root using:

./build/tools/make-standalone-toolchain.sh \
  --arch=arm --platform=android-21 \
  --toolchain=arm-linux-androideabi-4.9 \
  --install-dir=./standalone-toolchains/arm-linux-androideabi-4.9-android-21

(change the platform and toolchain if necessary) and update the
STANDALONE_TCHAIN_ROOT variable below.
"""

from build import Config

STANDALONE_TCHAIN_ROOT = (
    '/Users/jiayq/android-ndk-r10e/'
    'standalone-toolchains/arm-linux-androideabi-4.9-android-21/')

# We change necessary components in the Config class.

Config.CC = STANDALONE_TCHAIN_ROOT + 'bin/arm-linux-androideabi-g++'
Config.AR = STANDALONE_TCHAIN_ROOT + 'bin/arm-linux-androideabi-ar'
Config.GENDIR = "gen-android"
Config.USE_SYSTEM_PROTOBUF = False
Config.PROTOC_BINARY = 'gen/third_party/google/protoc'
Config.USE_LITE_PROTO = False
Config.USE_SYSTEM_EIGEN = False
Config.USE_GLOG = False
Config.USE_RTTI = False
Config.USE_OPENMP = False
Config.CUDA_DIR = "non-existing"
Config.MPICC = "non-existing"
Config.MPIRUN = "non-existing"
Config.OMPI_INFO = "non-existing"
Config.PYTHON_CONFIG = "non-existing"
Config.OPTIMIZATION_FLAGS = ["-Os"]


# brew.py
if __name__ == '__main__':
    from brewtool.brewery import Brewery
    import sys
    Brewery.Run(Config, sys.argv)
