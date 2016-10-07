"""Configuration for the Caffe2 installation.
"""

from build import Config
import sys

Config.USE_SYSTEM_PROTOBUF = False
Config.PROTOC_BINARY = 'gen/third_party/google/protoc'
Config.USE_OPENMP = False

if __name__ == '__main__':
    from brewtool.brewery import Brewery
    Brewery.Run(
        Config,
        ['build_android_prepare.py',
         'build', '//third_party/google:protoc'])
else:
    print('This script is not intended to be used as an imported module.')
    sys.exit(1)
