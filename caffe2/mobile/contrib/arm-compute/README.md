# Caffe2 - ARM Compute Backend

## Build

To build, clone and install scons

```
brew install scons
```

set ANDROID_NDK to /opt/android_ndk/xxx(e.g. 15c)

setup toolchain:
arm
```
rm -rf PATH_TO_TOOLCHAIN
$ANDROID_NDK/build/tools/make_standalone_toolchain.py --arch arm --api 21 --install-dir PATH_TO_TOOLCHAIN
```

arm64
```
rm -rf PATH_TO_TOOLCHAIN
$ANDROID_NDK/build/tools/make_standalone_toolchain.py --arch arm64 --api 21 --install-dir PATH_TO_TOOLCHAIN
```

add the toolchain path to .bashrc/.zshrc etc.
e.g.
```
export PATH=$PATH:PATH_TO_TOOLCHAIN
```

use the build\_android.sh:

for 32bit
```
./scripts/build_android.sh -DUSE_ACL=ON -DBUILD_TEST=ON
```

for 64bit
```
./scripts/build_android.sh -DUSE_ACL=ON -DBUILD_TEST=ON -DUSE_NNPACK=OFF -DUSE_ARM64=ON
```

Before switch between 32 bit and 64 bit, please make sure to delete build\_android folder:
```
rm -rf build_android
```
## Test
Plug in an android device, and run a test

```
cd build_android
adb push bin/gl_conv_op_test /data/local/tmp && adb shell '/data/local/tmp/gl_conv_op_test'
```
or use a script to run them all

In caffe2 top level directory
```
./caffe2/mobile/contrib/arm-compute/run_tests.sh build_android
```

Note that some tests(fully_connected and alignment) have been disabled until the next release of ACL.
