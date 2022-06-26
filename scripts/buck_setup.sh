#!/bin/bash
printf "\n[Creating .buckconfig]\n"
cp .buckconfig.oss .buckconfig

cd third_party || return

printf "\n[Generating wrappers for cpuionfo]\n"
python3 generate-cpuinfo-wrappers.py

printf "\n[Generating wrappers for xnnpack]\n"
python3 generate-xnnpack-wrappers.py

# bazel-skylib
printf "\n[Downloading bazel-skylib-1.0.2]\n"
rm -rf bazel-skylib; mkdir bazel-skylib
curl -L https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz|tar zx -C bazel-skylib

# glog
printf "\n[Downloading glog-0.4.0]\n"
rm -rf glog; mkdir glog
curl -L https://github.com/google/glog/archive/v0.4.0.tar.gz | tar zx -C glog --strip-components 1

# ruy
printf "\n[Downloading ruy]\n"
curl -L -o /tmp/ruy.zip https://github.com/google/ruy/archive/a09683b8da7164b9c5704f88aef2dc65aa583e5d.zip
unzip -q /tmp/ruy.zip -d /tmp/
rm -rf ruy/
mv /tmp/ruy-a09683b8da7164b9c5704f88aef2dc65aa583e5d ruy/
