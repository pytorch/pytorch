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
curl -L -o /tmp/bazel-skylib-1.0.2.tar.gz https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz
rm -rf bazel-skylib/
mkdir bazel-skylib
tar -xf /tmp/bazel-skylib-1.0.2.tar.gz -C bazel-skylib/

# glog
printf "\n[Downloading glog-0.4.0]\n"
curl -L -o /tmp/glog-0.4.0.tar.gz https://github.com/google/glog/archive/v0.4.0.tar.gz
tar -xf /tmp/glog-0.4.0.tar.gz -C /tmp/
rm -rf glog/
mv /tmp/glog-0.4.0/ glog/

# ruy
printf "\n[Downloading ruy]\n"
curl -L -o /tmp/ruy.zip https://github.com/google/ruy/archive/a09683b8da7164b9c5704f88aef2dc65aa583e5d.zip
unzip -q /tmp/ruy.zip -d /tmp/
rm -rf ruy/
mv /tmp/ruy-a09683b8da7164b9c5704f88aef2dc65aa583e5d ruy/
