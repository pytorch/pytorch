#!/bin/bash
printf "\nCreating .buckconfig\n"
cp .buckconfig.oss .buckconfig

PROXY=""
if [ "$1" == "devserver" ]; then
   echo -e '\n[download]\n   proxy_host=fwdproxy\n   proxy_port=8080\n   proxy_type=HTTP\n' >> .buckconfig
   PROXY="$(fwdproxy-config curl)"
   printf "using proxy $PROXY\n\n"
fi

cat .buckconfig

cd third_party || return

printf "\nGenerating cpuinfo wrappers\n"
python3 generate-cpuinfo-wrappers.py

printf "\nGenerating xnnpack wrappers\n"
python3 generate-xnnpack-wrappers.py

# bazel-skylib
printf "\nDownloading bazel-skylib\n"
rm -rf bazel-skylib; mkdir bazel-skylib
curl --retry 3 -L $PROXY https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz|tar zx -C bazel-skylib

# glog
printf "\nDownloading glog\n"
rm -rf glog; mkdir glog
curl --retry 3 -L $PROXY https://github.com/google/glog/archive/v0.4.0.tar.gz | tar zx -C glog --strip-components 1

# ruy
printf "\nDownloading ruy\n"
curl --retry 3 -L $PROXY -o /tmp/ruy.zip https://github.com/google/ruy/archive/a09683b8da7164b9c5704f88aef2dc65aa583e5d.zip
unzip -q /tmp/ruy.zip -d /tmp/
rm -rf ruy/
mv /tmp/ruy-a09683b8da7164b9c5704f88aef2dc65aa583e5d ruy/
