cd third_party
python generate-cpuinfo-wrappers.py
python generate-xnnpack-wrappers.py

curl -L -o /tmp/bazel-skylib-1.0.2.tar.gz https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz
mkdir bazel-skylib
tar -xf /tmp/bazel-skylib-1.0.2.tar.gz -C bazel-skylib/
