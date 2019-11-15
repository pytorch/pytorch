import argparse
import re
import sys

# Modify this variable if you want to change the set of default jobs
# which are run on all pull requests.
#
# WARNING: Actually, this is a lie; we're currently also controlling
# the set of jobs to run via the Workflows filters in CircleCI config.

default_set = set([
    # PyTorch CPU
    # Selected oldest Python 2 version to ensure Python 2 coverage
    'pytorch-linux-xenial-py2.7.9',
    # PyTorch CUDA
    'pytorch-linux-xenial-cuda9-cudnn7-py3',
    # PyTorch ASAN
    'pytorch-linux-xenial-py3-clang5-asan',
    # PyTorch DEBUG
    'pytorch-linux-xenial-py3.6-gcc5.4',
    # LibTorch
    'pytorch-libtorch-linux-xenial-cuda9-cudnn7-py3',

    # Caffe2 CPU
    'caffe2-py2-mkl-ubuntu16.04',
    # Caffe2 CUDA
    'caffe2-py3.5-cuda10.1-cudnn7-ubuntu16.04',
    # Caffe2 ONNX
    'caffe2-onnx-py2-gcc5-ubuntu16.04',
    'caffe2-onnx-py3.6-clang7-ubuntu16.04',
    # Caffe2 Clang
    'caffe2-py2-clang7-ubuntu16.04',
    # Caffe2 CMake
    'caffe2-cmake-cuda9.0-cudnn7-ubuntu16.04',
    # Caffe2 CentOS
    'caffe2-py3.6-devtoolset7-cuda9.0-cudnn7-centos7',

    # Binaries
    'manywheel 2.7mu cpu devtoolset7',
    'libtorch 2.7m cpu devtoolset7',
    'libtorch 2.7m cpu gcc5.4_cxx11-abi',
    'libtorch 2.7 cpu',
    'libtorch-ios-10.2.1-nightly-x86_64-build',
    'libtorch-ios-10.2.1-nightly-arm64-build',
    'libtorch-ios-10.2.1-nightly-binary-build-upload',

    # Caffe2 Android
    'caffe2-py2-android-ubuntu16.04',
    # Caffe2 OSX
    'caffe2-py2-system-macos10.13',
    # PyTorch OSX
    'pytorch-macos-10.13-py3',
    'pytorch-macos-10.13-cuda9.2-cudnn7-py3',
    # PyTorch Android
    'pytorch-linux-xenial-py3-clang5-android-ndk-r19c-x86_32-build',
    # PyTorch Android gradle
    'pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build-only-x86_32',
    # Pytorch iOS builds
    'pytorch-ios-10.2.1-x86_64_build',
    'pytorch-ios-10.2.1-arm64_build',

    # Pytorch backward compatibility check
    'pytorch-linux-backward-compatibility-check-test',

    # XLA
    'pytorch-xla-linux-xenial-py3.6-clang7',

    # GraphExecutor config jobs
    'pytorch-linux-xenial-py3.6-gcc5.4-ge_config_simple-test',
    'pytorch-linux-xenial-py3.6-gcc5.4-ge_config_legacy-test',

    # Other checks
    'pytorch-short-perf-test-gpu',
    'pytorch-python-doc-push',
    'pytorch-cpp-doc-push',
])

# Collection of jobs that are *temporarily* excluded from running on PRs.
# Use this if there is a long-running job breakage that we can't fix with a
# single revert.
skip_override = {
    # example entry:
    # 'pytorch-cpp-doc-push': "https://github.com/pytorch/pytorch/issues/<related issue>"
}

# Takes in commit message to analyze via stdin
#
# This script will query Git and attempt to determine if we should
# run the current CI job under question
#
# NB: Try to avoid hard-coding names here, so there's less place to update when jobs
# are updated/renamed
#
# Semantics in the presence of multiple tags:
#   - Let D be the set of default builds
#   - Let S be the set of explicitly specified builds
#   - Let O be the set of temporarily skipped builds
#   - Run S \/ (D - O)

parser = argparse.ArgumentParser()
parser.add_argument('build_environment')
args = parser.parse_args()

commit_msg = sys.stdin.read()

# Matches anything that looks like [foo ci] or [ci foo] or [foo test]
# or [test foo]
RE_MARKER = re.compile(r'\[(?:([^ \[\]]+) )?(?:ci|test)(?: ([^ \[\]]+))?\]')

markers = RE_MARKER.finditer(commit_msg)

for m in markers:
    if m.group(1) and m.group(2):
        print("Unrecognized marker: {}".format(m.group(0)))
        continue
    spec = m.group(1) or m.group(2)
    if spec is None:
        print("Unrecognized marker: {}".format(m.group(0)))
        continue
    if spec in args.build_environment or spec == 'all':
        print("Accepting {} due to commit marker {}".format(args.build_environment, m.group(0)))
        sys.exit(0)

skip_override_set = set(skip_override.keys())
should_run_set = default_set - skip_override_set
for spec in should_run_set:
    if spec in args.build_environment:
        print("Accepting {} as part of default set".format(args.build_environment))
        sys.exit(0)

print("Rejecting {}".format(args.build_environment))
for spec, issue in skip_override.items():
    if spec in args.build_environment:
        print("This job is temporarily excluded from running on PRs. Reason: {}".format(issue))
        break
sys.exit(1)
