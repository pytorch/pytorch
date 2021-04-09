#!/bin/bash

# Common util **functions** that can be sourced in other scripts.

# NB: define this function before set -x, so that we don't
# pollute the log with a premature EXITED_USER_LAND ;)
function cleanup {
  # Note that if you've exited user land, then CI will conclude that
  # any failure is the CI's fault.  So we MUST only output this
  # string
  retcode=$?
  set +x
  if [ $retcode -eq 0 ]; then
    echo "EXITED_USER_LAND"
  fi
}

function assert_git_not_dirty() {
    # TODO: we should add an option to `build_amd.py` that reverts the repo to
    #       an unmodified state.
    if [[ "$BUILD_ENVIRONMENT" != *rocm* ]] && [[ "$BUILD_ENVIRONMENT" != *xla* ]] ; then
        git_status=$(git status --porcelain)
        if [[ $git_status ]]; then
            echo "Build left local git repository checkout dirty"
            echo "git status --porcelain:"
            echo "${git_status}"
            exit 1
        fi
    fi
}

function pip_install() {
  # retry 3 times
  # old versions of pip don't have the "--progress-bar" flag
  pip install --progress-bar off "$@" || pip install --progress-bar off "$@" || pip install --progress-bar off "$@" ||\
  pip install "$@" || pip install "$@" || pip install "$@"
}

function pip_uninstall() {
  # uninstall 2 times
  pip uninstall -y "$@" || pip uninstall -y "$@"
}

function get_exit_code() {
  set +e
  "$@"
  retcode=$?
  set -e
  return $retcode
}

function file_diff_from_base() {
  # The fetch may fail on Docker hosts, but it's not always necessary.
  set +e
  git fetch origin master --quiet
  set -e
  git diff --name-only "$(git merge-base origin/master HEAD)" > "$1"
}

function get_bazel() {
  # download bazel version
  wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-linux-x86_64 -O tools/bazel
  # verify content
  echo '753434f4fa730266cf5ce21d1fdd425e1e167dd9347ad3e8adc19e8c0d54edca  tools/bazel' | sha256sum --quiet -c

  chmod +x tools/bazel
}

function install_monkeytype {
  # Install MonkeyType
  pip_install MonkeyType
}

TORCHVISION_COMMIT=8a2dc6f22ac4389ccba8859aa1e1cb14f1ee53db

function install_torchvision() {
  # Check out torch/vision at Jun 11 2020 commit
  # This hash must match one in .jenkins/caffe2/test.sh
  pip_install --user "git+https://github.com/pytorch/vision.git@$TORCHVISION_COMMIT"
}

function checkout_install_torchvision() {
  git clone https://github.com/pytorch/vision
  pushd vision
  git checkout "$TORCHVISION_COMMIT"
  time python setup.py install
  popd
}
