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

function get_pr_change_files() {
  # The fetch may fail on Docker hosts, this fetch is necessary for GHA
  # accepts PR_NUMBER and extract filename as arguments
  set +e
  tmp_file=$(mktemp)
  wget -O "$tmp_file" "https://api.github.com/repos/pytorch/pytorch/pulls/$1/files"
  # this regex extracts the filename list according to the GITHUB REST API result.
  sed -n "s/.*\"filename\": \"\(.*\)\",/\1/p" "$tmp_file" | tee "$2"
  set -e
}

function get_bazel() {
  if [[ $(uname) == "Darwin" ]]; then
    # download bazel version
    curl https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-darwin-x86_64  -Lo tools/bazel
    # verify content
    echo '74d93848f0c9d592e341e48341c53c87e3cb304a54a2a1ee9cff3df422f0b23c  tools/bazel' | shasum -a 256 -c >/dev/null
  else
    # download bazel version
    curl https://ossci-linux.s3.amazonaws.com/bazel-4.2.1-linux-x86_64 -o tools/bazel
    # verify content
    echo '1a4f3a3ce292307bceeb44f459883859c793436d564b95319aacb8af1f20557c  tools/bazel' | shasum -a 256 -c >/dev/null
  fi

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

function clone_pytorch_xla() {
  if [[ ! -d ./xla ]]; then
    git clone --recursive https://github.com/pytorch/xla.git
  fi
}
