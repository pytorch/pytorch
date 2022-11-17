#!/bin/bash

# Common util **functions** that can be sourced in other scripts.

# note: printf is used instead of echo to avoid backslash
# processing and to properly handle values that begin with a '-'.

log() { printf '%s\n' "$*"; }
error() { log "ERROR: $*" >&2; }
fatal() { error "$@"; exit 1; }

# compositional trap taken from https://stackoverflow.com/a/7287873/23845
# appends a command to a trap
#
# - 1st arg:  code to add
# - remaining args:  names of traps to modify
#
trap_add() {
    trap_add_cmd=$1; shift || fatal "${FUNCNAME[0]} usage error"
    for trap_add_name in "$@"; do
        trap -- "$(
            # helper fn to get existing trap command from output
            # of trap -p
            extract_trap_cmd() { printf '%s\n' "$3"; }
            # print existing trap command with newline
            eval "extract_trap_cmd $(trap -p "${trap_add_name}")"
            # print the new trap command
            printf '%s\n' "${trap_add_cmd}"
        )" "${trap_add_name}" \
            || fatal "unable to add to trap ${trap_add_name}"
    done
}
# set the trace attribute for the above function.  this is
# required to modify DEBUG or RETURN traps because functions don't
# inherit them unless the trace attribute is set
declare -f -t trap_add

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

function pip_install_whl() {
  # This is used to install PyTorch and other build artifacts wheel locally
  # without using any network connection
  python3 -mpip install --no-index --no-deps "$@"
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


function get_pinned_commit() {
  cat .github/ci_commit_pins/"${1}".txt
}

function install_torchvision() {
  local commit
  commit=$(get_pinned_commit vision)
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${commit}"
}

function checkout_install_torchvision() {
  local commit
  commit=$(get_pinned_commit vision)
  git clone https://github.com/pytorch/vision
  pushd vision
  git checkout "${commit}"
  time python setup.py install
  popd
}

function clone_pytorch_xla() {
  if [[ ! -d ./xla ]]; then
    git clone --recursive --quiet https://github.com/pytorch/xla.git
    pushd xla
    # pin the xla hash so that we don't get broken by changes to xla
    git checkout "$(cat ../.github/ci_commit_pins/xla.txt)"
    git submodule sync
    git submodule update --init --recursive
    popd
  fi
}

function install_filelock() {
  pip_install filelock
}

function install_triton() {
  local commit
  if [[ "${TEST_CONFIG}" == *rocm* ]]; then
    echo "skipping triton due to rocm"
  else
    commit=$(get_pinned_commit triton)
    pip_install --user "git+https://github.com/openai/triton@${commit}#subdirectory=python"
    pip_install --user jinja2
  fi
}

function setup_torchdeploy_deps(){
  conda install -y cmake
  conda install -y -c conda-forge libpython-static=3.10
  local CC
  local CXX
  CC="$(which gcc)"
  CXX="$(which g++)"
  export CC
  export CXX
  pip install --upgrade pip
}

function checkout_install_torchdeploy() {
  local commit
  setup_torchdeploy_deps
  pushd ..
  git clone --recurse-submodules https://github.com/pytorch/multipy.git
  pushd multipy
  python multipy/runtime/example/generate_examples.py
  pip install -e . --install-option="--cudatests"
  popd
  popd
}

function test_torch_deploy(){
 pushd ..
 pushd multipy
 ./multipy/runtime/build/test_deploy
 ./multipy/runtime/build/test_deploy_gpu
 popd
 popd
}

function install_huggingface() {
  local commit
  commit=$(get_pinned_commit huggingface)
  pip_install pandas
  pip_install scipy
  pip_install "git+https://github.com/huggingface/transformers.git@${commit}#egg=transformers"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)
  pip_install pandas
  pip_install scipy
  pip_install "git+https://github.com/rwightman/pytorch-image-models@${commit}"
}

function checkout_install_torchbench() {
  local commit
  commit=$(get_pinned_commit torchbench)
  git clone https://github.com/pytorch/benchmark torchbench
  pushd torchbench
  git checkout "${commit}"
  python install.py
  pip_install gym==0.25.2  # workaround issue in 0.26.0
  popd
}

function test_functorch() {
  python test/run_test.py --functorch --verbose
}

function print_sccache_stats() {
  echo 'PyTorch Build Statistics'
  sccache --show-stats

  if [[ -n "${OUR_GITHUB_JOB_ID}" ]]; then
    sccache --show-stats --stats-format json | jq .stats \
      > "sccache-stats-${BUILD_ENVIRONMENT}-${OUR_GITHUB_JOB_ID}.json"
  else
    echo "env var OUR_GITHUB_JOB_ID not set, will not write sccache stats to json"
  fi
}
