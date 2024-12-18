#!/bin/bash

# Common util **functions** that can be sourced in other scripts.

# note: printf is used instead of echo to avoid backslash
# processing and to properly handle values that begin with a '-'.

log() { printf '%s\n' "$*"; }
error() { log "ERROR: $*" >&2; }
fatal() { error "$@"; exit 1; }

retry () {
    "$@" || (sleep 10 && "$@") || (sleep 20 && "$@") || (sleep 40 && "$@")
}

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
        git_status=$(git status --porcelain | grep -v '?? third_party' || true)
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

  # Convert the input arguments into an array
  local args=("$@")

  # Check if the first argument contains multiple paths separated by spaces
  if [[ "${args[0]}" == *" "* ]]; then
    # Split the string by spaces into an array
    IFS=' ' read -r -a paths <<< "${args[0]}"
    # Loop through each path and install individually
    for path in "${paths[@]}"; do
      echo "Installing $path"
      python3 -mpip install --no-index --no-deps "$path"
    done
  else
    # Loop through each argument and install individually
    for path in "${args[@]}"; do
      echo "Installing $path"
      python3 -mpip install --no-index --no-deps "$path"
    done
  fi
}


function pip_install() {
  # retry 3 times
  pip_install_pkg="python3 -m pip install --progress-bar off"
  ${pip_install_pkg} "$@" || \
    ${pip_install_pkg} "$@" || \
    ${pip_install_pkg} "$@"
}

function pip_uninstall() {
  # uninstall 2 times
  pip3 uninstall -y "$@" || pip3 uninstall -y "$@"
}

function get_exit_code() {
  set +e
  "$@"
  retcode=$?
  set -e
  return $retcode
}

function get_bazel() {
  # Download and use the cross-platform, dependency-free Python
  # version of Bazelisk to fetch the platform specific version of
  # Bazel to use from .bazelversion.
  retry curl --location --output tools/bazel \
    https://raw.githubusercontent.com/bazelbuild/bazelisk/v1.23.0/bazelisk.py
  shasum --algorithm=1 --check \
    <(echo '01df9cf7f08dd80d83979ed0d0666a99349ae93c  tools/bazel')
  chmod u+x tools/bazel
}

function install_monkeytype {
  # Install MonkeyType
  pip_install MonkeyType
}


function get_pinned_commit() {
  cat .github/ci_commit_pins/"${1}".txt
}

function install_torchaudio() {
  local commit
  commit=$(get_pinned_commit audio)
  if [[ "$1" == "cuda" ]]; then
    # TODO: This is better to be passed as a parameter from _linux-test workflow
    # so that it can be consistent with what is set in build
    TORCH_CUDA_ARCH_LIST="8.0;8.6" pip_install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${commit}"
  else
    pip_install --no-use-pep517 --user "git+https://github.com/pytorch/audio.git@${commit}"
  fi

}

function install_torchtext() {
  local data_commit
  local text_commit
  data_commit=$(get_pinned_commit data)
  text_commit=$(get_pinned_commit text)
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/data.git@${data_commit}"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/text.git@${text_commit}"
}

function install_torchvision() {
  local orig_preload
  local commit
  commit=$(get_pinned_commit vision)
  orig_preload=${LD_PRELOAD}
  if [ -n "${LD_PRELOAD}" ]; then
    # Silence dlerror to work-around glibc ASAN bug, see https://sourceware.org/bugzilla/show_bug.cgi?id=27653#c9
    echo 'char* dlerror(void) { return "";}'|gcc -fpic -shared -o "${HOME}/dlerror.so" -x c -
    LD_PRELOAD=${orig_preload}:${HOME}/dlerror.so
  fi
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${commit}"
  if [ -n "${LD_PRELOAD}" ]; then
    LD_PRELOAD=${orig_preload}
  fi
}

function install_tlparse() {
  pip_install --user "tlparse==0.3.30"
  PATH="$(python -m site --user-base)/bin:$PATH"
}

function install_torchrec_and_fbgemm() {
  local torchrec_commit
  torchrec_commit=$(get_pinned_commit torchrec)
  local fbgemm_commit
  fbgemm_commit=$(get_pinned_commit fbgemm)
  pip_uninstall torchrec-nightly
  pip_uninstall fbgemm-gpu-nightly
  pip_install setuptools-git-versioning scikit-build pyre-extensions

  # TODO (huydhn): I still have no clue on why sccache doesn't work with only fbgemm_gpu here, but it
  # seems to be an sccache-related issue
  if [[ "$IS_A100_RUNNER" == "1" ]]; then
    unset CMAKE_CUDA_COMPILER_LAUNCHER
    sudo mv /opt/cache/bin /opt/cache/bin-backup
  fi

  # See https://github.com/pytorch/pytorch/issues/106971
  CUDA_PATH=/usr/local/cuda-12.1 pip_install --no-use-pep517 --user "git+https://github.com/pytorch/FBGEMM.git@${fbgemm_commit}#egg=fbgemm-gpu&subdirectory=fbgemm_gpu"
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/torchrec.git@${torchrec_commit}"

  if [[ "$IS_A100_RUNNER" == "1" ]]; then
    export CMAKE_CUDA_COMPILER_LAUNCHER=/opt/cache/bin/sccache
    sudo mv /opt/cache/bin-backup /opt/cache/bin
  fi
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

function checkout_install_torchbench() {
  local commit
  commit=$(get_pinned_commit torchbench)
  git clone https://github.com/pytorch/benchmark torchbench
  pushd torchbench
  git checkout "$commit"

  if [ "$1" ]; then
    python install.py --continue_on_fail models "$@"
  else
    # Occasionally the installation may fail on one model but it is ok to continue
    # to install and test other models
    python install.py --continue_on_fail
  fi
  echo "Print all dependencies after TorchBench is installed"
  python -mpip freeze
  popd
}

function install_torchao() {
  local commit
  commit=$(get_pinned_commit torchao)
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/ao.git@${commit}"
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
