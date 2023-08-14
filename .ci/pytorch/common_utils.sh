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
  # Download and use the cross-platform, dependency-free Python
  # version of Bazelisk to fetch the platform specific version of
  # Bazel to use from .bazelversion.
  retry curl --location --output tools/bazel \
    https://raw.githubusercontent.com/bazelbuild/bazelisk/v1.16.0/bazelisk.py
  shasum --algorithm=1 --check \
    <(echo 'd4369c3d293814d3188019c9f7527a948972d9f8  tools/bazel')
  chmod u+x tools/bazel
}

# This function is bazel specific because of the bug
# in the bazel that requires some special paths massaging
# as a workaround. See
# https://github.com/bazelbuild/bazel/issues/10167
function install_sccache_nvcc_for_bazel() {
  sudo mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc-real

  # Write the `/usr/local/cuda/bin/nvcc`
  cat << EOF | sudo tee /usr/local/cuda/bin/nvcc
#!/bin/sh
if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache /usr/local/cuda/bin/nvcc "\$@"
else
  exec external/local_cuda/cuda/bin/nvcc-real "\$@"
fi
EOF

  sudo chmod +x /usr/local/cuda/bin/nvcc
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
  local commit
  commit=$(get_pinned_commit vision)
  pip_install --no-use-pep517 --user "git+https://github.com/pytorch/vision.git@${commit}"
}

function install_numpy_pytorch_interop() {
  local commit
  commit=$(get_pinned_commit numpy_pytorch_interop)
  # TODO: --no-use-pep517 will result in failure.
  pip_install --user "git+https://github.com/Quansight-Labs/numpy_pytorch_interop.git@${commit}"
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

function checkout_install_torchdeploy() {
  local commit
  commit=$(get_pinned_commit multipy)
  pushd ..
  git clone --recurse-submodules https://github.com/pytorch/multipy.git
  pushd multipy
  git checkout "${commit}"
  python multipy/runtime/example/generate_examples.py
  BUILD_CUDA_TESTS=1 pip install -e .
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
  local version
  version=$(get_pinned_commit huggingface)
  pip_install pandas
  pip_install scipy
  pip_install z3-solver
  pip_install "transformers==${version}"
}

function install_timm() {
  local commit
  commit=$(get_pinned_commit timm)
  pip_install pandas
  pip_install scipy
  pip_install z3-solver
  pip_install "git+https://github.com/rwightman/pytorch-image-models@${commit}"
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
  popd
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
