#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

GPU_ARCH="${1:?Usage: pytorch_build_test.sh <cpu|cuda|rocm> [all|wheel|test]}"
MODE="${2:-all}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${MODE}" != "all" && "${MODE}" != "wheel" && "${MODE}" != "test" ]]; then
  echo "ERROR: unknown mode '${MODE}' (expected all|wheel|test)" >&2
  exit 1
fi

# Save kineto directory path before cloning PyTorch
KINETO_DIR=$(pwd)
echo "====: Kineto directory: ${KINETO_DIR}"

# Clone PyTorch as a sibling of the Kineto checkout, never inside it. Below we
# replace PyTorch's third_party/kineto with a symlink back to KINETO_DIR. If the
# PyTorch tree lived inside KINETO_DIR, that symlink would point at a directory
# that contains the clone, forming an endless path cycle.
PYTORCH_DIR="$(dirname "${KINETO_DIR}")/pytorch"

maybe_enable_sccache() {
  # Enable sccache so PyTorch object files persist across CI runs. Most Kineto
  # PRs touch only Kineto, so the bulk of PyTorch's objects are cache hits on
  # warm runs. Whether the runner can reach the S3 cache is architecture
  # specific (only the AWS-hosted runners can), so each config_<arch>.sh opts
  # in via KINETO_USE_SCCACHE.
  if [[ "${KINETO_USE_SCCACHE:-0}" != "1" ]]; then
    return 0
  fi

  # Note that the sscache binary we use is hard-coded to be Linux specific.
  # That's safe because we only have Linux specific callers. Also note that
  # GPU_ARCH does not determine a CPU architecture, so we determine that here.
  case "$(uname -m)" in
    x86_64)        SCCACHE_ARCH=x86_64 ;;
    aarch64|arm64) SCCACHE_ARCH=aarch64 ;;
    *) SCCACHE_ARCH="" ;;
  esac

  if [[ -z "${SCCACHE_ARCH}" ]]; then
    echo "====: Unsupported arch for sccache: $(uname -m); building uncached" >&2
  else
    SCCACHE_VERSION="v0.8.2"
    SCCACHE_PKG="sccache-${SCCACHE_VERSION}-${SCCACHE_ARCH}-unknown-linux-musl"
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_PKG}.tar.gz" | tar -xz -C /tmp
    install -m755 "/tmp/${SCCACHE_PKG}/sccache" /usr/local/bin/sccache

    export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2
    export SCCACHE_REGION=us-east-1
    export SCCACHE_S3_KEY_PREFIX=kineto
    export SCCACHE_IDLE_TIMEOUT=0
    export SCCACHE_ERROR_LOG=/tmp/sccache_error.log

    # Route compilers through sccache only if the cache backend actually
    # starts. A configured-but-unreachable bucket otherwise makes every
    # compile fail, so this keeps a cache problem from breaking the build.
    if sccache --start-server; then
      sccache --zero-stats || true
      export CMAKE_C_COMPILER_LAUNCHER=sccache
      export CMAKE_CXX_COMPILER_LAUNCHER=sccache
      export CMAKE_CUDA_COMPILER_LAUNCHER=sccache
      export CMAKE_HIP_COMPILER_LAUNCHER=sccache
      echo "====: Enabled sccache (${SCCACHE_VERSION}, ${SCCACHE_ARCH})"
    else
      echo "====: sccache cache unreachable; building without it" >&2
    fi
  fi
}

run_profiler_tests() {
  # Run from the PyTorch clone. Its test/conftest.py imports sibling helpers
  # such as pytest_shard_custom, which only resolve because pytest's default
  # import mode prepends the conftest's directory to sys.path.
  pushd "${PYTORCH_DIR}"

  # Download PyTorch's dynamic disabled tests list from S3. This is generated every
  # 15 minutes from DISABLED GitHub Issues in pytorch/pytorch, enabling automatic
  # skipping of known-broken/flaky tests without hardcoded deselections.
  # The function downloads, processes (converts format and filters re-enabled issues),
  # and caches the result to .pytorch-disabled-tests.json.
  python -c "from tools.stats.import_test_stats import get_disabled_tests; get_disabled_tests('.')"
  export DISABLED_TESTS_FILE=.pytorch-disabled-tests.json
  echo "====: Downloaded disabled tests list"

  # The deselected tests array is sourced from the architecture config above.
  local deselect_args=()
  local t
  for t in "${DESELECTED_TESTS[@]}"; do
    deselect_args+=(--deselect="$t")
  done

  # After a wheel install the clone's torch/ is Python sources with no compiled
  # extensions, so anything that puts the clone on sys.path shadows the real
  # torch. Two paths would: `python -m pytest` prepends the current directory,
  # hence the console script; and the tests that spawn `python -c` inherit that
  # same behaviour, hence PYTHONSAFEPATH (honoured by the children too).
  local pytest_cmd=(python -m pytest)
  if [[ "${MODE}" == "test" ]]; then
    pytest_cmd=(env PYTHONSAFEPATH=1 pytest)
  fi

  # Run PyTorch profiler tests under a per-test timeout so a hang fails that one
  # test instead of consuming the whole job's timeout. Use the signal method, not
  # thread: the thread method's watchdog thread is captured by the profiler and
  # inflates the thread counts that some profiler tests assert on. The tradeoff is
  # that signal cannot interrupt a hang holding the GIL in native code.
  pip install pytest pytest-timeout
  "${pytest_cmd[@]}" test/profiler/ -v --timeout=300 --timeout-method=signal \
    "${deselect_args[@]}"
  echo "====: Ran PyTorch profiler tests"
  popd
}

# Both ROCm jobs clone independently. Pin the test job to the SHA the wheel
# was built from so a viable/strict move in between cannot mix new tests with
# an older torch (or the reverse).
if [[ "${MODE}" == "test" ]]; then
  sha_file="${RUNNER_ARTIFACT_DIR:-/artifacts}/pytorch-sha.txt"
  if [[ ! -f "${sha_file}" ]]; then
    echo "ERROR: ${sha_file} missing; the wheel job must record the PyTorch SHA it built" >&2
    ls -la "${RUNNER_ARTIFACT_DIR:-/artifacts}" >&2 || true
    exit 1
  fi
  pytorch_ref="$(tr -d '[:space:]' < "${sha_file}")"
  if [[ ! "${pytorch_ref}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: invalid PyTorch SHA in ${sha_file}: '${pytorch_ref}'" >&2
    exit 1
  fi
  # Fetch the recorded commit by hash. Cloning --branch viable/strict would
  # miss it if that ref was force-moved after the wheel job.
  mkdir -p "${PYTORCH_DIR}"
  git -C "${PYTORCH_DIR}" init
  git -C "${PYTORCH_DIR}" remote add origin https://github.com/pytorch/pytorch.git
  git -C "${PYTORCH_DIR}" fetch --recurse-submodules origin "${pytorch_ref}"
  git -C "${PYTORCH_DIR}" checkout --detach FETCH_HEAD
  git -C "${PYTORCH_DIR}" submodule update --init --recursive
else
  git clone --recursive --branch viable/strict https://github.com/pytorch/pytorch.git "${PYTORCH_DIR}"
fi
echo "====: Cloned PyTorch $(git -C "${PYTORCH_DIR}" rev-parse HEAD)"

# Load architecture-specific build env vars and deselected tests
# shellcheck source=/dev/null
source "${SCRIPTS_DIR}/config_${GPU_ARCH}.sh"

if [[ "${MODE}" != "test" ]]; then
  rm -rf "${PYTORCH_DIR}/third_party/kineto"
  ln -s "${KINETO_DIR}" "${PYTORCH_DIR}/third_party/kineto"
  echo "====: Linked PR version of Kineto to PyTorch (${KINETO_DIR} -> third_party/kineto)"

  maybe_enable_sccache

  pushd "${PYTORCH_DIR}"
  pip install -r requirements.txt

  # Hipify PyTorch source code for ROCm build
  if [[ "${GPU_ARCH}" == "rocm" ]]; then
    python tools/amd_build/build_amd.py
    echo "====: Hipified PyTorch source for ROCm"
  fi

  if [[ "${MODE}" == "wheel" ]]; then
    python -m build --wheel --no-isolation
    echo "====: Built PyTorch wheel"
    sccache --show-stats || true
    mkdir -p "${KINETO_DIR}/artifacts-to-be-uploaded"
    cp -v dist/*.whl "${KINETO_DIR}/artifacts-to-be-uploaded/"
    git rev-parse HEAD > "${KINETO_DIR}/artifacts-to-be-uploaded/pytorch-sha.txt"
    echo "====: Copied wheel and PyTorch SHA $(cat "${KINETO_DIR}/artifacts-to-be-uploaded/pytorch-sha.txt") to artifacts-to-be-uploaded"
    popd
    exit 0
  fi

  python -m pip install --no-build-isolation -v -e .
  echo "====: Built PyTorch from source"
  # Surface cache hit rates so warm-vs-cold runs are diagnosable from the log.
  # Harmless when sccache was not enabled (no server running).
  sccache --show-stats || true
  run_profiler_tests
  popd
  exit 0
fi

ARTIFACT_ROOT="${RUNNER_ARTIFACT_DIR:-/artifacts}"
shopt -s nullglob
wheels=("${ARTIFACT_ROOT}"/*.whl)
shopt -u nullglob
if [[ "${#wheels[@]}" -eq 0 ]]; then
  echo "ERROR: no torch wheel in ${ARTIFACT_ROOT}" >&2
  ls -la "${ARTIFACT_ROOT}" >&2 || true
  exit 1
fi
python -m pip install --no-build-isolation -v "${wheels[0]}"
echo "====: Installed $(basename "${wheels[0]}")"

# The wheel brings torch but not its test-time imports (numpy, expecttest,
# sympy, ...) that torch.testing._internal pulls in. The build job gets these
# from the same file.
(
  cd "${PYTORCH_DIR}"
  pip install -r requirements.txt
)
echo "====: Installed PyTorch test requirements"

# Confirm the tests will exercise the wheel built from this PR, not the clone's
# uncompiled torch/ sources. Checked from outside the clone, where `python -c`
# would otherwise put the clone on sys.path.
python -c "
import pathlib, torch
p = pathlib.Path(torch.__file__).resolve()
print(f'====: torch.__file__={p}')
clone = pathlib.Path(r'${PYTORCH_DIR}').resolve()
assert clone not in p.parents and p != clone, p
"

run_profiler_tests
