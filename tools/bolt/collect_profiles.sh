#!/bin/bash
# Collect LLVM BOLT profiles for the libraries of a torch wheel.
#
# The wheel must be BOLT-compatible but not BOLT-optimized: build it with
# USE_LLVM_BOLT=1 and LLVM_BOLT_PROFILES_DIR pointing to an empty directory
# (see cmake/bolt_profiles/README.md). Collection is split into three steps so
# that only the workload has to run on a GPU machine:
#
#   instrument <wheel> <out-dir>
#       Instrument every library of the wheel that was linked with
#       --emit-relocs.
#   run <instrumented-dir> <out-dir>
#       Copy the instrumented libraries over those of the installed wheel, run
#       tools/bolt/synthetic_workload.py and collect the per-process profiles.
#       The installed libraries are restored when the step exits.
#   finalize <wheel> <run-dir> <out-dir>
#       Merge the profiles of each library with merge-fdata, convert them to
#       YAML, validate them with a trial optimization that uses the options of
#       the build-time BOLT step, and package them as torch.tar.zst in the
#       layout of cmake/bolt_profiles/torch.tar.zst.
#
# Environment:
#   LLVM_BOLT_VERSION          LLVM release to take llvm-bolt from (default
#                              21.1.8, the version the manywheel images build
#                              in .ci/docker/common/install_llvm_tools.sh, so
#                              the profiles match the BOLT that consumes them)
#   LLVM_BOLT_HOME             use the llvm-bolt and merge-fdata of this LLVM
#                              install instead of downloading a release
#   BOLT_FDATA_DIR             directory the instrumented libraries write their
#                              profiles to (instrument, default /tmp/bolt-fdata)
#   BOLT_WORKLOAD_VARIANT      synthetic_workload.py --variant (run, default all)
#   BOLT_WORKLOAD_STEPS        synthetic_workload.py --steps (run, default 320000)
#   BOLT_WORKLOAD_WARMUP       synthetic_workload.py --warmup (run, default 50)
#   BOLT_WORKLOAD_MAX_SECONDS  synthetic_workload.py --max-seconds (run,
#                              default 0, no limit)
#   BOLT_VALIDATE              set to 0 to skip the trial optimization (finalize)

set -euo pipefail

LLVM_BOLT_VERSION="${LLVM_BOLT_VERSION:-21.1.8}"
BOLT_FDATA_DIR="${BOLT_FDATA_DIR:-/tmp/bolt-fdata}"
BOLT_WORKLOAD_VARIANT="${BOLT_WORKLOAD_VARIANT:-all}"
BOLT_WORKLOAD_STEPS="${BOLT_WORKLOAD_STEPS:-320000}"
BOLT_WORKLOAD_WARMUP="${BOLT_WORKLOAD_WARMUP:-50}"
BOLT_WORKLOAD_MAX_SECONDS="${BOLT_WORKLOAD_MAX_SECONDS:-0}"
BOLT_VALIDATE="${BOLT_VALIDATE:-1}"

# Options of the build-time optimization in torch_optimize_layout_if_enabled
# (cmake/public/utils.cmake). Keep in sync.
BOLT_OPTIMIZE_OPTIONS=(
  -lite -infer-stale-profile
  -reorder-blocks=ext-tsp -reorder-functions=cdsort
  -split-functions -split-all-cold -split-eh -dyno-stats
  --update-debug-sections
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
  echo "[collect_profiles] $*"
}

die() {
  echo "[collect_profiles] ERROR: $*" >&2
  exit 1
}

usage() {
  echo "Usage: $0 instrument <wheel> <out-dir>" >&2
  echo "       $0 run <instrumented-dir> <out-dir>" >&2
  echo "       $0 finalize <wheel> <run-dir> <out-dir>" >&2
  exit 1
}

# Run a command with its output in a log file; show the tail on failure.
run_logged() {
  local logfile="$1"
  shift
  if ! "$@" > "${logfile}" 2>&1; then
    tail -n 50 "${logfile}" >&2
    die "command failed, see ${logfile}: $*"
  fi
}

# Create an empty output directory and set OUTPUT_DIR to its absolute path.
make_output_dir() {
  mkdir -p "$1"
  OUTPUT_DIR="$(cd "$1" && pwd)"
  if [[ -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
    die "output directory ${OUTPUT_DIR} is not empty"
  fi
  mkdir -p "${OUTPUT_DIR}/logs"
}

# Extract torch/lib/*.so of the wheel $1 into ${WORK_DIR}/original.
extract_libraries() {
  [[ -f "$1" ]] || die "no such wheel: $1"
  unzip -q -j -o "$1" 'torch/lib/*.so' -d "${WORK_DIR}/original"
}

# Set LLVM_BOLT and MERGE_FDATA, downloading an LLVM release if needed.
setup_llvm() {
  local llvm_home llvm_arch archive url
  if [[ -n "${LLVM_BOLT_HOME:-}" ]]; then
    llvm_home="${LLVM_BOLT_HOME}"
  else
    case "$(uname -m)" in
      aarch64) llvm_arch=ARM64 ;;
      x86_64) llvm_arch=X64 ;;
      *) die "unsupported architecture: $(uname -m)" ;;
    esac
    # Every release ships .tar.xz; newer ones also ship .tar.zst.
    archive="LLVM-${LLVM_BOLT_VERSION}-Linux-${llvm_arch}.tar.xz"
    url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_BOLT_VERSION}/${archive}"
    log "Downloading ${url}"
    curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 -o "${WORK_DIR}/${archive}" "${url}"
    llvm_home="${WORK_DIR}/llvm"
    mkdir -p "${llvm_home}"
    # Only the BOLT tools and the instrumentation runtime are needed. llvm-bolt
    # looks the runtime up in ../lib relative to its own location.
    tar -x --xz -f "${WORK_DIR}/${archive}" -C "${llvm_home}" \
      --strip-components=1 --wildcards \
      '*/bin/llvm-bolt' '*/bin/merge-fdata' '*/lib/libbolt_rt_instr.a'
    rm -f "${WORK_DIR:?}/${archive}"
  fi
  LLVM_BOLT="${llvm_home}/bin/llvm-bolt"
  MERGE_FDATA="${llvm_home}/bin/merge-fdata"
  "${LLVM_BOLT}" --version | tee "${OUTPUT_DIR}/logs/llvm-bolt-version.txt"
}

# Check that the libraries in directory $1 are the ones listed in sha256sums
# file $2, i.e. the libraries that were instrumented.
verify_libraries() {
  (cd "$1" && sha256sum --check --quiet "$2") \
    || die "the libraries in $1 differ from the instrumented ones"
}

cmd_instrument() {
  [[ $# -eq 2 ]] || usage
  make_output_dir "$2"
  mkdir -p "${OUTPUT_DIR}/lib"
  extract_libraries "$1"

  # BOLT needs the static relocations that --emit-relocs keeps in .rela.text.
  # The build links only the libraries it would BOLT-optimize with it.
  local path lib sections
  local -a libs=()
  for path in "${WORK_DIR}"/original/lib*.so; do
    lib="$(basename "${path}")"
    sections="$(readelf -S --wide "${path}")"
    if [[ "${sections}" != *" .rela.text "* ]]; then
      log "Skipping ${lib}: no .rela.text"
      continue
    fi
    if [[ "${sections}" == *" .note.bolt_info "* ]]; then
      die "${lib} is already BOLT-optimized; build with an empty LLVM_BOLT_PROFILES_DIR"
    fi
    libs+=("${lib}")
  done
  [[ ${#libs[@]} -gt 0 ]] || die "no library in $1 has .rela.text; build it with USE_LLVM_BOLT=1"
  log "Libraries to profile: ${libs[*]}"

  setup_llvm
  for lib in "${libs[@]}"; do
    log "Instrumenting ${lib}"
    # Every process that loads the library writes <stem>.<pid>.fdata to
    # BOLT_FDATA_DIR at exit. The path is compiled into the library.
    run_logged "${OUTPUT_DIR}/logs/instrument-${lib%.so}.txt" \
      "${LLVM_BOLT}" "${WORK_DIR}/original/${lib}" -instrument \
      --instrumentation-file-append-pid \
      "--instrumentation-file=${BOLT_FDATA_DIR}/${lib%.so}" \
      -o "${OUTPUT_DIR}/lib/${lib}"
  done
  (cd "${WORK_DIR}/original" && sha256sum "${libs[@]}") > "${OUTPUT_DIR}/sha256sums"
  echo "${BOLT_FDATA_DIR}" > "${OUTPUT_DIR}/fdata-dir"
  log "Instrumented libraries written to ${OUTPUT_DIR}/lib"
}

cmd_run() {
  [[ $# -eq 2 ]] || usage
  local instrumented_dir fdata_dir lib
  instrumented_dir="$(cd "$1" && pwd)"
  make_output_dir "$2"
  mkdir -p "${OUTPUT_DIR}/raw"
  fdata_dir="$(cat "${instrumented_dir}/fdata-dir")"
  mkdir -p "${fdata_dir}"
  [[ -z "$(ls -A "${fdata_dir}")" ]] || die "${fdata_dir} is not empty"

  # Import torch from site-packages, never from a source checkout in the cwd.
  cd "${WORK_DIR}"
  TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  log "torch libraries: ${TORCH_LIB_DIR}"
  verify_libraries "${TORCH_LIB_DIR}" "${instrumented_dir}/sha256sums"

  local -a libs
  mapfile -t libs < <(awk '{print $2}' "${instrumented_dir}/sha256sums")
  for lib in "${libs[@]}"; do
    cp "${TORCH_LIB_DIR}/${lib}" "${WORK_DIR}/original/${lib}"
    RESTORE_LIBS+=("${lib}")
    cp "${instrumented_dir}/lib/${lib}" "${TORCH_LIB_DIR}/${lib}"
  done

  log "Running the synthetic workload on the instrumented libraries"
  python3 "${SCRIPT_DIR}/synthetic_workload.py" \
    --variant "${BOLT_WORKLOAD_VARIANT}" \
    --steps "${BOLT_WORKLOAD_STEPS}" \
    --warmup "${BOLT_WORKLOAD_WARMUP}" \
    --max-seconds "${BOLT_WORKLOAD_MAX_SECONDS}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/workload.txt"
  restore_libraries

  local -a raw
  for lib in "${libs[@]}"; do
    raw=("${fdata_dir}/${lib%.so}".*.fdata)
    [[ -e "${raw[0]}" ]] || die "no profile was written for ${lib}; did the workload exit normally?"
    log "${lib}: ${#raw[@]} profile(s)"
    mv "${raw[@]}" "${OUTPUT_DIR}/raw/"
  done
  cp "${instrumented_dir}/sha256sums" "${OUTPUT_DIR}/sha256sums"
  cp "${instrumented_dir}/logs/"* "${OUTPUT_DIR}/logs/"
  {
    python3 - << 'EOF'
import sysconfig

import torch

print(f"torch: {torch.__version__} (git {torch.version.git_version})")
print(f"cuda: {torch.version.cuda}")
print(f"python: {sysconfig.get_config_var('SOABI')}")
print(f"device: {torch.cuda.get_device_name(0)}")
EOF
    echo "workload: --variant ${BOLT_WORKLOAD_VARIANT} --steps ${BOLT_WORKLOAD_STEPS} --warmup ${BOLT_WORKLOAD_WARMUP} --max-seconds ${BOLT_WORKLOAD_MAX_SECONDS}"
    grep -E '^\[bolt-synth\] (variants=|done |stopping |WARNING: dropping)' \
      "${OUTPUT_DIR}/logs/workload.txt" || true
  } > "${OUTPUT_DIR}/run-info.txt"
  log "Profiles written to ${OUTPUT_DIR}/raw"
}

cmd_finalize() {
  [[ $# -eq 3 ]] || usage
  local run_dir lib stem opt_log
  run_dir="$(cd "$2" && pwd)"
  make_output_dir "$3"
  mkdir -p "${OUTPUT_DIR}"/{fdata,yaml}
  extract_libraries "$1"
  verify_libraries "${WORK_DIR}/original" "${run_dir}/sha256sums"
  cp "${run_dir}/logs/"* "${OUTPUT_DIR}/logs/"
  setup_llvm

  local -a libs raw
  mapfile -t libs < <(awk '{print $2}' "${run_dir}/sha256sums")
  for lib in "${libs[@]}"; do
    stem="${lib%.so}"
    raw=("${run_dir}/raw/${stem}".*.fdata)
    log "Merging ${#raw[@]} profile(s) of ${lib}"
    run_logged "${OUTPUT_DIR}/logs/merge-fdata-${stem}.txt" \
      "${MERGE_FDATA}" "${raw[@]}" -o "${OUTPUT_DIR}/fdata/${stem}.fdata"
    [[ -s "${OUTPUT_DIR}/fdata/${stem}.fdata" ]] || die "merge-fdata produced an empty profile for ${lib}"

    # The build-time step consumes YAML profiles named lib<target>.yaml.
    # Unlike fdata, they carry block hashes that let -infer-stale-profile
    # match profiles to binaries built from slightly different sources.
    log "Converting the ${lib} profile to YAML"
    run_logged "${OUTPUT_DIR}/logs/convert-${stem}.txt" \
      "${LLVM_BOLT}" "${WORK_DIR}/original/${lib}" -o /dev/null \
      "-data=${OUTPUT_DIR}/fdata/${stem}.fdata" -w "${OUTPUT_DIR}/yaml/${stem}.yaml"

    if [[ "${BOLT_VALIDATE}" == "1" ]]; then
      log "Trial optimization of ${lib}"
      run_logged "${OUTPUT_DIR}/logs/optimize-${stem}.txt" \
        "${LLVM_BOLT}" "${WORK_DIR}/original/${lib}" -o "${WORK_DIR}/bolted.so" \
        "-data=${OUTPUT_DIR}/yaml/${stem}.yaml" "${BOLT_OPTIMIZE_OPTIONS[@]}"
      rm -f "${WORK_DIR}/bolted.so"
    fi
  done

  (
    cd "${OUTPUT_DIR}/yaml"
    if command -v cmake > /dev/null; then
      cmake -E tar cf ../torch.tar.zst --zstd -- *.yaml
    else
      tar --zstd -cf ../torch.tar.zst -- *.yaml
    fi
  )

  {
    cat "${run_dir}/run-info.txt"
    echo "llvm-bolt: $(grep -m1 'LLVM version' "${OUTPUT_DIR}/logs/llvm-bolt-version.txt")"
    for lib in "${libs[@]}"; do
      stem="${lib%.so}"
      echo
      echo "${lib}:"
      echo "  sha256: $(grep " ${lib}\$" "${run_dir}/sha256sums" | cut -d' ' -f1)"
      echo "  build-id: $(readelf -n "${WORK_DIR}/original/${lib}" | awk '/Build ID/ {print $3}')"
      echo "  merged profiles: $(compgen -G "${run_dir}/raw/${stem}.*.fdata" | wc -l)"
      grep -m1 'have non-empty execution profile' "${OUTPUT_DIR}/logs/convert-${stem}.txt" \
        | sed 's/^BOLT-INFO: /  /' || true
      opt_log="${OUTPUT_DIR}/logs/optimize-${stem}.txt"
      if [[ -f "${opt_log}" ]]; then
        grep -m1 'could not be optimized' "${opt_log}" | sed 's/^BOLT-INFO: /  /' || true
        grep -m1 -o 'function CFG discontinuity.*' "${opt_log}" | sed 's/^/  /' || true
        awk '/: taken branches \(/ {print "  taken branches after BOLT: " $1 " " $5; exit}' "${opt_log}"
      fi
    done
  } > "${OUTPUT_DIR}/manifest.txt"

  cat "${OUTPUT_DIR}/manifest.txt"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
      echo '### BOLT profiles'
      echo '```'
      cat "${OUTPUT_DIR}/manifest.txt"
      echo '```'
    } >> "${GITHUB_STEP_SUMMARY}"
  fi
  log "Profiles written to ${OUTPUT_DIR}"
}

# Libraries of the installed torch that cmd_run replaced, restored on exit.
RESTORE_LIBS=()
restore_libraries() {
  local lib
  for lib in "${RESTORE_LIBS[@]}"; do
    cp "${WORK_DIR}/original/${lib}" "${TORCH_LIB_DIR}/${lib}"
  done
  RESTORE_LIBS=()
}

cleanup() {
  restore_libraries
  cd /
  rm -rf "${WORK_DIR}"
}

[[ $# -ge 1 ]] || usage
command="$1"
shift
WORK_DIR="$(mktemp -d)"
mkdir -p "${WORK_DIR}/original"
trap cleanup EXIT
case "${command}" in
  instrument) cmd_instrument "$@" ;;
  run) cmd_run "$@" ;;
  finalize) cmd_finalize "$@" ;;
  *) usage ;;
esac
